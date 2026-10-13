from __future__ import annotations

import cv2
import numpy as np

from .noise import smoothstep


class TextRegionDetector:
    """Finds text-line regions without OCR so original glyphs can be restored.

    Pipeline: morphological gradient → adaptive threshold → wide horizontal
    closing (letters fuse into lines) → component heuristics on the rotated
    minimum-area box (so diagonal watermarks pass the aspect test). The kept
    blobs are then intersected with a high-local-contrast overlay AND a
    glyph-tone gate (near-white or near-black pixels only), so only the
    actual glyph strokes stay protected — nearby boats, wakes or rock texture
    that fused into the same blob still gets damaged. Returns a feathered
    float mask in [0, 1] or None.
    """

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        height, width = image.shape[:2]
        if min(height, width) < 96:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gradient = cv2.morphologyEx(
            gray, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)
        )
        binary = cv2.adaptiveThreshold(
            gradient,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=31,
            C=9.0,
        )

        close_width = max(12, width // 48)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (close_width, 3)),
        )
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        )

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros((height, width), dtype=np.uint8)
        for contour in contours:
            if not self._is_text_line(contour, binary, width, height):
                continue
            cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

        covered = int(np.count_nonzero(mask))
        if covered < 150:
            return None
        if covered / float(height * width) > 0.45:
            return None

        grow = max(4, round(min(height, width) * 0.006))
        kernel = np.ones((grow * 2 + 1, grow * 2 + 1), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)
        sigma = max(1.2, min(height, width) * 0.0018)
        feathered = cv2.GaussianBlur(mask, (0, 0), sigma).astype(np.float32) / 255.0

        background = cv2.GaussianBlur(
            gray, (0, 0), max(4.0, min(height, width) * 0.02)
        )
        flat_size = min(64, max(15, round(min(height, width) * 0.06))) | 1
        flat_kernel = np.ones((flat_size, flat_size), dtype=np.uint8)
        cleared_bright = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, flat_kernel)
        cleared_dark = cv2.morphologyEx(gray, cv2.MORPH_OPEN, flat_kernel)
        contrast = np.maximum.reduce(
            [
                cv2.absdiff(gray, background),
                cv2.absdiff(gray, cleared_bright),
                cv2.absdiff(gray, cleared_dark),
            ]
        ).astype(np.float32)
        dense = cv2.GaussianBlur(smoothstep(contrast, 30.0, 70.0), (0, 0), 1.2)
        gray_f = gray.astype(np.float32)
        bright_glyph = smoothstep(gray_f, 150.0, 205.0)
        dark_glyph = smoothstep(90.0 - gray_f, 10.0, 40.0)
        tone = np.maximum(bright_glyph, dark_glyph)
        overlay = (dense >= 0.4).astype(np.float32) * tone
        overlay = (overlay >= 0.5).astype(np.float32)
        return (feathered * overlay).astype(np.float32)

    @staticmethod
    def _is_text_line(
        contour: np.ndarray,
        binary: np.ndarray,
        width: int,
        height: int,
    ) -> bool:
        if cv2.contourArea(contour) < 120:
            return False

        (_center, sides, _angle) = cv2.minAreaRect(contour)
        long_side = float(max(sides))
        short_side = float(min(sides))
        if short_side < 6.0 or long_side < 16.0:
            return False
        if long_side / short_side < 1.6:
            return False
        if short_side > 0.45 * min(height, width):
            return False
        if long_side > 0.98 * max(height, width):
            return False
        if long_side * short_side > 0.55 * width * height:
            return False

        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < 16 or box_height < 9:
            return False
        window = binary[y : y + box_height, x : x + box_width]
        density = float(np.count_nonzero(window)) / float(box_width * box_height)
        return 0.04 <= density <= 0.92


class WatermarkStamp:
    """Stamps the literal WATERMARK graphic, hard-edged and layered on top.

    Used whenever the upload contains no detectable text of its own, so the
    joke output always carries a perfectly sharp watermark.
    """

    TEXT = "WATERMARK"

    def __init__(self, *, angle: float = -20.0, opacity: float = 0.78) -> None:
        self._angle = angle
        self._opacity = opacity

    def stamp(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        (base_width, _base_height), _ = cv2.getTextSize(self.TEXT, font, 1.0, 2)
        target_width = max(int(width * 0.72), base_width)
        scale = max(target_width / max(base_width, 1), 0.6)
        thickness = max(2, int(round(scale * 0.45)))

        (text_width, text_height), _ = cv2.getTextSize(self.TEXT, font, scale, thickness)
        origin = (
            int(width * 0.5 - text_width * 0.5),
            int(height * 0.5 + text_height * 0.5),
        )

        layer = np.zeros((height, width), dtype=np.uint8)
        cv2.putText(
            layer,
            self.TEXT,
            origin,
            font,
            scale,
            255,
            thickness,
            lineType=cv2.LINE_AA,
        )

        matrix = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), self._angle, 1.0)
        rotated = cv2.warpAffine(
            layer,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        outline_size = max(3, round(min(height, width) * 0.004) | 1)
        outline = cv2.dilate(
            rotated, np.ones((outline_size, outline_size), dtype=np.uint8)
        )

        alpha = (rotated.astype(np.float32) / 255.0) * self._opacity
        outline_alpha = (outline.astype(np.float32) / 255.0) * (self._opacity * 0.7)

        stamped = image.astype(np.float32)
        stamped = stamped * (1.0 - outline_alpha[..., None]) + np.array(
            [12.0, 12.0, 12.0], dtype=np.float32
        ) * outline_alpha[..., None]
        stamped = stamped * (1.0 - alpha[..., None]) + 255.0 * alpha[..., None]
        return np.clip(stamped, 0, 255).astype(np.uint8)


def protect_text(original: np.ndarray, damaged: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Layer the untouched original pixels back over the damaged render."""
    blended = damaged.astype(np.float32) * (1.0 - mask[..., None]) + (
        original.astype(np.float32) * mask[..., None]
    )
    return np.clip(blended, 0, 255).astype(np.uint8)
