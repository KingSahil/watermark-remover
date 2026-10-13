"""Water segmentation with CLIPSeg.

The refined rd64 CLIPSeg model is prompted with several water synonyms; the
per-prompt probability maps are unioned, thresholded, and cleaned up
morphologically so the mask is a few solid blobs rather than confetti. Works
on any photo containing water — lakes, rivers, seas, pools — not just one
trained scene.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .ai_models import WaterPipelineError, get_clipseg

logger = logging.getLogger("watermark-remover.segment")

PROMPTS = ("water", "lake", "river", "sea", "pool")

# Coverage window (fraction of the image) accepted as a sane water mask.
_MIN_COVERAGE = 0.005
_MAX_COVERAGE = 0.92


def _clean_mask(binary: np.ndarray) -> np.ndarray:
    """Open/close cleanup, then keep only the largest components."""
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if count <= 2:
        return closed
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max())
    keep_area = max(largest // 4, 200)
    keep = {1 + index for index, area in enumerate(areas) if area >= keep_area}
    result = np.isin(labels, list(keep)).astype(np.uint8) * 255
    return result


def _chroma_water(image_bgr: np.ndarray) -> np.ndarray:
    """Cheap turquoise detector: blue-dominant saturated pixels.

    CLIPSeg scores thin/distant water low, so this supplements the AI mask —
    it is only ever applied *inside* a grown AI zone, never standalone, which
    keeps teal foliage and blue skies out.
    """
    data = image_bgr.astype(np.float32)
    blue, red = data[..., 0], data[..., 2]
    brightest = data.max(axis=-1)
    darkest = data.min(axis=-1)
    saturation = (brightest - darkest) / np.maximum(brightest, 1.0)
    return (((blue - red) > 35.0) & (saturation > 0.22)).astype(np.uint8) * 255


def _union_chroma(candidate: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    """Pull chroma-turquoise pixels touching the AI mask into it, generously.

    Catches the far-shore strip, shoreline rims and speckles that CLIPSeg
    scores below threshold, without letting distant teal texture in. Finishes
    with a wide close (fills speckle holes and boat-puddle gaps under ~25px)
    and dilates outward, because every pixel of missed water survives
    compositing as a vivid rim — over-masking the shoreline slightly blends
    seamlessly, under-masking leaves turquoise behind.
    """
    height, width = candidate.shape[:2]
    reach = (max(9, round(min(height, width) * 0.035)) | 1)
    grown = cv2.dilate(candidate, np.ones((reach, reach), np.uint8))
    chroma = _chroma_water(image_bgr)
    union = cv2.bitwise_or(
        candidate, cv2.bitwise_and(chroma, chroma, mask=grown)
    )
    close_size = (min(51, max(15, round(min(height, width) * 0.06))) | 1)
    closed = cv2.morphologyEx(
        union, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8)
    )
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(opened, np.ones((reach, reach), np.uint8))


def _bottom_fallback(shape: tuple[int, ...]) -> np.ndarray:
    """Last-resort mask: the lower half of the frame, heavily feathered.

    The joke only works if *something* evaporates, so a photo with no
    detectable water still loses its bottom half to the lakebed.
    """
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[round(height * 0.52) :, :] = 255
    return cv2.GaussianBlur(mask, (0, 0), max(3.0, height * 0.01))


def segment_water(image_bgr: np.ndarray) -> np.ndarray:
    """Return a uint8 mask (255 = water) at the input image's resolution."""
    try:
        import torch
    except Exception as error:  # pragma: no cover - torch missing/misinstalled
        raise WaterPipelineError(
            "The AI backend (PyTorch) is not available. Reinstall requirements."
        ) from error

    processor, model, device = get_clipseg()
    height, width = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    from PIL import Image

    pil = Image.fromarray(rgb)
    try:
        inputs = processor(
            text=list(PROMPTS), images=[pil] * len(PROMPTS),
            return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits  # (prompts, 352, 352)
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        probability = torch.sigmoid(logits).max(dim=0).values
        probability = probability.cpu().numpy().astype(np.float32)
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise WaterPipelineError(
                "The GPU ran out of memory while detecting water. "
                "Close other GPU apps and retry."
            ) from error
        raise

    score = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)

    for threshold in (0.5, 0.4, 0.3, 0.2):
        candidate = _clean_mask((score >= threshold).astype(np.uint8) * 255)
        coverage = float(np.count_nonzero(candidate)) / float(height * width)
        if _MIN_COVERAGE <= coverage <= _MAX_COVERAGE:
            return _union_chroma(candidate, image_bgr)
        if coverage > _MAX_COVERAGE:
            break  # looser thresholds only get bigger; keep the tighter one

    logger.warning("No water detected; falling back to the lower-half mask")
    return _bottom_fallback(image_bgr.shape)
