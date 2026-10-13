from __future__ import annotations

import asyncio
import base64
import logging
import queue
import threading
from collections.abc import AsyncIterator
from typing import Any

import cv2
import numpy as np
from starlette.concurrency import run_in_threadpool

from ..config import settings
from ..errors import InvalidImageError
from .ai_models import WaterPipelineError
from .inpaint import inpaint_lakebed, sweep_residual_water
from .segment import segment_water
from .text import TextRegionDetector, WatermarkStamp, protect_text

logger = logging.getLogger("watermark-remover.processor")

# Progress map: segmentation 0→15%, diffusion steps 15→90%, protect→100.
_DETECT_END = 0.15
_DIFFUSE_END = 0.90
_PROTECT = 0.92
_MASK_PREVIEW_WIDTH = 256


class WatermarkProcessor:
    """End-to-end AI pipeline for one image.

    CLIPSeg finds the water (any photo, not just one lake) → Stable Diffusion
    inpaints it into a dry, cracked lakebed, reporting each diffusion step →
    the original glyphs are layered back razor sharp (or the literal WATERMARK
    is stamped when the upload has no text of its own) → PNG + downscaled
    water mask are emitted. Progress is real: the bar only moves when
    something actually happened.
    """

    def decode(self, raw: bytes) -> np.ndarray:
        if not raw:
            raise InvalidImageError("The uploaded file is empty.")

        buffer = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise InvalidImageError("That file could not be read as an image.")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            rgb = image[:, :, :3].astype(np.float32)
            image = np.clip(rgb * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)

        image = self._to_uint8(image)
        image = self._cap_size(image)
        return np.ascontiguousarray(image, dtype=np.uint8)

    async def stream(
        self, image: np.ndarray, seed: int
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            yield {"stage": "detect", "label": "Detecting water…", "progress": 0.02}

            def _detect() -> tuple[np.ndarray | None, np.ndarray]:
                return TextRegionDetector().detect(image), segment_water(image)

            text_mask, water_mask = await run_in_threadpool(_detect)

            yield {
                "stage": "mask",
                "label": "Detecting water…",
                "progress": _DETECT_END,
                "mask": self.encode_mask_preview(water_mask),
            }

            composited: np.ndarray | None = None
            async for event in self._diffuse(image, water_mask, seed):
                if event.get("stage") == "inpaint_done":
                    composited = event["result"]
                else:
                    yield event
            assert composited is not None

            yield {
                "stage": "protect",
                "label": "Keeping watermark…",
                "progress": _PROTECT,
            }

            def _protect() -> np.ndarray:
                if text_mask is not None:
                    kept = protect_text(image, composited, text_mask)
                else:
                    kept = WatermarkStamp().stamp(composited)
                # Sweep AFTER protect: diffusion repaints teal around boats
                # despite the negative prompt, and protection can restore
                # vivid counters — the drain is saturation-based so glyphs
                # are immune.
                return sweep_residual_water(kept, water_mask)

            final = await run_in_threadpool(_protect)

            png = await run_in_threadpool(self.encode_png, final)
            yield {
                "stage": "done",
                "progress": 1.0,
                "image": base64.b64encode(png).decode("ascii"),
                "width": int(final.shape[1]),
                "height": int(final.shape[0]),
                "mask": self.encode_mask_preview(water_mask),
            }
        except WaterPipelineError as error:
            logger.error("AI pipeline failed: %s", error, exc_info=error.__cause__)
            yield {"stage": "error", "error": str(error)}
        except Exception:
            logger.exception("Watermark processing failed")
            yield {
                "stage": "error",
                "error": "Processing failed for this image. Please try another file.",
            }

    async def _diffuse(
        self, image: np.ndarray, water_mask: np.ndarray, seed: int
    ) -> AsyncIterator[dict[str, Any]]:
        """Run the SD inpainting in a worker thread, streaming step progress.

        Yields ``evaporate`` progress events per diffusion step, then a final
        ``inpaint_done`` event carrying the composited full-resolution image.
        """
        events: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()

        def on_step(done: int, total: int) -> None:
            events.put(("step", done / total))

        def _worker() -> None:
            try:
                events.put(("done", inpaint_lakebed(image, water_mask, seed, on_step)))
            except BaseException as error:  # forwarded into the async stream
                events.put(("error", error))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        while True:
            kind, payload = await asyncio.to_thread(events.get)
            if kind == "step":
                yield {
                    "stage": "evaporate",
                    "label": "Evaporating…",
                    "progress": _DETECT_END
                    + (_DIFFUSE_END - _DETECT_END) * float(payload),
                }
            elif kind == "done":
                thread.join()
                yield {"stage": "inpaint_done", "result": payload}
                return
            else:
                thread.join()
                raise payload

    @staticmethod
    def encode_mask_preview(mask: np.ndarray) -> str:
        """Downscale the water mask to a small grayscale PNG for the client."""
        height, width = mask.shape[:2]
        scale = _MASK_PREVIEW_WIDTH / float(width)
        small = cv2.resize(
            mask,
            (_MASK_PREVIEW_WIDTH, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return base64.b64encode(WatermarkProcessor.encode_png(small)).decode("ascii")

    @staticmethod
    def encode_png(image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(
            ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, settings.png_compression]
        )
        if not ok:
            raise RuntimeError("PNG encoding failed")
        return encoded.tobytes()

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image
        if image.dtype == np.uint16:
            return (image >> 8).astype(np.uint8)
        data = image.astype(np.float32)
        if data.size and float(data.max()) <= 1.0:
            data *= 255.0
        return np.clip(data, 0, 255).astype(np.uint8)

    @staticmethod
    def _cap_size(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= settings.max_side:
            return image
        ratio = settings.max_side / longest
        resized = cv2.resize(
            image,
            (max(1, round(width * ratio)), max(1, round(height * ratio))),
            interpolation=cv2.INTER_AREA,
        )
        return resized
