"""Dry-bed inpainting with Stable Diffusion.

The water region is inpainted at 512 px (fp16 on CUDA) into sun-baked mud,
then the repainted area is composited back into the full-resolution original
with a feathered edge, so everything outside the water stays pixel-sharp.
Each diffusion step reports through ``on_step`` so the NDJSON stream can show
real progress.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import cv2
import numpy as np

from .ai_models import WaterPipelineError, get_sd_pipeline, gpu_lock

logger = logging.getLogger("watermark-remover.inpaint")

WORK_SIZE = 512
STEPS = 20
GUIDANCE = 7.0
PROMPT = "dry cracked earth lakebed, sun-baked mud, photorealistic, matching sunlight"
NEGATIVE_PROMPT = (
    "water, lake, river, blue, turquoise, teal, cyan, green water, "
    "reflection, puddle, waves, ripples, sign, billboard, signpost, "
    "text, letters, words"
)


def _round8(value: int) -> int:
    return max(64, round(value / 8.0) * 8)


def sweep_residual_water(
    image_bgr: np.ndarray,
    water_mask: np.ndarray,
) -> np.ndarray:
    """Drain any turquoise the diffusion left behind (or protection restored).

    Catches SD context bleed (a boat "wants" water around it, so diffusion
    repaints teal despite the negative prompt), chroma in glyph counters, and
    shoreline specks: strict water-chroma pixels inside a generous grown zone
    are desaturated toward gray. This runs AFTER protect with no text
    exclusion — white/black glyphs have ~zero saturation, so they are immune
    by physics while vivid leftovers die. Leftovers read as damp spots, which
    suits a dried lakebed.
    """
    data = image_bgr.astype(np.float32)
    blue, red = data[..., 0], data[..., 2]
    brightest = data.max(axis=-1)
    darkest = data.min(axis=-1)
    saturation = (brightest - darkest) / np.maximum(brightest, 1.0)
    chroma = ((blue - red) > 40.0) & (saturation > 0.25)

    height, width = image_bgr.shape[:2]
    reach = max(25, round(min(height, width) * 0.09))
    zone = cv2.dilate(water_mask, np.ones((reach, reach), np.uint8)) > 0
    chroma_u8 = (chroma).astype(np.uint8) * 255
    # Diffusion is nondeterministic run-to-run (same seed, different bleed),
    # so the sweep cannot trust observed shapes: bridges of faint wash can
    # connect specks to the sky in any single run. Eroding first severs
    # ≤5px bridges; regrowing restores the sky boundary but not the specks.
    eroded = cv2.erode(chroma_u8, np.ones((3, 3), np.uint8))
    _count, labels = cv2.connectedComponents(eroded, connectivity=8)[:2]
    sky_ids = set(int(v) for v in np.unique(labels[0, :])) - {0}
    sky_core = np.isin(labels, list(sky_ids)) if sky_ids else np.zeros_like(chroma)
    sky_full = cv2.dilate(sky_core.astype(np.uint8) * 255, np.ones((3, 3), np.uint8)) > 0
    # Water only ever lives at/below the lake's top edge: small leftovers are
    # bleached inside that band (never in mountain-sky gaps above it). The
    # sky itself is capped at a line above the waterline, so a wash bridge
    # painted from lake into sky cannot smuggle specks past the gate — sky
    # dips below the line survive via the big-blob rule (giant, no zone
    # touch), never via sky membership.
    col_tops = (water_mask > 0).argmax(axis=0)
    wet_cols = (water_mask > 0).any(axis=0)
    water_top = int(col_tops[wet_cols].min()) if bool(wet_cols.any()) else 0
    band = np.zeros((height, width), dtype=bool)
    band[max(0, water_top - 40) :, :] = True
    skyline_cut = np.zeros((height, width), dtype=bool)
    skyline_cut[: max(0, water_top - 30), :] = True
    capped_sky = sky_full & skyline_cut
    _full, full_labels = cv2.connectedComponents(chroma_u8, connectivity=8)[:2]
    small = np.zeros_like(chroma)
    big = np.zeros_like(chroma)
    for blob_id in range(1, _full):
        blob = full_labels == blob_id
        if int(blob.sum()) < 800:
            if bool((blob & capped_sky).any()) or not bool((blob & band).any()):
                continue
            small |= blob
        elif bool((blob & zone).any()):
            big |= blob
    residual = ((small | big)).astype(np.uint8) * 255
    found = int(np.count_nonzero(residual))
    if found == 0:
        return image_bgr
    logger.info("Draining %d residual water pixels", found)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)[..., None].astype(np.float32)
    drained = data * 0.05 + gray * 0.95
    keep = (residual > 0)[..., None]
    return np.clip(np.where(keep, drained, data), 0, 255).astype(np.uint8)


def inpaint_lakebed(
    image_bgr: np.ndarray,
    water_mask: np.ndarray,
    seed: int,
    on_step: Callable[[int, int], None],
) -> np.ndarray:
    """Repaint the masked water as a dry lakebed; return full-resolution BGR."""
    import torch
    from PIL import Image

    height, width = image_bgr.shape[:2]
    scale = WORK_SIZE / max(height, width)
    work_w = _round8(round(width * scale))
    work_h = _round8(round(height * scale))

    small = cv2.resize(image_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(water_mask, (work_w, work_h), interpolation=cv2.INTER_NEAREST)
    # Grow (scale-aware) so the model repaints the shoreline and the water
    # ring around objects sitting in the lake (boats): an eroded mask leaves
    # a rim of original water around every edge. Bed painted marginally over
    # the shore blends seamlessly.
    grow_work = max(5, round(min(work_h, work_w) * 0.03))
    grown = cv2.dilate(mask_small, np.ones((grow_work, grow_work), np.uint8))
    soft_small = cv2.GaussianBlur(grown, (0, 0), 4.0)

    pipe = get_sd_pipeline()
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    def _callback(_pipe, step_index: int, _timestep, callback_kwargs: dict) -> dict:
        on_step(step_index + 1, STEPS)
        return callback_kwargs

    pil_image = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(soft_small)  # white (255) = repaint as lakebed

    try:
        with gpu_lock():
            result = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                generator=generator,
                callback_on_step_end=_callback,
            ).images[0]
    except torch.cuda.OutOfMemoryError as error:
        raise WaterPipelineError(
            "The GPU ran out of memory while evaporating the water. "
            "Close other GPU apps and retry."
        ) from error

    dry = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    dry_full = cv2.resize(dry, (width, height), interpolation=cv2.INTER_LANCZOS4)
    grow_full = max(9, round(min(height, width) * 0.04))
    alpha = cv2.GaussianBlur(
        cv2.dilate(water_mask, np.ones((grow_full, grow_full), np.uint8)), (0, 0), 5.0
    ).astype(np.float32) / 255.0
    blended = image_bgr.astype(np.float32) * (1.0 - alpha[..., None]) + dry_full.astype(
        np.float32
    ) * alpha[..., None]
    return np.clip(blended, 0, 255).astype(np.uint8)
