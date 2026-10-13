"""Shared AI model registry for the water-removal pipeline.

Models are heavy (~5 GB combined), so they load lazily on the first request,
are cached process-wide, and GPU inference is serialized behind a lock so two
concurrent uploads cannot blow past the 8 GB VRAM budget.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger("watermark-remover.ai_models")


class WaterPipelineError(RuntimeError):
    """User-facing failure inside the AI pipeline (load failure, CUDA OOM)."""


_CLIPSEG_MODEL_ID = "CIDAS/clipseg-rd64-refined"
_SD_MODEL_ID = "runwayml/stable-diffusion-inpainting"

_model_lock = threading.Lock()
_clipseg = None
_sd_pipeline = None


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # pragma: no cover - torch import itself failing
        return False


def get_clipseg():
    """Load (once) and return the CLIPSeg processor + model pair."""
    global _clipseg
    with _model_lock:
        if _clipseg is not None:
            return _clipseg
        try:
            import torch
            from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

            logger.info("Loading CLIPSeg (%s)", _CLIPSEG_MODEL_ID)
            processor = CLIPSegProcessor.from_pretrained(_CLIPSEG_MODEL_ID)
            model = CLIPSegForImageSegmentation.from_pretrained(_CLIPSEG_MODEL_ID)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device).eval()
        except Exception as error:
            raise WaterPipelineError(
                "Could not load the water-detection model. Check your internet "
                "connection and Hugging Face cache, then retry."
            ) from error
        _clipseg = (processor, model, device)
        return _clipseg


def get_sd_pipeline():
    """Load (once) and return the Stable Diffusion inpainting pipeline."""
    global _sd_pipeline
    with _model_lock:
        if _sd_pipeline is not None:
            return _sd_pipeline
        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline

            logger.info("Loading Stable Diffusion inpainting (%s)", _SD_MODEL_ID)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                _SD_MODEL_ID,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
                # 8 GB VRAM safety: trade a little speed for headroom.
                if hasattr(pipe, "enable_attention_slicing"):
                    pipe.enable_attention_slicing()
                if hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()
                elif hasattr(pipe.vae, "enable_slicing"):
                    pipe.vae.enable_slicing()
            pipe.set_progress_bar_config(disable=True)
        except Exception as error:
            raise WaterPipelineError(
                "Could not load the AI inpainting model. Check your internet "
                "connection and Hugging Face cache, then retry."
            ) from error
        _sd_pipeline = pipe
        return _sd_pipeline


def gpu_lock() -> threading.Lock:
    """Lock serializing GPU inference across worker threads."""
    return _model_lock
