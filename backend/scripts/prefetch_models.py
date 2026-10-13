"""Pre-download the AI models so the first /api/process request is fast.

Run:  python scripts/prefetch_models.py
Downloads CIDAS/clipseg-rd64-refined and runwayml/stable-diffusion-inpainting
(~4.5 GB total) into the default Hugging Face cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from huggingface_hub import snapshot_download  # noqa: E402

MODELS = (
    "CIDAS/clipseg-rd64-refined",
    "runwayml/stable-diffusion-inpainting",
)


def main() -> None:
    for model_id in MODELS:
        print(f"downloading {model_id} …", flush=True)
        path = snapshot_download(model_id)
        print(f"  cached at {path}", flush=True)
    print("models ready")


if __name__ == "__main__":
    main()
