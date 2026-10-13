"""Generate synthetic inputs for the Watermark Remover pipeline.

Run:  python scripts/make_sample.py
Writes to scripts/out/:
  sample_watermarked.png  photo-like sheet with a diagonal WATERMARK graphic
  sample_plain.png        photo-like sheet with no text at all
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

import cv2
import numpy as np

from app.processing.noise import FractalNoise
from app.processing.text import WatermarkStamp

OUT_DIR = Path(__file__).resolve().parent / "out"


def photo_base(width: int = 1200, height: int = 800, seed: int = 11) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    sky = np.array([232.0, 200.0, 150.0], dtype=np.float32)
    ground = np.array([95.0, 140.0, 110.0], dtype=np.float32)
    ramp = np.clip((y - 0.34) / 0.28, 0.0, 1.0)[..., None]
    gradient = sky[None, None, :] * (1.0 - ramp) + ground[None, None, :] * ramp

    glow = np.exp(-(((x - 0.78) ** 2 + (y - 0.18) ** 2) / 0.03))[..., None]
    texture = FractalNoise(seed).field(height, width, octaves=5, base_cells=6)
    horizon = np.exp(-((y - 0.62) ** 2) / 0.0004)[..., None]

    image = gradient
    image = image + glow * np.array([26.0, 54.0, 86.0], dtype=np.float32)
    image = image + (texture - 0.5)[..., None] * 26.0
    image = image - horizon * np.array([18.0, 26.0, 16.0], dtype=np.float32)

    # A lake across the lower third: blue-cyan, darker with depth, with faint
    # ripple texture and a sun-glint streak — actual water for CLIPSeg to find.
    lake = np.clip((y - 0.62) / 0.38, 0.0, 1.0)
    ripples = FractalNoise(seed + 5).field(height, width, octaves=4, base_cells=24)
    deep = np.array([120.0, 70.0, 25.0], dtype=np.float32)
    shallow = np.array([185.0, 130.0, 70.0], dtype=np.float32)
    water = shallow[None, None, :] * (1.0 - lake[..., None]) + deep[None, None, :] * lake[
        ..., None
    ]
    water = water + (ripples - 0.5)[..., None] * 18.0
    glint = np.exp(-(((x - 0.72) ** 2) / 0.004 + (y - 0.78) ** 2 / 0.02))[
        ..., None
    ] * np.array([40.0, 30.0, 10.0], dtype=np.float32)
    water = water + glint
    blend = np.clip((y - 0.615) / 0.03, 0.0, 1.0)[..., None]
    image = image * (1.0 - blend) + water * blend
    return np.clip(image, 0, 255).astype(np.uint8)


def build_watermarked() -> np.ndarray:
    return WatermarkStamp().stamp(photo_base())


def build_plain() -> np.ndarray:
    return photo_base(seed=23)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    watermarked = build_watermarked()
    plain = build_plain()
    cv2.imwrite(str(OUT_DIR / "sample_watermarked.png"), watermarked)
    cv2.imwrite(str(OUT_DIR / "sample_plain.png"), plain)
    print(f"Wrote {OUT_DIR / 'sample_watermarked.png'} ({watermarked.shape[1]}x{watermarked.shape[0]})")
    print(f"Wrote {OUT_DIR / 'sample_plain.png'} ({plain.shape[1]}x{plain.shape[0]})")


if __name__ == "__main__":
    main()
