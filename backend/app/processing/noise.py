from __future__ import annotations

import cv2
import numpy as np


def normalize(field: np.ndarray) -> np.ndarray:
    """Rescale a float array into [0, 1]."""
    low = float(field.min())
    high = float(field.max())
    if high - low < 1e-6:
        return np.zeros_like(field)
    return ((field - low) / (high - low)).astype(np.float32)


def smoothstep(value: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    """Hermite smoothstep used for soft stain thresholds."""
    span = max(edge1 - edge0, 1e-6)
    scaled = np.clip((value - edge0) / span, 0.0, 1.0)
    return (scaled * scaled * (3.0 - 2.0 * scaled)).astype(np.float32)


class FractalNoise:
    """Multi-octave value noise in [0, 1] built from upscaled random lattices."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def _octave(self, height: int, width: int, cells: int) -> np.ndarray:
        rows = max(2, min(cells, height))
        cols = max(2, min(cells, width))
        lattice = self._rng.random((rows + 1, cols + 1)).astype(np.float32)
        return cv2.resize(lattice, (width, height), interpolation=cv2.INTER_CUBIC)

    def field(
        self,
        height: int,
        width: int,
        *,
        octaves: int = 5,
        base_cells: int = 4,
        persistence: float = 0.55,
    ) -> np.ndarray:
        total = np.zeros((height, width), dtype=np.float32)
        amplitude = 1.0
        weight = 0.0
        for octave in range(octaves):
            cells = base_cells * (2**octave)
            total += amplitude * self._octave(height, width, cells)
            weight += amplitude
            amplitude *= persistence
        return normalize(total / weight)
