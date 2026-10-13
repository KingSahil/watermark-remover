"""End-to-end pipeline assertions for the Watermark Remover.

Run:  python scripts/smoke_test.py
"""

from __future__ import annotations

import asyncio
import base64
import sys
import zlib
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

import cv2
import numpy as np

from app.processing import WatermarkProcessor
from app.processing.text import TextRegionDetector
from make_sample import OUT_DIR, build_plain, build_watermarked


def _encode(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok, "could not encode sample input"
    return encoded.tobytes()


def _decode_payload(encoded: str) -> np.ndarray:
    return cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )


def _water_signature(image: np.ndarray, region: np.ndarray) -> float:
    """Fraction of the region that is saturated blue/cyan — i.e. looks like water."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hues, sats = hsv[..., 0].astype(np.int32), hsv[..., 1].astype(np.int32)
    watery = ((hues >= 75) & (hues <= 135) & (sats >= 50)) & (region > 0)
    return float(np.count_nonzero(watery)) / float(max(1, np.count_nonzero(region)))


def _residual_water(image: np.ndarray, zone_mask: np.ndarray) -> int:
    """Strict turquoise left inside the grown water zone, excluding the sky.

    Mirrors the pipeline's sweep gate: mountains keep their pre-existing teal
    (correctly untouched), the sky stays blue, but the lake zone must read
    zero — this is the "water fully removed" assertion.
    """
    data = image.astype(np.float32)
    brightest = data.max(axis=-1)
    darkest = data.min(axis=-1)
    saturation = (brightest - darkest) / np.maximum(brightest, 1.0)
    chroma = ((data[..., 0] - data[..., 2]) > 40.0) & (saturation > 0.25)
    zone = (
        cv2.dilate((zone_mask > 0).astype(np.uint8), np.ones((40, 40), np.uint8))
        > 0
    )
    _count, labels = cv2.connectedComponents(chroma.astype(np.uint8), connectivity=8)[
        :2
    ]
    sky_ids = set(int(v) for v in np.unique(labels[0, :])) - {0}
    sky = np.isin(labels, list(sky_ids)) if sky_ids else np.zeros_like(chroma)
    return int(np.count_nonzero(chroma & zone & ~sky))


def run_case(name: str, image: np.ndarray) -> None:
    processor = WatermarkProcessor()
    raw = _encode(image)
    decoded = processor.decode(raw)

    async def _collect() -> list[dict]:
        return [
            event
            async for event in processor.stream(decoded, zlib.crc32(raw) & 0xFFFFFFFF)
        ]

    events = asyncio.run(_collect())

    assert not any("error" in event for event in events), f"{name}: pipeline emitted an error"
    progresses = [event["progress"] for event in events if "progress" in event]
    assert progresses == sorted(progresses), f"{name}: progress went backwards: {progresses}"

    masks = [event for event in events if event.get("stage") == "mask"]
    assert masks, f"{name}: no mask event streamed before done"
    mask_preview = _decode_payload(masks[0]["mask"])
    assert mask_preview is not None, f"{name}: streamed mask did not decode"
    assert mask_preview.shape[1] == 256, (
        f"{name}: mask preview should be ~256px wide, got {mask_preview.shape[1]}"
    )

    done = [event for event in events if event.get("stage") == "done"]
    assert done, f"{name}: no done event"
    payload = done[0]
    assert "mask" in payload, f"{name}: done event is missing the water mask"
    assert payload["width"] == decoded.shape[1] and payload["height"] == decoded.shape[0]

    output = _decode_payload(payload["image"])
    assert output is not None, f"{name}: output PNG did not decode"
    assert output.shape == decoded.shape, f"{name}: output size changed"

    # The streamed water mask, upsampled to output resolution, drives the
    # regional assertions.
    water = cv2.resize(
        mask_preview, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    core = cv2.erode(water, np.ones((7, 7), np.uint8))
    assert int(np.count_nonzero(core)) > 500, f"{name}: water mask has no solid core"

    damage = float(cv2.absdiff(output, decoded).mean())
    assert damage > 2.0, f"{name}: image looks untouched (mean abs diff {damage:.2f})"

    water_diff = float(cv2.absdiff(output, decoded)[core > 0].mean())
    assert water_diff > 12.0, (
        f"{name}: water region was not repainted (mean diff {water_diff:.2f})"
    )

    sig_in = _water_signature(decoded, core)
    sig_out = _water_signature(output, core)
    assert sig_out < max(0.5 * sig_in, 0.02) or sig_in < 0.02, (
        f"{name}: water signature did not collapse ({sig_in:.3f} -> {sig_out:.3f})"
    )

    residual = _residual_water(output, water)
    assert residual < 100, (
        f"{name}: water not fully removed ({residual} turquoise pixels left)"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_DIR / f"{name}_input.png"), decoded)
    cv2.imwrite(str(OUT_DIR / f"{name}_output.png"), output)
    cv2.imwrite(str(OUT_DIR / f"{name}_mask.png"), water)

    text_mask = TextRegionDetector().detect(decoded)
    if name == "watermarked":
        assert text_mask is not None, "watermarked: text region was not detected"
        core_text = text_mask >= 0.999
        assert int(core_text.sum()) > 400, "watermarked: feathered mask has no solid core"
        # Glyphs are pasted back from the original after inpainting, so the
        # core must stay bright and close to the original.
        difference = np.abs(
            output.astype(np.int16)[core_text] - decoded.astype(np.int16)[core_text]
        )
        assert float(difference.mean()) < 45.0, (
            f"watermarked: glyph zone drifted (mean diff {float(difference.mean()):.1f})"
        )
        assert float(output[core_text].mean()) > 150.0, (
            "watermarked: watermark did not stay bright"
        )
    else:
        assert text_mask is None, "plain: detector produced a false positive on texture"
        # The stamp is white at 0.78 opacity, so it brightens rather than
        # saturates: count clearly-brightened pixels instead of near-255 ones.
        input_bright = int(np.count_nonzero(decoded >= 210))
        output_bright = int(np.count_nonzero(output >= 210))
        assert output_bright > input_bright + 20_000, (
            f"plain: WATERMARK stamp was not applied "
            f"({input_bright} -> {output_bright} bright pixels)"
        )

    steps = [event for event in events if event.get("stage") == "evaporate"]
    print(
        f"[ok] {name}: {events[-1]['stage']} at {payload['width']}x{payload['height']}, "
        f"damage={damage:.2f}, water_diff={water_diff:.2f}, "
        f"sig {sig_in:.3f}->{sig_out:.3f}, residual={residual}, "
        f"diffusion_steps={len(steps)}"
    )


def main() -> None:
    run_case("watermarked", build_watermarked())
    run_case("plain", build_plain())
    print("smoke test passed")


if __name__ == "__main__":
    main()
