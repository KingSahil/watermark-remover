"""HTTP-level check of the /api/process NDJSON stream."""

from __future__ import annotations

import io
import json
import sys
import uuid
import urllib.request
from pathlib import Path

SAMPLE = Path(__file__).resolve().parent / "out" / "sample_watermarked.png"
URL = "http://127.0.0.1:8000/api/process"


def main() -> None:
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(
        f'Content-Disposition: form-data; name="file"; filename="{SAMPLE.name}"\r\n'.encode()
    )
    body.write(b"Content-Type: image/png\r\n\r\n")
    body.write(SAMPLE.read_bytes())
    body.write(f"\r\n--{boundary}--\r\n".encode())

    request = urllib.request.Request(
        URL,
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        print("status", response.status, response.headers.get("Content-Type"))
        total = 0
        last = None
        stages: list[str] = []
        for line in response:
            total += 1
            message = json.loads(line)
            last = message.get("stage")
            if message.get("stage"):
                stages.append(message["stage"])
            print(
                f"  line {total}: stage={message.get('stage')} "
                f"progress={message.get('progress')} "
                f"mask_b64={len(message.get('mask', ''))} "
                f"image_b64={len(message.get('image', ''))}"
            )
        assert total >= 10, f"expected at least 10 lines (20 diffusion steps), got {total}"
        assert "mask" in stages, "no mask event in the stream"
        assert stages.count("evaporate") >= 10, (
            f"expected per-step evaporate events, got {stages.count('evaporate')}"
        )
        assert last == "done", f"last stage was {last}"
    print("endpoint stream OK")


if __name__ == "__main__":
    sys.exit(main())
