"""Verify the 422 error path for non-image uploads."""

from __future__ import annotations

import io
import uuid
import urllib.error
import urllib.request

boundary = uuid.uuid4().hex
body = io.BytesIO()
body.write(f"--{boundary}\r\n".encode())
body.write(b'Content-Disposition: form-data; name="file"; filename="x.png"\r\n\r\n')
body.write(b"this is not an image")
body.write(f"\r\n--{boundary}--\r\n".encode())

request = urllib.request.Request(
    "http://127.0.0.1:8000/api/process",
    data=body.getvalue(),
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)

try:
    urllib.request.urlopen(request, timeout=30)
    raise SystemExit("unexpected 200")
except urllib.error.HTTPError as error:
    detail = error.read().decode()
    assert error.code == 422, f"expected 422, got {error.code}"
    assert "could not be read" in detail, detail
    print(f"status {error.code} {detail}")
    print("error path OK")
