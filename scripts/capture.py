#!/usr/bin/env python3
"""Capture a mikage resource through the asynchronous jobs API (standard library only)."""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request


def request(base, path, token, timeout, payload=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.headers.get_content_type(), response.read()
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {path}: {detail}") from error


def json_response(response, expected_status):
    status, content_type, body = response
    if status != expected_status or content_type != "application/json":
        raise RuntimeError(f"expected HTTP {expected_status} JSON, received {status} {content_type}")
    return json.loads(body)


def write_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:3939")
    parser.add_argument("--connection-file", type=Path)
    parser.add_argument("--token", default=os.environ.get("MIKAGE_TOKEN"))
    parser.add_argument("--target", default="window")
    parser.add_argument("--format", choices=("png", "raw"), default="png")
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--at-tick", type=int)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or (args.at_tick is not None and args.at_tick < 0):
        parser.error("timeout must be finite and positive and at-tick nonnegative")
    if args.connection_file:
        connection = json.loads(args.connection_file.read_text())
        args.url = "http://" + connection["addr"]

    deadline = time.monotonic() + args.timeout

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise RuntimeError("capture deadline exceeded; the output file was not replaced")
        return seconds

    payload = {"target": args.target, "format": args.format, "exact": args.exact}
    if args.at_tick is not None:
        payload["at_tick"] = args.at_tick
    accepted = json_response(request(args.url, "/captures", args.token, remaining(), payload), 202)
    job_id = accepted["id"]
    if not isinstance(job_id, int) or job_id < 1:
        raise RuntimeError("server returned an invalid job identifier")
    while True:
        status = json_response(request(args.url, f"/jobs/{job_id}", args.token, remaining()), 200)
        if status["state"] == "failed":
            raise RuntimeError(f"capture job {job_id} failed: {status.get('error', 'unknown error')}")
        if status["state"] == "completed":
            break
        if status["state"] != "pending":
            raise RuntimeError(f"unexpected job state: {status['state']}")
        time.sleep(min(0.05, remaining()))

    code, content_type, data = request(args.url, f"/jobs/{job_id}/result", args.token, remaining())
    expected_type = "image/png" if args.format == "png" else "application/octet-stream"
    if code != 200 or content_type != expected_type:
        raise RuntimeError(f"expected HTTP 200 {expected_type}, received {code} {content_type}")
    if args.format == "png" and not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("server response is not a PNG")
    write_atomic(args.output, data)
    print(json.dumps({"job_id": job_id, "output": str(args.output), "bytes": len(data), "metadata": status.get("metadata")}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, urllib.error.URLError) as error:
        raise SystemExit(f"capture failed: {error}") from error
