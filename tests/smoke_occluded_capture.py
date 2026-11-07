#!/usr/bin/env python3
"""Check agent_capture scene/GUI captures, including from an occluded window."""
import argparse
import json
from pathlib import Path
import struct
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection-file", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:3939")
    parser.add_argument("--require-no-present", action="store_true",
                        help="cover/hide the window first; assert captures do not present")
    parser.add_argument("--output-dir", type=Path, default=Path("target/occluded-capture-smoke"))
    args = parser.parse_args()
    base = ("http://" + json.loads(args.connection_file.read_text())["addr"]
            if args.connection_file else args.url).rstrip("/")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    def request(path, payload=None):
        req = urllib.request.Request(
            base + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.headers.get_content_type(), response.read()

    def js(path, payload=None):
        kind, body = request(path, payload)
        assert kind == "application/json", (path, kind)
        return json.loads(body)

    def start(path, payload):
        return js(path, payload)["id"]

    def finish(job_id):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            job = js(f"/jobs/{job_id}")
            assert job["state"] != "failed", job
            if job["state"] == "completed":
                kind, body = request(f"/jobs/{job_id}/result")
                return job, kind, body
            time.sleep(0.01)
        raise AssertionError(f"job {job_id} stalled")

    status = js("/status")
    assert status["ready"] and status["screenshot_supported"], status
    size = tuple(status["window_size"])
    js("/command", {"op": "runtime.pause"})
    tick = js("/status")["progress"]["submitted_ticks"]
    # Let startup GUI/visibility events settle before counting presentations.
    time.sleep(0.3)
    presented = js("/status")["progress"]["presented_frames"]
    _, kind, body = finish(start("/command", {
        "op": "run_until_completed", "target_tick": tick + 7, "dt": 0.01,
    }))
    assert kind == "application/json"
    assert json.loads(body)["result"]["completed_tick"] == tick + 7
    tick += 7

    def image(job_id, name, expected_tick):
        job, kind, body = finish(job_id)
        assert kind == "image/png" and body.startswith(b"\x89PNG\r\n\x1a\n"), job
        assert struct.unpack(">II", body[16:24]) == size, job
        assert job["metadata"]["tick_id"] == expected_tick, job
        assert job["metadata"]["frame_id"] > 0, job
        assert job["metadata"]["submission_id"] > 0, job
        (output / f"{name}.png").write_bytes(body)
        return body

    images = [image(start("/captures", {"target": name, "exact": True}), name, tick)
              for name in ("scene", "window")]
    assert images[0] != images[1], "window capture must include GUI"
    assert js("/status")["progress"]["submitted_ticks"] == tick, "capture advanced physics"

    # Two frame captures share a future boundary while an ordered GPU command
    # and buffer readback exercise capacity and exact-fence handling.
    future = tick + 3
    scene = start("/captures", {"target": "scene", "exact": True, "at_tick": future})
    window = start("/captures", {"target": "window", "exact": True, "at_tick": future})
    values = start("/captures", {
        "target": "values", "format": "raw", "exact": True, "at_tick": future,
    })
    reset = start("/command", {
        "op": "app.gpu_command", "payload": {"reset": 100}, "at_tick": tick + 2,
    })
    run = start("/command", {"op": "run_until_completed", "target_tick": tick + 5, "dt": 0.01})
    image(scene, "future-scene", future)
    image(window, "future-window", future)
    job, kind, body = finish(values)
    assert kind == "application/octet-stream" and struct.unpack("<4I", body) == (101,) * 4, job
    assert job["metadata"]["tick_id"] == future, job
    finish(reset)
    finish(run)
    kind, body = request("/screenshot")
    assert kind == "image/png" and body.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", body[16:24]) == size
    final = js("/status")["progress"]
    assert final["submitted_ticks"] == final["completed_ticks"] == tick + 5, final
    if args.require_no_present:
        assert final["presented_frames"] == presented, (presented, final)
    print(json.dumps({"ok": True, "progress": final, "presented_before": presented,
                      "require_no_present": args.require_no_present, "output": str(output)}))


if __name__ == "__main__":
    main()
