#!/usr/bin/env python3
"""Real window/GPU integration checks using the agent_capture example (stdlib only)."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/debug/examples/agent_capture"))
    parser.add_argument("--output-dir", type=Path, default=Path("target/agent-smoke"))
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--msaa4", action="store_true")
    parser.add_argument("--vsync", action="store_true")
    parser.add_argument("--tick-micros", type=int)
    parser.add_argument("--submission-slots", type=int, default=8)
    args = parser.parse_args()
    args.fixed |= args.tick_micros is not None
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    connection = output / "connection.json"
    connection.unlink(missing_ok=True)
    log = (output / "app.log").open("w")
    proc = subprocess.Popen([str(args.binary.resolve()), "--fixed" if args.fixed else "--manual",
                             "--port", "0", "--connection-file", str(connection),
                             "--submission-slots", str(args.submission_slots)]
                            + (["--msaa4"] if args.msaa4 else [])
                            + (["--vsync"] if args.vsync else [])
                            + (["--tick-micros", str(args.tick_micros)] if args.tick_micros else []), stdout=log, stderr=subprocess.STDOUT)
    base = None

    def request(path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            raise AssertionError((path, error.code, error.read().decode())) from error
        with response as result:
            return result.status, result.headers.get_content_type(), result.read()

    def js(path, payload=None, expected=200):
        code, kind, body = request(path, payload)
        assert (code, kind) == (expected, "application/json"), (code, kind, body[:200])
        return json.loads(body)

    def job(path, payload):
        return js(path, payload, 202)["id"]

    def wait_job(job_id):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            state = js(f"/jobs/{job_id}")
            assert state["state"] != "failed", state
            if state["state"] == "completed":
                code, kind, body = request(f"/jobs/{job_id}/result")
                assert code == 200
                return state, kind, body
            time.sleep(0.01)
        raise AssertionError(f"job {job_id} did not complete")

    try:
        deadline = time.monotonic() + 20
        while not connection.exists() and time.monotonic() < deadline:
            assert proc.poll() is None, (output / "app.log").read_text()
            time.sleep(0.02)
        base = "http://" + json.loads(connection.read_text())["addr"]
        while not js("/status")["ready"]:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        initial = js("/status")["progress"]["submitted_ticks"]
        if not args.fixed:
            assert initial == 0
        cadence = None
        if args.vsync and args.tick_micros:
            # Exercise the ordinary wall-clock scheduler before diagnostic stepping.
            before_cadence = js("/status")["progress"]
            started = time.monotonic()
            time.sleep(0.5)
            after_cadence = js("/status")["progress"]
            wall = time.monotonic() - started
            ticks = after_cadence["submitted_ticks"] - before_cadence["submitted_ticks"]
            frames = after_cadence["submitted_frames"] - before_cadence["submitted_frames"]
            simulated = after_cadence["elapsed"] - before_cadence["elapsed"]
            assert ticks > frames * 2 and frames > 0, (ticks, frames)
            assert abs(simulated - ticks * args.tick_micros / 1e6) < 1e-6
            assert simulated <= wall + 0.02, (simulated, wall)
            cadence = {"ticks": ticks, "frames": frames, "wall_seconds": wall,
                       "simulation_seconds": simulated}
            initial = after_cadence["submitted_ticks"]
        if args.fixed:
            # A sub-millisecond clock can pass a short target before a second
            # HTTP request arrives. Establish a stable boundary first.
            js("/command", {"op": "runtime.pause"})
            initial = js("/status")["progress"]["submitted_ticks"]
        target = initial + (20 if args.fixed else 7)
        _, _, result = wait_job(job("/command", {"op": "run_until_completed", "target_tick": target, "dt": 0.01}))
        assert json.loads(result)["result"]["completed_tick"] == target
        before = js("/status")["progress"]
        assert before["completed_ticks"] == target, before
        time.sleep(0.08)
        assert js("/status")["progress"]["submitted_ticks"] == target

        images = []
        for name in ("scene", "window"):
            state, kind, body = wait_job(job("/captures", {"target": name, "exact": True}))
            assert kind == "image/png" and body.startswith(b"\x89PNG\r\n\x1a\n")
            assert struct.unpack(">II", body[16:24]) == (1280, 720)
            assert state["metadata"]["tick_id"] == target, state
            (output / f"{name}.png").write_bytes(body)
            images.append(body)
        assert images[0] != images[1], "window capture must include GUI"
        state, kind, body = wait_job(job("/captures", {"target": "values", "format": "raw", "exact": True}))
        assert kind == "application/octet-stream"
        assert list(struct.unpack("<4I", body)) == [1 + target, 2 + target, 3 + target, 4 + target]
        assert js("/status")["progress"]["submitted_ticks"] == target, "capture added a tick"
        if args.vsync:
            # Drain GUI follow-up repaints, then ensure reactive presentation sleeps.
            time.sleep(0.15)
            idle = js("/status")["progress"]["submitted_frames"]
            time.sleep(0.2)
            assert js("/status")["progress"]["submitted_frames"] == idle
            js("/command", {"op": "redraw"})
            deadline = time.monotonic() + 5
            while js("/status")["progress"]["submitted_frames"] == idle:
                assert time.monotonic() < deadline, "redraw did not wake presentation"
                time.sleep(0.01)
            assert js("/status")["progress"]["submitted_ticks"] == target

        _, _, body = wait_job(job("/command", {"op": "app.gpu_command", "payload": {"reset": 42}}))
        assert json.loads(body)["result"]["values"] == [42] * 4, body
        capture = job("/captures", {"target": "values", "format": "raw", "exact": True, "at_tick": target + 3})
        reset = job("/command", {"op": "app.gpu_command", "at_tick": target + 2, "payload": {"reset": 100}})
        run = job("/command", {"op": "run_until_completed", "target_tick": target + 5, "dt": 0.01})
        wait_job(reset)
        state, _, raw = wait_job(capture)
        assert struct.unpack("<4I", raw) == (101,) * 4, raw
        assert state["metadata"]["tick_id"] == target + 3, state
        wait_job(run)
        _, _, body = wait_job(job("/command", {"op": "app.gpu_command", "payload": {}}))
        assert json.loads(body)["result"]["values"] == [103] * 4, body
        final = js("/status")
        assert final["progress"]["completed_ticks"] == target + 5, final
        if args.fixed:
            js("/command", {"op": "runtime.resume"})
            time.sleep(0.08)
            assert js("/status")["progress"]["submitted_ticks"] > target + 5
            js("/command", {"op": "runtime.pause"})
        # The synchronous compatibility endpoint is also a capture-only render.
        paused_tick = js("/status")["progress"]["submitted_ticks"]
        code, kind, png = request("/screenshot")
        assert code == 200 and kind == "image/png" and png.startswith(b"\x89PNG")
        assert js("/status")["progress"]["submitted_ticks"] == paused_tick
        print(json.dumps({"ok": True, "mode": "fixed" if args.fixed else "manual", "vsync": args.vsync, "cadence": cadence, "samples": 4 if args.msaa4 else 1, "progress": final["progress"], "output": str(output)}))
        js("/command", {"op": "shutdown"})
        proc.wait(timeout=10)
        assert proc.returncode == 0, (output / "app.log").read_text()
    finally:
        if proc.poll() is None:
            if base:
                try:
                    js("/command", {"op": "shutdown"})
                    proc.wait(timeout=3)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    proc.terminate()
            else:
                proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        log.close()


if __name__ == "__main__":
    main()
