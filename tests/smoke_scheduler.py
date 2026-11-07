#!/usr/bin/env python3
"""Native event-loop regression for heavy CPU-only Fixed ticks (Issue #27).

Build with `cargo build --example agent_capture --features agent`, then run this
script with the window visible. It switches from GPU compute to CPU-only ticks
whose empty GPU submissions can complete while the next CPU tick is running.
This checks ordinary redraw/presentation and host command progress; native input
event injection is intentionally outside this HTTP-based fixture.
"""

import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/debug/examples/agent_capture"))
    parser.add_argument("--output-dir", type=Path, default=Path("target/scheduler-smoke"))
    parser.add_argument("--cpu-tick-micros", type=int, default=4000)
    args = parser.parse_args()
    assert 400 < args.cpu_tick_micros <= 100_000
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    connection = output / "connection.json"
    connection.unlink(missing_ok=True)
    base = None

    with (output / "app.log").open("w") as log:
        proc = subprocess.Popen(
            [str(args.binary.resolve()), "--vsync", "--tick-micros", "400",
             "--submission-slots", "8", "--port", "0",
             "--connection-file", str(connection)],
            stdout=log, stderr=subprocess.STDOUT,
        )

        def request(path, payload=None):
            data = json.dumps(payload).encode() if payload is not None else None
            req = urllib.request.Request(
                base + path, data=data, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=3) as response:
                    assert response.status == 200
                    return json.load(response)
            except urllib.error.HTTPError as error:
                raise AssertionError((path, error.code, error.read().decode())) from error

        def status():
            assert proc.poll() is None, (output / "app.log").read_text()
            return request("/status")

        def command(payload):
            return request("/command", {"op": "app.command", "payload": payload})

        def wait_progress(before, timeout=4, probe=None):
            deadline = time.monotonic() + timeout
            fields = ("submitted_ticks", "completed_ticks", "completed_submissions",
                      "submitted_frames", "presented_frames")
            while time.monotonic() < deadline:
                after = status()
                if (all(after["progress"][field] > before["progress"][field] for field in fields)
                        and (probe is None or after["app"]["command_probe"] == probe)):
                    return after
                time.sleep(0.03)
            raise AssertionError({
                "error": "CPU-only tick completion wakes starved ordinary presentation",
                "before": before["progress"], "after": after["progress"],
            })

        try:
            deadline = time.monotonic() + 20
            while not connection.exists():
                assert proc.poll() is None, (output / "app.log").read_text()
                assert time.monotonic() < deadline, "agent connection file was not written"
                time.sleep(0.02)
            base = "http://" + json.loads(connection.read_text())["addr"]
            while not status()["ready"]:
                assert time.monotonic() < deadline, "runner did not become ready"
                time.sleep(0.02)

            # Establish ordinary GPU rendering first, then reproduce GPU -> CPU
            # mode switching without a diagnostic pause or exact stepping job.
            gpu = wait_progress(status())
            command({"cpu_tick_micros": args.cpu_tick_micros})
            previous = status()
            samples = []
            for probe in range(1, 4):
                time.sleep(0.15)
                command({"probe": probe})
                after = wait_progress(previous, probe=probe)
                assert after["app"]["cpu_tick_micros"] == args.cpu_tick_micros, after
                assert after["app"]["command_probe"] == probe, after
                ticks = after["progress"]["submitted_ticks"] - previous["progress"]["submitted_ticks"]
                elapsed = after["progress"]["elapsed"] - previous["progress"]["elapsed"]
                assert abs(elapsed - ticks * 0.0004) < 1e-6, (ticks, elapsed)
                samples.append(after["progress"])
                previous = after

            # Pausing must still get through to the host after sustained load.
            request("/command", {"op": "runtime.pause"})
            paused = status()["progress"]["submitted_ticks"]
            time.sleep(0.1)
            assert status()["progress"]["submitted_ticks"] == paused
            print(json.dumps({"ok": True, "cpu_tick_micros": args.cpu_tick_micros,
                              "gpu_progress": gpu["progress"], "cpu_samples": samples,
                              "output": str(output)}))
            request("/command", {"op": "shutdown"})
            proc.wait(timeout=5)
            assert proc.returncode == 0, (output / "app.log").read_text()
        finally:
            # A regression can leave the event loop permanently busy, including
            # its shutdown command. Cleanup must not depend on host cooperation.
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == "__main__":
    main()
