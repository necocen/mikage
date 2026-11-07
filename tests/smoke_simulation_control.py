#!/usr/bin/env python3
"""Native public SimulationControl/RedrawRequest integration checks (Issue #28).

Build with `cargo build --example agent_capture --features agent`. Run once with
the default Fixed 400 us clock and once with --per-redraw. Keep the window visible.
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
    parser.add_argument("--output-dir", type=Path, default=Path("target/simulation-control-smoke"))
    parser.add_argument("--per-redraw", action="store_true")
    parser.add_argument("--tick-micros", type=int, default=400)
    args = parser.parse_args()
    mode = "per-redraw" if args.per_redraw else "fixed"
    output = (args.output_dir / mode).resolve()
    output.mkdir(parents=True, exist_ok=True)
    connection = output / "connection.json"
    connection.unlink(missing_ok=True)
    base = None
    command_line = [str(args.binary.resolve()), "--vsync", "--port", "0",
                    "--connection-file", str(connection)]
    if not args.per_redraw:
        command_line += ["--tick-micros", str(args.tick_micros)]

    with (output / "app.log").open("w") as log:
        proc = subprocess.Popen(command_line, stdout=log, stderr=subprocess.STDOUT)

        def request(path, payload=None, expected=200):
            data = json.dumps(payload).encode() if payload is not None else None
            req = urllib.request.Request(base + path, data=data,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    assert response.status == expected, (path, response.status)
                    return json.load(response)
            except urllib.error.HTTPError as error:
                raise AssertionError((path, error.code, error.read().decode())) from error

        def status():
            assert proc.poll() is None, (output / "app.log").read_text()
            return request("/status")

        def wait_for(predicate, description, timeout=5):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                snapshot = status()
                if predicate(snapshot):
                    return snapshot
                time.sleep(0.01)
            raise AssertionError((description, snapshot))

        def command(payload):
            return request("/command", {"op": "app.command", "payload": payload})

        def control(operation):
            return command({"simulation": operation})

        def diagnostic(operation):
            return request("/command", {"op": "runtime." + operation})

        def idle(expected_tick=None):
            # GUI follow-up repaints and already submitted GPU work may finish
            # after the pause request. Wait for all of them to settle.
            deadline = time.monotonic() + 5
            signature = None
            stable_since = time.monotonic()
            while time.monotonic() < deadline:
                snapshot = status()
                progress = snapshot["progress"]
                if expected_tick is not None:
                    assert progress["submitted_ticks"] == expected_tick, snapshot
                current = (progress["submitted_ticks"], progress["submitted_submissions"],
                           progress["submitted_frames"], progress["presented_frames"])
                if current != signature:
                    signature = current
                    stable_since = time.monotonic()
                if (progress["completed_submissions"] == progress["submitted_submissions"]
                        and time.monotonic() - stable_since > 0.2):
                    return snapshot
                time.sleep(0.01)
            raise AssertionError(("paused runner did not become idle", snapshot))

        def job(payload):
            identifier = request("/command", payload, 202)["id"]
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                result = request(f"/jobs/{identifier}")
                assert result["state"] != "failed", result
                if result["state"] == "completed":
                    return request(f"/jobs/{identifier}/result")
                time.sleep(0.01)
            raise AssertionError(("GPU job timed out", result))

        def gpu(payload):
            return job({"op": "app.gpu_command", "payload": payload})["result"]

        def resumed(before):
            result = wait_for(
                lambda s: s["progress"]["submitted_ticks"] > before,
                "automatic simulation did not resume",
            )
            if not args.per_redraw:
                # Measure in the App hook rather than across HTTP: even after
                # a long pause the first automatic tick waits one fresh dt.
                delay = result["app"]["first_tick_delay_micros"]
                assert delay is not None and delay >= args.tick_micros, (delay, args.tick_micros)
            return result

        try:
            deadline = time.monotonic() + 20
            while not connection.exists():
                assert proc.poll() is None, (output / "app.log").read_text()
                assert time.monotonic() < deadline, "agent connection file was not written"
                time.sleep(0.02)
            base = "http://" + json.loads(connection.read_text())["addr"]
            initial = wait_for(
                lambda s: s["ready"] and s["progress"]["completed_ticks"] > 0
                and s["progress"]["presented_frames"] > 0,
                "initial simulation/presentation did not start", timeout=20,
            )
            assert initial["app"]["control_installs"] == 1, initial

            assert control("pause")["simulation_paused"]
            paused = wait_for(lambda s: s["app"]["simulation_paused"], "pause was not observed")
            tick = paused["progress"]["submitted_ticks"]
            paused = idle(tick)
            time.sleep(0.25)
            assert status()["progress"] == paused["progress"], "paused idle still submitted GPU work"

            # A later worker result uses the public redraw handle, with no
            # polling tick. Each resulting native submission is a render frame.
            command({"worker_snapshot": 17})
            wait_for(lambda s: s["app"]["rendered_snapshot"] == 17,
                     "worker redraw did not prepare the new snapshot")
            redrawn = idle(tick)
            assert redrawn["progress"]["submitted_frames"] > paused["progress"]["submitted_frames"]
            assert (redrawn["progress"]["submitted_submissions"] - paused["progress"]["submitted_submissions"]
                    == redrawn["progress"]["submitted_frames"] - paused["progress"]["submitted_frames"])

            # Existing GPU commands/readbacks remain live during public pause;
            # explicit diagnostic ticks are also independent of this gate.
            assert gpu({"reset": 42})["values"] == [42] * 4
            assert status()["progress"]["submitted_ticks"] == tick
            target = tick + 3
            stepped = job({"op": "run_until_completed", "target_tick": target, "dt": 0.0004})
            assert stepped["result"]["completed_tick"] == target, stepped
            assert gpu({})["values"] == [45] * 4
            assert idle(target)["app"]["simulation_paused"]

            # run_until_completed leaves the diagnostic pause in effect. Public
            # resume cannot override it, even though its requested state changes.
            assert not control("resume")["simulation_paused"]
            idle(target)
            diagnostic("resume")
            resumed(target)

            # Conversely, diagnostic resume must respect a public pause.
            control("pause")
            stopped = idle()
            diagnostic("pause")
            diagnostic("resume")
            idle(stopped["progress"]["submitted_ticks"])

            # A CPU -> GPU transfer is completed before public resume. Pausing
            # inside the first tick must stop the next tick in the same batch.
            assert gpu({"reset": 500})["values"] == [500] * 4
            before_tick = status()["progress"]["submitted_ticks"]
            control("pause_on_tick")
            from_tick = wait_for(
                lambda s: s["app"]["pause_hook"] == "tick" and s["app"]["simulation_paused"],
                "tick did not request pause",
            )
            marker = from_tick["app"]["pause_tick"]
            assert marker == before_tick + 1, (before_tick, from_tick)
            idle(marker)
            assert gpu({})["values"] == [501] * 4

            # Completion hooks can also change scheduling. Existing submissions
            # still finish, but no new tick may follow the recorded boundary.
            control("pause_on_complete")
            from_complete = wait_for(
                lambda s: s["app"]["pause_hook"] == "after_complete"
                and s["app"]["simulation_paused"],
                "completion hook did not request pause",
            )
            idle(from_complete["app"]["pause_tick"])

            # Coalesced pause+resume still establishes a fresh wall-time anchor.
            before_rapid = status()["progress"]["submitted_ticks"]
            control("pause_resume")
            final = resumed(before_rapid)
            control("pause")
            final_idle = idle()
            print(json.dumps({"ok": True, "mode": mode, "tick_micros": args.tick_micros,
                              "first_tick_delay_micros": final["app"]["first_tick_delay_micros"],
                              "progress": final_idle["progress"], "output": str(output)}))
            request("/command", {"op": "shutdown"})
            proc.wait(timeout=5)
            assert proc.returncode == 0, (output / "app.log").read_text()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == "__main__":
    main()
