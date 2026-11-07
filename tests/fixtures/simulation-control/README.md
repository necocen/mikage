# Automatic tick control in a browser window

This fixture uses the real winit/WebGPU window runner, Reactive drawing and VSync
presentation. The default Fixed policy uses 400 µs ticks, at most four ticks per
wake and 1.6 ms of backlog. Its ticks intentionally leave their GPU encoders empty.

It checks that public pause stops automatic ticks and their submissions, drains
existing GPU work, and reaches an idle without GPU submissions. A CPU-only Web
Worker then sends a snapshot to the window host, which requests a redraw and
renders the new value without a tick. Resume retains fixed dt and excludes paused
wall time. Reentrant pause requests from `tick`, `after_submit`, `after_complete`
and `gui` must stop the next automatic tick; a rapid pause/resume must retain its
wake. `PerRedraw` and `Manual` use the same public control API.

Build from the repository root using the installed wasm-bindgen CLI 0.2.128:

```sh
CARGO_TARGET_DIR=target cargo build \
  --manifest-path tests/fixtures/simulation-control/Cargo.toml \
  --target wasm32-unknown-unknown --release
mkdir -p target/simulation-control/dist
wasm-bindgen target/wasm32-unknown-unknown/release/mikage_simulation_control.wasm \
  --target web --out-dir target/simulation-control/dist --out-name mikage_simulation_control
cp tests/fixtures/simulation-control/{index.html,probe.js,snapshot-worker.js} \
  target/simulation-control/dist/
python3 -m http.server 8769 --bind 127.0.0.1 --directory target/simulation-control/dist
```

Open `http://127.0.0.1:8769/?policy=fixed` in a foreground WebGPU-enabled browser.
Repeat with `policy=per-redraw` and `policy=manual`. Each must report `"ok": true`;
browser automation can read `globalThis.__simulationControlResult`. Test one
foreground tab at a time because background throttling changes callback timing.
The counters describe App hooks and framework submissions, not physical scanout.
