# Browser completion regression

This WebGPU fixture exercises `AppRuntime` without winit, rendering, or a host
polling loop. It holds actual GPU completion notifications until an empty submit
encourages their delivery. `slow` holds them for 100 ms even when probed, checking
that retries never bypass the two-submission limit. Shutdown, Drop, and device
loss must cancel the timer. Probe submits must not add tokens or callbacks.

Build from the repository root with wasm-bindgen CLI 0.2.128:

```sh
CARGO_TARGET_DIR=target/web-completion/build cargo build \
  --manifest-path tests/fixtures/web-completion/Cargo.toml \
  --target wasm32-unknown-unknown --release
mkdir -p target/web-completion/dist
wasm-bindgen target/web-completion/build/wasm32-unknown-unknown/release/mikage_web_completion.wasm \
  --target web --out-dir target/web-completion/dist --out-name mikage_web_completion
cp tests/fixtures/web-completion/index.html tests/fixtures/web-completion/probe.js \
  target/web-completion/dist/
python3 -m http.server 8767 --bind 127.0.0.1 --directory target/web-completion/dist
```

Open `http://127.0.0.1:8767/?case=delayed` in Firefox and Chrome. Repeat with
`case=normal`, `slow`, `shutdown`, `drop`, `device-loss`, and `timer-error`.
Add `&worker=1` to repeat in a module Web Worker without a browser `Window`.
Each page must report `"ok": true`; browser automation can read
`globalThis.__webCompletionResult`.
Use one foreground browser at a time when comparing timing. The `normal` case
uses unmodified native notification timing, so slow implementations may still
need probes. Timings describe host notification delivery, not GPU timestamps.

For window scheduling and real workloads, also compare the same optimized WASM
in Firefox and Chrome using Fixed 0.4 ms, max 4 ticks/wake, max backlog 1.6 ms,
8 submission slots, Reactive + Vsync, and a 1280×720 DPR1 viewport. Instrument
`GPUQueue.submit` and `onSubmittedWorkDone` before loading WASM, then compare the
progress path enabled/disabled/enabled, suppressing only empty submits for the
disabled comparison. Seed the workload identically and measure the same tick
range in each run (for example, 5,000 warmup ticks and 10,000 measured ticks).
The nominal 2 ms threshold does not promise timer accuracy; nested timers may
be clamped to 4 ms or delayed further. Report simulation speed,
render submissions/s, notification mean/p95, and max pending notifications;
retain errors and adapter information to verify hardware acceleration.
