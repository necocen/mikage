# v0.7.2 verification

Issue [#2](https://github.com/necocen/mikage/issues/2) reported that reactive
windows stopped after one camera update and applied preceding idle time to newly
started animations. Validation ran on macOS with Rust 1.95.0.

## Automated regression coverage

```sh
cargo fmt --all -- --check
cargo test --offline --lib --features agent -- --skip readback::tests::gpu_
cargo test --offline --doc --features agent
cargo clippy --offline --lib --features agent -- -D warnings
bash scripts/check-features.sh
```

The library test run passed 143 tests, including all 56 camera tests and nine
runner tests. The two pre-existing GPU readback tests were excluded to keep this
run CPU-only. HTTP transport tests require local TCP sockets; the sandbox denied
those sockets, so the successful run used ordinary host permissions. Doctests
passed two executable examples and one compile-fail example; four existing
examples remain marked ignored.

Strict library Clippy passed with warnings denied. All 20 native/WASM feature
configurations and core GUI/HTTP dependency isolation checks passed. The WASM
checks needed ordinary host permissions to unpack dependencies in Cargo's
cache; no dependency specifications were changed for this patch.

Regression cases cover one-shot smooth zoom, released pan/orbit inertia,
disabled input while animations settle, no-damping and held-drag idle states,
long idle before new input/release, bounded camera dt, and independent continuous
and GUI repaint scheduling. App invalidation tests cover worker-thread requests,
coalescing, and a new request after the runner drains pending work.

Zoom tests exercise 60, 360, and 1000 Hz timesteps, including zoom 999 → 1000,
small smoothing, zero/tiny first dt, and changing smoothing while a target is
pending. Fractional interpolation progress is retained between updates so f32
rounding cannot keep the reactive loop awake indefinitely. Zoom anchor positions
remain stable through interpolation and final snapping.

## Light native window smoke

A temporary standalone fixture used only the `window,agent` features, a clear-only
scene, `Camera2d` with damping 0.95 and zoom smoothing 0.2,
`RedrawPolicy::Reactive`, and `SimulationPolicy::Manual`. The requested window
was 320 × 240 logical pixels (640 × 480 physical pixels on the test display).
There were no GUI repaint requests or simulation workloads.

| Check | Observed behavior |
| --- | --- |
| Initial idle | Frame count remained at 1. |
| One-shot zoom | Advanced from frame 1 to 86, reached zoom 1.10000002, then stopped. |
| Held drag and release | Held drag stopped at frame 87; release inertia advanced to frame 390 and stopped. |
| Worker invalidation | Woke from idle and rendered the worker's changed visible state, ending at frame 392. |
| Invalidation during render | Scheduled a subsequent frame, ending at frame 394. |
| New zoom after 1.5 s idle | First frame covered 5.24% of the target gap, then reached zoom 1.21000004 at frame 467 and stopped. |
| Manual simulation | Encoded/submitted/completed ticks stayed at zero; all 467 frames completed. |

The HTTP API has no drag-release operation, so a fixture camera wrapper called
the actual `Camera2d::on_drag_end` from an app-command flag on its next update.
This checked native redraw scheduling rather than physical mouse delivery;
the deterministic runner test separately checks a release after 60 s idle.
The window and HTTP listener were shut down after verification.

Native smoke coverage is limited to this macOS host. Browser behavior was
compile-checked; browser interaction and other native window backends were not
executed for this patch.
