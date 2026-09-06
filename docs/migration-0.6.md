# Migrating from mikage 0.5 to 0.6

0.6 intentionally changes public APIs. Update applications and any direct wgpu
usage together: use wgpu 30.0.1 (or `mikage::wgpu`), egui 0.36.1 and Rust 1.95+.
No old App compatibility adapter is installed.

## Initialization and pipeline targets

The factory is now `FnOnce(&GpuContext, RenderTargetConfig, PhysicalSize<u32>)`:

```rust,ignore
mikage::run(|gpu, target, size| MyApp::new(gpu, target, size),
            mikage::RunConfig::new("My app"))?;
```

`GpuContext` owns a logical GPU, not a render format. Move calls to
`gpu.render_format()`, `sample_count()` and `render_target_config()` to the target.
Pass the target as the second argument of `SolidRenderer::new`,
`InstanceRenderer::new` and `InstanceRenderer::with_shader`.
`create_depth_texture(gpu, size, target)` uses its depth format and sample count.
Retain the target config in the App if depth resources must be recreated on resize.

Replace `GpuContextDescriptor` and `headless(format, samples)` with
`GpuContext::headless(GpuDescriptor::default()).await?`. Create an `OffscreenTarget`
separately when you actually render. `HeadlessHarness` combines these steps for
App tests and offline capture, allocating its render target lazily.

Move `RunConfig.wgpu_features` and `wgpu_limits` into
`config.gpu.requirements.required_features` and `required_limits`. Use
`optional_features` for opportunistic timestamps and `preferred_limits` for a
faster pipeline variant with a portable fallback. Inspect `gpu.capabilities()`
before choosing a pipeline layout. Required limits are never silently reduced.

`run` and `run_with_agent` now return `Result<(), RunError>`. Propagate or handle
that result so failed GPU initialization, Surface recovery and device loss are
distinguishable from normal window closure. On WASM, `run` returns after starting
the browser event loop; later failures stop the app and show an error overlay.

## Split the former update/encode methods

| Old responsibility | New method |
|---|---|
| Physics time/state, simulation uniforms, compute dispatch | `tick(TickContext)` |
| Camera matrices and other view-dependent uploads | `prepare_render(RenderUpdateContext)` |
| Scene draw calls | `render(RenderContext)` |
| Input state and window controls | `on_input(WindowInputContext)` |
| UI | `gui(&mut egui::Ui)` |

`tick` has an encoder and explicit `dt`, `elapsed`, `tick_id`. It has no window or
presentation target. Move compute dispatches out of the old mixed `encode` method.
Each tick gets a separate submission, so ordinary queue uploads remain ordered.
Never advance physics from `prepare_render` or `render`: these run while paused
and during capture.

Use `ctx.target.size` in rendering, `ctx.target_size` in render preparation, and
`ctx.color_attachment(...)` for correct MSAA resolve. Keep one-shot input actions
in App state until a tick consumes them; do not clear them merely because a frame
rendered. See `examples/boids.rs` for the full compute/render split.

For egui windows use `ui.ctx()`; panel APIs in egui 0.36 use the root `Ui`.
Keep `#[cfg(feature = "gui")]` on an optional App GUI method if your own crate
forwards the feature. Pure external/headless consumers need no egui dependency.

## Explicit progress and stopping

Use `AppRuntime::advance_ticks(n, dt)` to submit exactly N additional ticks without
rendering, then `wait_for(&token)` for GPU completion. WASM provides asynchronous
variants. `render` does not tick. `after_submit` receives a SubmissionToken and
`after_complete` runs on the runtime owner after GPU completion.

Ticks, frames and submissions have separate counters. The old agent `frame_count`
now aliases submitted render frames; use `/status.progress` for precise metrics.
Presented means `Queue::present` was called, not physical display completion.
Cloned queues refer to the same logical queue; app-owned raw submits are not
included in runtime counters.

Fixed scheduling is opt-in:

```rust,ignore
let config = mikage::RunConfig::new("fixed steps")
    .with_simulation_policy(mikage::SimulationPolicy::fixed(
        std::time::Duration::from_secs_f64(1.0 / 60.0)));
```

## External surfaces

Use `GpuContext::for_surface` for a safe owned/borrowed target, or
`unsafe { GpuContext::for_surface_unsafe(target, gpu_descriptor, surface_descriptor).await }`.
Both return GPU and SurfaceContext separately. The latter supports a host-owned
`SurfaceTargetUnsafe::CoreAnimationLayer`; keep its native objects alive until all
surfaces and acquired frames are gone. Do not construct a winit Window/EventLoop.

Acquire from `surface.acquire_surface_texture()`, handle `CurrentSurfaceTexture`,
create the sRGB-aware view with `surface.create_view`, submit, then
`gpu.queue.present(frame)`. Resize through `surface.resize(&gpu, size)?`.
A zero size suspends acquisition. Lost requires `gpu.attach_surface` or its unsafe
counterpart, after dropping the previous frame/surface. Device loss requires
application teardown/reconstruction by the host.

The independent macOS fixture under `tests/fixtures/external-surface` demonstrates
native layer ownership and teardown without adding Cocoa dependencies to mikage.

## Diagnostics and capture

Use `POST /captures` to obtain a 202 job and polling URL. `scene` captures before
GUI composition; `window` includes GUI. Register named buffers/textures through
`App::capture_targets`. Buffers and supported textures outside the PNG formats use raw results with metadata.
`GET /screenshot` remains a synchronous HTTP wrapper; it no longer blocks the
render thread on PNG encoding. `scripts/capture.py` validates responses before
saving them. See `docs/agent-http-api.md` for exact job and command formats.

GPU commands use `encode_agent_command`, followed by `complete_agent_command`
after their submission completes. Use the latter to return readback-derived JSON.
`run_until_completed` pauses at its target; send `runtime.resume` to
continue. `shutdown` runs once while the GPU is still available for worker cleanup.
