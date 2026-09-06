# mikage

A lightweight GPU application framework for windows, external surfaces, and
headless execution. Create pipelines, buffers, and bind groups directly through
`mikage::wgpu`; the framework supplies application scheduling, presentation,
cameras, rendering helpers, and optional egui and HTTP diagnostics.

Version 0.6 uses **wgpu 30.0.1**, **egui 0.36.1**, and **winit 0.30.13** and
requires **Rust 1.95+**. See the [0.5 → 0.6 migration guide](docs/migration-0.6.md)
for the public API changes.

## Quick start

Use this checkout while the 0.6 vendored compatibility patch is in place
(adjust the path to your checkout):

```toml
[dependencies]
mikage = { path = "../mikage" }
```

```rust
use mikage::{App, OrbitCamera, RenderContext, RunConfig, wgpu};

struct MyApp;

impl App for MyApp {
    type Camera = OrbitCamera;

    fn render(&mut self, ctx: &mut RenderContext<'_, OrbitCamera>) {
        let attachment = ctx.color_attachment(wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            store: wgpu::StoreOp::Store,
        });
        let _pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear"),
            color_attachments: &[Some(attachment)],
            ..Default::default()
        });
    }
}

fn main() {
    mikage::run(
        |_gpu, _target, _size| MyApp,
        RunConfig::new("My app"),
    ).expect("mikage run failed");
}
```

The factory runs after GPU initialization and receives
`(&GpuContext, RenderTargetConfig, dpi::PhysicalSize<u32>)`. Create GPU resources
there using the supplied target format and sample count.

## Features and platforms

| Feature | Enabled by default | Provides |
| --- | --- | --- |
| Core, with no features | Always | GPU selection, external surfaces, offscreen targets, `AppRuntime`, headless harness, cameras, renderers, shaders, raw readback |
| `window` | Via `window-gui` | winit runner, window input, redraw and simulation scheduling |
| `gui` | Via `window-gui` | egui, egui-wgpu, and the `App::gui` hook for hosts that integrate GUI themselves |
| `window-gui` | Yes | `window` + `gui` + automatic egui-winit input and rendering integration |
| `agent` | No | Native HTTP jobs, PNG/raw capture worker, and host-driven agent bridge; works without `window` |
| `webgl` | No | Optional WebGL2 fallback on WASM |

`window` and `gui` are independent; select `window-gui` for their automatic
integration. For a custom native host or GPU-only application:

```toml
mikage = { path = "../mikage", default-features = false }
# Add features = ["window"] for a winit app without egui.
# Add features = ["agent"] for native diagnostics without a winit event loop.
```

Native backends include Metal, Vulkan, and DX12. Browser applications use WebGPU;
WebGL2 supports rendering but cannot run compute shaders. The `agent` HTTP API is
native-only. Core readback and async runtime operations are also available on WASM.

## Application lifecycle

Only the `Camera` associated type is mandatory; every lifecycle hook has a
default implementation. Choose `OrbitCamera`, `Camera2d`, `()`, or a custom
`InteractiveCamera`.

| Hook | Responsibility |
| --- | --- |
| `on_input(&mut WindowInputContext)` | Convert window input into app-owned controls and pending actions (`window`) |
| `tick(&mut TickContext)` | Advance simulation and encode compute for one explicit timestep |
| `prepare_render(&mut RenderUpdateContext<Camera>)` | Upload camera matrices and view-dependent state |
| `render(&mut RenderContext<Camera>)` | Encode scene drawing into the supplied target |
| `gui(&mut egui::Ui)` | Build UI and update parameters or pending actions (`gui`) |
| `resize(&GpuContext, PhysicalSize<u32>)` | Recreate app-owned size-dependent resources |
| `on_window_event(&WindowEvent)` | Handle other window events, such as file drops (`window`) |
| `after_submit(&GpuContext, &SubmissionToken)` | Record the submitted queue endpoint |
| `after_complete(&GpuContext, &SubmissionToken)` | Consume GPU completion on the runtime owner thread |
| `capture_targets(&mut CaptureRegistry)` | Register named textures or buffer regions for capture |
| `shutdown(&GpuContext)` | Stop app-owned workers before GPU resources are released |

`TickContext` contains `gpu`, `encoder`, `tick_id`, `dt`, and simulation `elapsed`.
It has no window or camera dependency. `RenderUpdateContext` supplies a read-only
camera, target size/configuration, wall-time `dt`, and simulation `elapsed`.
`RenderContext` supplies
the encoder, camera, frame identity, and borrowed `RenderTarget`; use
`ctx.target.size` for dimensions and `ctx.color_attachment(ops)` for MSAA resolve.
`WindowInputContext` supplies the window, filtered input state, and mutable camera.

Simulation and rendering are independent. A host can run zero or several ticks
before a render, and rendering a paused app does not advance simulation. Each
tick has its own queue submission, preserving the order of per-tick uploads and
compute passes. Keep transient input actions in app state until a tick consumes
them. See [runtime and completion semantics](docs/runtime.md).

## GPU and render destinations

`GpuContext` owns the device, queue, adapter, and instance. It can serve several
surfaces and offscreen targets. `RenderTargetConfig` describes one destination's
color format, depth format, and MSAA sample count; it supplies
`color_target_state`, `multisample_state`, and `depth_stencil_state` helpers for
pipeline construction.

Configure GPU selection through `GpuDescriptor` and its `GpuRequirements`:

```rust
use mikage::{GpuDescriptor, GpuRequirements, wgpu};

let descriptor = GpuDescriptor {
    requirements: GpuRequirements {
        optional_features: wgpu::Features::TIMESTAMP_QUERY,
        ..Default::default()
    },
    ..Default::default()
};
```

Required features and limits fail initialization when unsupported. Optional
features and preferred limits are negotiated against the adapter. Inspect
`gpu.capabilities()` for separate supported and enabled features/limits before
selecting a pipeline variant. Pass the descriptor through `RunConfig::gpu` or
`with_gpu` for a window, or to `GpuContext::headless(descriptor).await?` when a
surface is unnecessary.

`OffscreenTarget` owns color/MSAA/depth textures. `SurfaceContext` owns surface
configuration and presentation resources. Both provide the target configuration;
renderers accept it explicitly alongside `&GpuContext`.

## Runtime and headless execution

`AppRuntime::new(gpu, app, camera)` owns application execution without an event
loop or surface. `advance_ticks(n, dt)` submits exactly N additional ticks and
returns the last submission endpoint; `wait_for` waits for its GPU completion.
`render(target)` prepares and renders once with zero additional ticks. Async
stepping and completion methods support browser and async hosts.

Tick, frame, and submission identifiers are independent. `RuntimeProgress`
distinguishes encoded, submitted, and completed work. Presented frames count
calls to `Queue::present`, rather than GPU completion or display scanout.
Nonblocking operations return `RuntimeError::WouldBlock` when the configured
submission capacity is full; completion notifications can wake the host.

`HeadlessHarness` combines an app factory, runtime, lazy offscreen target, and
named raw capture. It can capture registered compute buffers without allocating
a render texture. `render_once` renders explicitly, and `capture_named("scene")`
captures that render without advancing simulation or rendering again.

```sh
cargo run --no-default-features --example headless
```

The [headless example](examples/headless.rs) advances 120 exact ticks, waits for
completion, checks a named buffer, and captures a scene. See also the
[runtime guide](docs/runtime.md) and [GPU runtime tests](tests/runtime.rs).

## External surfaces

Use `GpuContext::for_surface` with a safe owned or borrowed surface target, or
`GpuContext::for_surface_unsafe` with a host-owned raw target such as a
`SurfaceTargetUnsafe::CoreAnimationLayer`. Both return a GPU and `SurfaceContext`
separately and accept a `SurfaceDescriptor` containing the initial size and
presentation options. This supports embedding into an existing application or
screensaver host without creating a winit window or event loop.

The host owns acquisition, rendering, `gpu.queue.present(frame)`, resize, and
teardown. Keep native objects alive until all acquired textures and the surface
have been released. A zero size suspends acquisition. Recreate a lost surface
through `attach_surface` or `attach_surface_unsafe` on the existing GPU.

The [standalone macOS CAMetalLayer fixture](tests/fixtures/external-surface/README.md)
exercises ownership, MSAA, resize, suspension, recreation, and teardown without
winit or egui dependencies. Native ScreenSaverView/FFI implementation belongs to
the embedding application. See the [external surface migration notes](docs/migration-0.6.md#external-surfaces).

## Window configuration

`RunConfig<C>` configures the window runner. Use
`RunConfig::new("title").with_camera(camera)` for a custom camera, or
`RunConfig::with_defaults(())` for a camera-free app.

| Field | Default | Purpose |
| --- | --- | --- |
| `title` | `"mikage"` | Window title |
| `width` / `height` | 1280 / 720 | Initial window size |
| `camera` | `OrbitCamera` | Camera controller |
| `gpu` | `GpuDescriptor::default()` | Adapter selection and feature/limit requirements |
| `present_mode` | `AutoVsync` | Presentation mode |
| `sample_count` | 1 | MSAA sample count |
| `redraw_policy` | `Continuous` | Continuous or reactive presentation |
| `simulation_policy` | `PerRedraw` | Per-redraw, fixed-step, or manual simulation |
| `max_in_flight_submissions` | 8 | Bounded runtime queue capacity |
| `init_logging` | `true` | Initialize tracing (`RUST_LOG` on native, browser console on WASM) |
| `canvas` | `None` | CSS selector for an existing WASM canvas |
| `pixel_scroll_per_line` | 50.0 | Pixel-wheel normalization |
| `touch_pinch_sensitivity` | 5.0 | Touch pinch zoom multiplier |

Fixed simulation has its own timer and bounded catch-up work, independent of
redraw scheduling:

```rust
let config = mikage::RunConfig::new("fixed steps")
    .with_simulation_policy(mikage::SimulationPolicy::fixed(
        std::time::Duration::from_secs_f64(1.0 / 60.0),
    ));
```

`SimulationPolicy::Manual` advances only through explicit requests.
Capture-only redraws do not introduce an extra tick.

## Captures and agent HTTP API

`ReadbackRing` records buffer/texture copies and maps them after submission. Its
three default staging slots are reusable and return `Busy` when occupied. Map
callbacks only signal readiness; the caller consumes mapped bytes and removes
row padding through `take_ready`. Core readback returns bytes without an image
encoding dependency.

Enable `agent` and call
`run_with_agent(factory, run_config, AgentConfig::default())` for the native
HTTP API at `127.0.0.1:3939`. It exposes status, camera controls, deferred app GPU
commands, and bounded asynchronous capture jobs. `scene` excludes GUI; `window`
includes it. Register other textures/buffer regions through `capture_targets`.
PNG encoding and capture readback run on a worker thread. Raw captures preserve
the source channel order and include tick/frame/submission metadata.

```sh
cargo run --example agent_capture --features agent
python3 scripts/capture.py /tmp/mikage.png
python3 scripts/capture.py /tmp/values.bin --target values --format raw --exact
```

`POST /captures` returns HTTP 202 and a job id; poll `/jobs/{id}` and fetch
`/jobs/{id}/result`. `/screenshot` remains a synchronous HTTP compatibility
wrapper. `run_until_completed` pauses at an absolute tick boundary after GPU
completion; `runtime.resume` restarts the configured simulation policy.
Hosts without winit can drive `agent::AgentBridge` and `AgentCaptureWorker`
themselves. See the [HTTP API guide](docs/agent-http-api.md),
[capture example](examples/agent_capture.rs), and [Python client](scripts/capture.py).

## Helpers

### wgpu boilerplate reducers

| Name | Purpose |
|------|---------|
| `storage_buffer_entry(binding, visibility, read_only)` | 1-line `BindGroupLayoutEntry` for storage buffers |
| `uniform_buffer_entry(binding, visibility)` | 1-line `BindGroupLayoutEntry` for uniform buffers |
| `UniformBuffer<T: Pod>` | Typed uniform buffer with `new` / `write` / `buffer()` |
| `create_storage_buffer_init(device, label, data)` | Create `STORAGE \| COPY_DST` buffer from `&[T]` |
| `MeshBuffers::from_position_normal(...)` | Interleave positions+normals into vertex/index buffers |
| `POSITION_NORMAL_LAYOUT` | `VertexBufferLayout` for interleaved `Float32x3` position + normal (stride 24) |
| `create_compute_pipeline(device, label, wgsl, bgls, entry)` | Create compute pipeline from WGSL source in one call |

### Scene & depth

| Name | Purpose |
|------|---------|
| `SceneBinding` | Bundles SceneUniform buffer + bind group layout + bind group |
| `SceneUniform` | View-projection + camera position + lighting uniform struct |
| `create_depth_texture(gpu, size, target)` | Create depth texture + view |
| `DEPTH_FORMAT` | `Depth32Float` constant |

### Mesh generators

| Name | Purpose |
|------|---------|
| `IcoSphereMesh::generate(n)` | Generate icosphere mesh with n subdivisions |
| `CubeMesh::generate()` | Generate unit cube mesh with per-face normals |
| `PlaneMesh::generate()` | Generate unit plane mesh (XZ plane, +Y normal) |
| `QuadMesh2d::generate()` | Generate unit quad mesh (XY plane, +Z normal) |
| `RegularPolygonMesh::generate(n)` | Generate regular n-sided polygon mesh |

## Renderers

### SolidRenderer

Construct renderers with `&GpuContext` and the destination's `RenderTargetConfig`.
For example, `SolidRenderer::new(gpu, target, scene.layout())`.

Renders solid-colored meshes with per-object model matrix and RGBA color. Objects with alpha >= 1.0 use the lit (Lambert diffuse) pipeline; alpha < 1.0 uses the unlit pipeline.

### InstanceRenderer

Renders many copies of a single mesh with per-instance data. Generic over the `InstanceVertex` trait — use the built-in `InstanceData` (position + scale + color, 32 bytes) or define your own layout:

```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileInstance {
    pos_angle_scale: [f32; 4], // xy=position, z=angle, w=scale
}

impl InstanceVertex for TileInstance {
    fn vertex_attributes() -> Vec<wgpu::VertexAttribute> {
        vec![wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 0,
            shader_location: 2,
        }]
    }
}

// Use with a custom shader:
let renderer = InstanceRenderer::<TileInstance>::with_shader(
    gpu, target, scene.layout(),
    &mesh.positions, &mesh.normals, &mesh.indices,
    &resolved_shader, config,
);
```

## Compute shaders

Create buffers, bind groups, and pipelines with raw wgpu or the helpers above.
Encode simulation compute work in `tick`:

```rust,ignore
fn tick(&mut self, ctx: &mut mikage::TickContext<'_>) {
    self.params_buffer.write(&ctx.gpu.queue, &self.params);
    let mut pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("simulation"),
        ..Default::default()
    });
    pass.set_pipeline(&self.pipeline);
    pass.set_bind_group(0, &self.bind_group, &[]);
    pass.dispatch_workgroups(self.num_elements.div_ceil(64), 1, 1);
}
```

Encode drawing in `render` and view-dependent uploads in `prepare_render`.
[The boids example](examples/boids.rs) demonstrates double-buffered storage and
compute-to-instance rendering. Add `COPY_SRC` when creating buffers that should
be available to readback or named capture.

## Shader Processor

Lightweight WGSL preprocessor that resolves `#import` directives:

```rust
let mut sp = ShaderProcessor::new();
sp.register("mikage::scene_types", mikage::SCENE_TYPES_WGSL);
sp.register("my_app::utils", include_str!("shaders/utils.wgsl"));
let resolved = sp.resolve(include_str!("shaders/main.wgsl"))?;
```

In WGSL files:
```wgsl
#import mikage::scene_types
@group(0) @binding(0) var<uniform> scene: SceneUniform;
```

- Modules are hoisted to the top of the output in dependency order
- Each module is expanded exactly once (deduplication)
- Recursive imports and circular dependency detection
- `#import module::{Item1, Item2}` syntax accepted (imports full module)

## Cameras and input

`Camera` exposes read-only matrices and position to rendering. `InteractiveCamera`
adds mouse, touch, trackpad, and inertia hooks for a host to drive. Window egui
integration filters UI-consumed input before camera handling.

| Camera | Use case | Mouse | Touch |
| --- | --- | --- | --- |
| `OrbitCamera` | 3D | Left drag: orbit; right drag: pan; scroll: zoom | One finger: orbit; two fingers: pinch zoom + pan |
| `Camera2d` | 2D | Left drag: pan; scroll: zoom | One finger: pan; two fingers: pinch zoom |
| `()` | Camera-free apps | Ignored | Ignored |

## WASM

```sh
trunk build
trunk serve
```

GPU initialization is asynchronous. WebGPU is used by default; enable `webgl`
for a WebGL2 fallback when no WebGPU adapter is available. WebGL2 has no compute
shader support, so the boids example requires WebGPU. The egui-winit dependency
is vendored with a minimal WASM compatibility fix; see
[vendor/egui-winit](vendor/egui-winit).

## Examples

```sh
cargo run --example clear              # Color-cycling clear screen
cargo run --example egui_demo          # egui UI
cargo run --example orbit_camera       # IcoSphere + orbit camera
cargo run --example instancing_2d      # Hex grid with Camera2d
cargo run --example instancing_3d      # Sphere grid with wave animation
cargo run --example custom_instance    # Custom instance vertex layout
cargo run --example boids              # GPU compute flocking
cargo run --no-default-features --example headless
cargo run --example agent_capture --features agent
```

See [0.6 validation and reproducible checks](docs/verification-0.6.md) for the
feature matrix, GPU tests, external-surface host and browser coverage.
