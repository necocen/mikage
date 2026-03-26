# mikage

A lightweight application framework built on wgpu + winit.

Provides GPU rendering, compute shaders, and egui UI integration for both Native and WASM (WebGPU) targets. Unlike a full engine, mikage doesn't abstract over wgpu — you create your own pipelines, buffers, and bind groups directly.

## Features

- **Raw wgpu access** — The framework manages the window and surface; you manage everything else.
- **egui integration** — Build UI in `gui()`. Input lock between egui and camera is automatic.
- **Camera system** — `Camera` trait (read-only) + `InteractiveCamera` trait (input handling). Built-in `OrbitCamera` and `Camera2d`.
- **Compute shaders** — Encode compute passes in `encode()`, before render passes.
- **SolidRenderer** — Opaque (lit) and transparent (unlit) pipelines for solid-colored meshes.
- **InstanceRenderer** — GPU-instanced rendering with generic per-instance vertex data via `InstanceVertex` trait.
- **ShaderProcessor** — Lightweight WGSL preprocessor with `#import` resolution, recursive dependency handling, and circular import detection.
- **Multi-platform** — Native (Metal / Vulkan / DX12) and WASM (WebGPU).
- **tracing** — `RUST_LOG` on native, `tracing-web` (browser console) on WASM.

## Quick Start

```rust
use mikage::{App, FrameContext, OrbitCamera, RunConfig, UpdateContext};

struct MyApp;

impl App for MyApp {
    type Camera = OrbitCamera;

    fn update(&mut self, _ctx: &mut UpdateContext<OrbitCamera>) {}

    fn encode(&mut self, ctx: &mut FrameContext<OrbitCamera>) {
        let _pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.surface_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
    }
}

fn main() {
    mikage::run(|_ctx, _size| MyApp, RunConfig::default());
}
```

## App Trait

Each app specifies its camera type via an associated type:

```rust
impl App for MyApp {
    type Camera = OrbitCamera;  // or Camera2d, or your own
    fn update(&mut self, ctx: &mut UpdateContext<OrbitCamera>) { /* ... */ }
    fn encode(&mut self, ctx: &mut FrameContext<OrbitCamera>) { /* ... */ }
}
```

| Method | When called | Required |
|--------|------------|----------|
| `type Camera` | Associated type | Yes |
| `update()` | Every frame (logic) | Yes |
| `encode()` | Every frame (compute + render) | Yes |
| `gui()` | Every frame (egui UI) | No |
| `resize()` | On window resize | No |
| `on_window_event()` | On unhandled window events (file drops, focus, etc.) | No |

## run()

```rust
mikage::run(
    |ctx, size| MyApp::new(ctx, size),  // factory closure (called after GPU init)
    RunConfig::default(),
);
```

GPU resources can be created directly in the factory closure — no `Option<T>` needed.

## Frame Loop

```
winit events → InputState → egui input
→ Camera update → App::update()
→ App::encode() → App::gui() → egui render
→ submit → present
```

## Context Types

### `UpdateContext<C: InteractiveCamera>`
- `dt: f32` — seconds since last frame
- `elapsed: f64` — total elapsed time
- `gpu: &GpuContext` — device / queue
- `input: &InputState` — keyboard / mouse state
- `camera: &mut C` — mutable camera controller

### `FrameContext<C: Camera>`
- `gpu: &GpuContext` — device / queue (`gpu.device`, `gpu.queue`)
- `encoder` — encode compute and render passes
- `surface_view` — render target
- `window_size` — current window size
- `camera: &C` — read-only camera

## RunConfig

`RunConfig<C: InteractiveCamera>` is generic over the camera type (defaults to `OrbitCamera`).

```rust
// OrbitCamera (default) — supports Default
let config = RunConfig {
    title: "my app".to_string(),
    camera,
    ..Default::default()
};

// Camera2d — use with_defaults or builder
let config = RunConfig::new("boids").with_camera(camera);
let config = RunConfig { title: "boids".to_string(), ..RunConfig::with_defaults(camera) };
```

| Field | Default | Purpose |
|-------|---------|---------|
| `title` | `"mikage"` | Window title |
| `width` / `height` | 1280 / 720 | Initial window size |
| `camera` | `OrbitCamera` | Camera controller (generic `C: InteractiveCamera`) |
| `present_mode` | `AutoVsync` | wgpu presentation mode |
| `wgpu_features` | empty | Required wgpu features |
| `wgpu_limits` | `None` (downlevel defaults) | Required wgpu limits |
| `init_logging` | `true` | Whether to initialize the tracing logger |
| `sample_count` | 1 | MSAA sample count |
| `canvas` | `None` | CSS selector for existing canvas (WASM only) |

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
| `create_depth_texture()` | Create depth texture + view |
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
    &device, format, scene.layout(),
    &mesh.positions, &mesh.normals, &mesh.indices,
    &resolved_shader, config,
);
```

## Compute Shaders

Encode GPU compute passes in `encode()`, before render passes.

### Setup

```rust
use mikage::{
    create_compute_pipeline, create_storage_buffer_init,
    storage_buffer_entry, uniform_buffer_entry, UniformBuffer,
};

// In the factory closure / constructor:

// 1. Create buffers
let data_buffer = create_storage_buffer_init(&device, "my_data", &initial_data);
let params = UniformBuffer::new(&device, "params", &my_params);

// 2. Create bind group layout + pipeline (one call)
let cs = wgpu::ShaderStages::COMPUTE;
let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    label: Some("compute_bgl"),
    entries: &[
        storage_buffer_entry(0, cs, false),  // read-write storage
        uniform_buffer_entry(1, cs),         // uniform params
    ],
});

let pipeline = create_compute_pipeline(
    &device,
    "my_compute",
    include_str!("shaders/compute.wgsl"),
    &[&bgl],
    "main",
);

// 3. Create bind group (raw wgpu — binding indices are shader contracts)
let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: Some("compute_bg"),
    layout: &bgl,
    entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: data_buffer.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: params.buffer().as_entire_binding() },
    ],
});
```

### Dispatch

```rust
fn encode(&mut self, ctx: &mut FrameContext) {
    // Update params
    self.params_buffer.write(&ctx.gpu.queue, &self.params);

    // Compute pass
    let mut cpass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("my_compute_pass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(&self.pipeline);
    cpass.set_bind_group(0, &self.bind_group, &[]);
    cpass.dispatch_workgroups(self.num_elements.div_ceil(64), 1, 1);
    drop(cpass);

    // Render pass
    let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        // ...
    });
    // ...
}
```

See `examples/boids.rs` for a full example with double-buffered storage and compute-to-instance rendering.

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

## Camera

The camera system is split into two traits:

- **`Camera`** — Read-only interface: `view_matrix()`, `projection_matrix()`, `position()`. Exposed in `FrameContext`.
- **`InteractiveCamera`** — Extends `Camera` with input handling (`on_mouse_drag`, `on_scroll`, `update`). Exposed in `UpdateContext`.

Built-in implementations:

| Camera | Use case | Controls |
|--------|----------|----------|
| `OrbitCamera` | 3D | Left drag: orbit, Right drag: pan, Scroll: zoom |
| `Camera2d` | 2D | Left drag: pan, Scroll: zoom |

Implement `InteractiveCamera` for a custom camera and pass it via `RunConfig::camera`.

## WASM

```bash
trunk build                   # build
trunk serve --open=false      # local dev server
```


Uses WebGPU backend. GPU initialization is asynchronous on WASM.

## Examples

```bash
cargo run -p mikage --example clear              # Color-cycling clear screen
cargo run -p mikage --example egui_demo          # egui window demo
cargo run -p mikage --example orbit_camera       # IcoSphere + orbit camera + egui
cargo run -p mikage --example instancing_2d      # 2D hex grid (Camera2d + pan/zoom)
cargo run -p mikage --example instancing_3d      # 3D sphere grid with wave animation
cargo run -p mikage --example custom_instance    # Custom InstanceVertex with 2D rotation
cargo run -p mikage --example boids              # GPU compute flocking (10k boids)
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| `wgpu` | GPU rendering and compute |
| `winit` | Window and event loop |
| `egui` + `egui-winit` + `egui-wgpu` | Immediate-mode UI |
| `glam` | Vector and matrix math |
| `bytemuck` | Safe GPU buffer casting |
| `tracing` | Logging |
