# mikage

A lightweight application framework built on wgpu + winit.

Provides GPU rendering, compute shaders, and egui UI integration for both Native and WASM (WebGPU) targets. Unlike a full engine, mikage doesn't abstract over wgpu — you create your own pipelines, buffers, and bind groups directly.

## Features

- **Raw wgpu access** — The framework manages the window and surface; you manage everything else.
- **egui integration** — Build UI in `gui()`. Input lock between egui and camera is automatic.
- **Camera system** — `Camera` trait (read-only) + `CameraController` trait (input handling). Built-in `OrbitCamera` and `Camera2d`.
- **Compute shaders** — Encode compute passes in `compute()`, which runs before rendering.
- **SolidRenderer** — Opaque (lit) and transparent (unlit) pipelines for solid-colored meshes.
- **InstanceRenderer** — GPU-instanced rendering with generic per-instance vertex data via `InstanceVertex` trait.
- **ShaderProcessor** — Lightweight WGSL preprocessor with `#import` resolution, recursive dependency handling, and circular import detection.
- **Multi-platform** — Native (Metal / Vulkan / DX12) and WASM (WebGPU).
- **tracing** — `RUST_LOG` on native, `tracing-web` (browser console) on WASM.

## Quick Start

```rust
use mikage::{App, GpuContext, RenderContext, RunConfig, UpdateContext};
use winit::dpi::PhysicalSize;

struct MyApp;

impl App for MyApp {
    fn init(&mut self, _ctx: &GpuContext, _size: PhysicalSize<u32>) {}
    fn update(&mut self, _ctx: &mut UpdateContext) {}

    fn render(&mut self, ctx: &mut RenderContext) {
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

    fn resize(&mut self, _ctx: &GpuContext, _new_size: PhysicalSize<u32>) {}
}

fn main() {
    mikage::run(MyApp, RunConfig::default());
}
```

## App Trait

| Method | When called | Required |
|--------|------------|----------|
| `init()` | Once, after GPU init | Yes |
| `update()` | Every frame (logic) | Yes |
| `compute()` | Every frame (before render) | No |
| `render()` | Every frame (drawing) | Yes |
| `gui()` | Every frame (egui UI) | No |
| `resize()` | On window resize | Yes |
| `on_window_event()` | On unhandled window events (file drops, focus, etc.) | No |

## Frame Loop

```
winit events → InputState → egui input
→ Camera update → App::update() → App::compute()
→ App::render() → App::gui() → egui render
→ submit → present
```

## Context Types

### `UpdateContext`
- `dt: f32` — seconds since last frame
- `elapsed: f64` — total elapsed time
- `gpu: &GpuContext` — device / queue
- `input: &InputState` — keyboard / mouse state
- `camera: &mut dyn CameraController` — mutable camera controller

### `ComputeContext`
- `device`, `queue`, `encoder` — encode compute passes

### `RenderContext`
- `device`, `queue`, `encoder` — encode render passes
- `surface_view` — render target
- `camera: &dyn Camera` — read-only camera

## RunConfig

| Field | Default | Purpose |
|-------|---------|---------|
| `title` | `"mikage"` | Window title |
| `width` / `height` | 1280 / 720 | Initial window size |
| `camera` | `OrbitCamera` | Camera controller (`Box<dyn CameraController>`) |
| `present_mode` | `AutoVsync` | wgpu presentation mode |
| `wgpu_features` | empty | Required wgpu features |
| `wgpu_limits` | `None` (downlevel defaults) | Required wgpu limits |
| `init_logging` | `true` | Whether to initialize the tracing logger |

## Helpers

| Name | Purpose |
|------|---------|
| `SceneBinding` | Bundles SceneUniform buffer + bind group layout + bind group |
| `SceneUniform` | View-projection + camera position + lighting uniform struct |
| `create_depth_texture()` | Create depth texture + view |
| `DEPTH_FORMAT` | `Depth32Float` constant |
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

- **`Camera`** — Read-only interface: `view_matrix()`, `projection_matrix()`, `position()`. Exposed in `RenderContext`.
- **`CameraController`** — Extends `Camera` with input handling (`on_mouse_drag`, `on_scroll`, `update`). Exposed in `UpdateContext`.

Built-in implementations:

| Camera | Use case | Controls |
|--------|----------|----------|
| `OrbitCamera` | 3D | Left drag: orbit, Right drag: pan, Scroll: zoom |
| `Camera2d` | 2D | Left drag: pan, Scroll: zoom |

Implement `CameraController` for a custom camera and pass it via `RunConfig::camera`.

## WASM

```bash
trunk build                   # build
trunk serve --open=false      # local dev server
```

Required headers (set in `Trunk.toml`):
```toml
[serve]
headers = { Cross-Origin-Opener-Policy = "same-origin", Cross-Origin-Embedder-Policy = "require-corp" }
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
