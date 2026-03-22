# mikage

A lightweight application framework built on wgpu + winit.

Provides GPU rendering, compute shaders, and egui UI integration for both Native and WASM (WebGPU) targets. Unlike a full engine, mikage doesn't abstract over wgpu — you create your own pipelines, buffers, and bind groups directly.

## Features

- **Raw wgpu access** — The framework manages the window and surface; you manage everything else.
- **egui integration** — Build UI in `gui()`. Input lock between egui and camera is automatic.
- **Camera system** — `Camera` trait (read-only) + `CameraController` trait (input handling). Built-in `OrbitCamera` implements both.
- **Compute shaders** — Encode compute passes in `compute()`, which runs before rendering.
- **SolidRenderer** — Opaque (lit) and transparent (unlit) pipelines for solid-colored meshes.
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
| `create_depth_texture()` | Create depth texture + view |
| `DEPTH_FORMAT` | `Depth32Float` constant |
| `SceneUniform` | View-projection + camera position + lighting uniform struct |
| `IcoSphereMesh::generate(n)` | Generate icosphere mesh with n subdivisions |
| `CubeMesh::generate()` | Generate unit cube mesh with per-face normals |
| `PlaneMesh::generate()` | Generate unit plane mesh (XZ plane, +Y normal) |
| `SolidRenderer` | Opaque (lit) and transparent (unlit) mesh renderer using SceneUniform |

## Camera

The camera system is split into two traits:

- **`Camera`** — Read-only interface: `view_matrix()`, `projection_matrix()`, `position()`. Exposed in `RenderContext`.
- **`CameraController`** — Extends `Camera` with input handling (`on_mouse_drag`, `on_scroll`, `update`). Exposed in `UpdateContext`.

`OrbitCamera` is the built-in implementation:

| Input | Action |
|-------|--------|
| Left drag | Orbit (rotate) |
| Right drag | Pan (translate target) |
| Scroll | Zoom (adjust distance) |

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
cargo run -p mikage --example clear          # Color-cycling clear screen
cargo run -p mikage --example egui_demo      # egui window demo
cargo run -p mikage --example orbit_camera   # IcoSphere + orbit camera + egui
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
