# Changelog

## v0.3.0

### Added

- **MSAA support** — Enable 4x MSAA via `RunConfig::sample_count`. Query the current setting with `GpuContext::sample_count()` / `msaa_view()`. Invalid values panic at startup (only 1 or 4 are accepted).
- **`RenderTargetConfig`** — `GpuContext::render_target_config()` returns color format, depth format, and sample count in one struct. Helper methods `color_target_state()` / `multisample_state()` / `depth_stencil_state()` reduce pipeline creation boilerplate.
- **`FrameContext::color_attachment()`** — Builds a color attachment that automatically handles MSAA resolve.
- **`InstanceRenderer::prepare_compute()`** — Single-call API for compute shader integration. Handles buffer allocation, instance count, and reallocation detection. Returns `ComputeBufferState`.
- **`InteractiveCamera` gesture hooks** — Added `on_touch_drag()` / `on_touch_drag_end()` / `on_pinch_pan()` as trait methods with default implementations that delegate to mouse/scroll methods. Custom cameras can override per-gesture behavior.
- **`InteractiveCamera::set_viewport_size()` / `set_cursor_position()`** — Trait methods to notify the camera of window size and cursor position.
- **High-level renderer API** — `SolidRenderer::add_object()` / `update_object()` and `InstanceRenderer::update_instances()` now accept `&GpuContext`, eliminating the need to pass raw wgpu types. Previous signatures are preserved as `*_raw()` (`#[doc(hidden)]`, not part of the stable API).

### Changed

- **`FrameContext::surface_view` is now private** — Replace direct `ctx.surface_view` access with `ctx.color_attachment(ops)`. Works correctly with or without MSAA.
- **`InteractiveCamera::enabled` is now private** — Access via `set_enabled()` / `is_enabled()` trait methods. `enabled` only suppresses input events; inertial motion in `update()` continues.
- **Camera2d panning accuracy** — Panning now uses pixel-to-world conversion. The hardcoded `pan_speed` field has been removed. Zooming preserves the world coordinate under the cursor (zoom-to-cursor).
- **Frame-rate independent damping** — Inertial damping in `OrbitCamera` / `Camera2d` now uses an exact geometric series, eliminating frame-rate dependent behavior.
- **Drag lifecycle fixes** — Button state gating, `on_drag_end` on all-buttons-released, and correct drag termination on egui capture transitions.
- **Improved egui event filtering** — Keyboard and pointer events are filtered by category, preventing stuck input during egui interaction.
- **Removed `Default` from `SolidObjectId`** — Eliminates the panic trap from invalid default IDs. `update_object()` now panics with an explicit message on invalid IDs.
- **Renamed `InputState::begin_frame` to `end_frame`** — Changed to `pub(crate)` (internal API).
- **SurfaceError recovery** — All error arms in `render_frame()` now call `request_redraw()`, preventing the render loop from silently stopping on WASM.
- **`SceneUniform::with_light` NaN prevention** — A zero-length `light_dir` now falls back to `Vec3::Y` instead of producing NaN.

### Documentation

- Documented `SolidRenderer` transparency policy (automatic pipeline switch based on alpha, no depth sorting for transparent objects).
- Added "pub for integration testing only, not part of the stable API" notes to all `#[doc(hidden)]` methods.
- Added SAFETY comment and TODO to the `unsafe mem::zeroed()` KeyEvent test helper.
- Documented gesture default assumptions in `InteractiveCamera`.

### Breaking changes migration guide

```rust
// FrameContext: surface_view -> color_attachment()
// Before (v0.2):
let attachment = wgpu::RenderPassColorAttachment {
    view: ctx.surface_view,
    resolve_target: None,
    ops: ...,
};
// After (v0.3):
let attachment = ctx.color_attachment(ops);

// InteractiveCamera: enabled field -> methods
// Before (v0.2):
camera.enabled = false;
if camera.enabled { ... }
// After (v0.3):
camera.set_enabled(false);
if camera.is_enabled() { ... }

// SolidObjectId: Default removed
// Before (v0.2):
let id = SolidObjectId::default(); // compiles but panics on use
// After (v0.3):
let id = solid.add_object(&gpu, &positions, &normals, &indices);

// Camera2d: pan_speed removed
// Before (v0.2):
let cam = Camera2d { pan_speed: 0.01, ..Default::default() };
// After (v0.3):
let cam = Camera2d::default(); // pan speed is derived from pixel-to-world conversion

// InteractiveCamera: new methods (have default implementations)
// Custom implementations can override set_viewport_size() /
// set_cursor_position() as needed.
```
