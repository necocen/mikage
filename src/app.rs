use winit::dpi::PhysicalSize;

use crate::camera::{Camera, InteractiveCamera};
use crate::context::GpuContext;
use crate::input::InputState;

/// Context for encoding GPU command passes (compute and render).
///
/// Passed to [`App::encode`]. Encode compute passes and/or render passes on `encoder`.
/// Draw to `surface_view` for rendering. Access device/queue via `gpu.device` / `gpu.queue`.
pub struct FrameContext<'a, C: Camera> {
    /// GPU context. Access `gpu.device` and `gpu.queue`.
    pub gpu: &'a GpuContext,
    /// Command encoder. Record compute and render passes here.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// The surface texture view for this frame. Use as the color attachment target.
    pub surface_view: &'a wgpu::TextureView,
    /// Current window size in pixels.
    pub window_size: PhysicalSize<u32>,
    /// The active camera (read-only). Use for view/projection matrices.
    pub camera: &'a C,
}

/// Context for per-frame logic updates.
///
/// Passed to [`App::update`]. Use for physics, input handling, and GPU data uploads.
pub struct UpdateContext<'a, C: InteractiveCamera> {
    /// Seconds elapsed since the previous frame.
    pub dt: f32,
    /// Total seconds elapsed since app start.
    pub elapsed: f64,
    /// Current window size in pixels.
    pub window_size: PhysicalSize<u32>,
    /// GPU context. Use `gpu.queue.write_buffer()` for data uploads.
    pub gpu: &'a GpuContext,
    /// Input state for the current frame.
    pub input: &'a InputState,
    /// The active camera controller (mutable). Provides both view matrices and input control.
    pub camera: &'a mut C,
}

/// The core application trait.
///
/// Implement this trait to build an application on mikage.
/// Create your app in a factory closure and pass it to [`run`](crate::run).
///
/// The associated type [`Camera`](App::Camera) specifies the camera controller type.
///
/// # Lifecycle
///
/// ```text
/// factory closure → [update() → encode() → gui()]* → resize() (on resize)
/// ```
///
/// - [`update`](App::update): Called every frame. Run simulation, process input, upload data.
/// - [`encode`](App::encode): Called every frame. Encode GPU compute and render passes.
/// - [`gui`](App::gui): Called every frame. Build egui UI.
/// - [`resize`](App::resize): Called on window resize. Recreate size-dependent resources.
pub trait App: 'static {
    /// The camera controller type used by this application.
    type Camera: InteractiveCamera;

    /// Called every frame for logic updates.
    ///
    /// Use this for simulation, input handling, and uploading data to the GPU
    /// via `ctx.gpu.queue.write_buffer()`. The camera is available as a mutable
    /// reference through `ctx.camera`.
    fn update(&mut self, ctx: &mut UpdateContext<Self::Camera>);

    /// Called every frame to encode GPU command passes.
    ///
    /// Encode compute passes and render passes here. Draw to `ctx.surface_view`.
    /// Access device/queue via `ctx.gpu.device` / `ctx.gpu.queue`.
    /// The framework renders egui on top after [`gui`](App::gui), so you don't
    /// need to handle egui rendering here.
    fn encode(&mut self, ctx: &mut FrameContext<Self::Camera>);

    /// Called every frame to build egui UI.
    ///
    /// Use `egui::Window`, `egui::SidePanel`, etc. to create UI elements.
    /// The default does nothing (no UI).
    fn gui(&mut self, _egui_ctx: &egui::Context) {}

    /// Called when the window is resized.
    ///
    /// Recreate size-dependent resources here (e.g., depth textures).
    /// The default does nothing.
    fn resize(&mut self, _ctx: &GpuContext, _new_size: PhysicalSize<u32>) {}

    /// Called for window events not handled internally by the framework.
    ///
    /// Use this to handle file drops, focus changes, scale factor changes,
    /// keyboard shortcuts, etc. The default does nothing.
    fn on_window_event(&mut self, _event: &winit::event::WindowEvent) {}
}
