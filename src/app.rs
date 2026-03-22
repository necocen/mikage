use winit::dpi::PhysicalSize;

use crate::camera::{Camera, CameraController};
use crate::context::GpuContext;
use crate::input::InputState;

/// Context for encoding render passes.
///
/// Passed to [`App::render`]. Create a render pass on `encoder` and
/// draw to `surface_view`.
pub struct RenderContext<'a> {
    /// The wgpu device.
    pub device: &'a wgpu::Device,
    /// The wgpu queue. Useful for `queue.write_buffer()` if needed.
    pub queue: &'a wgpu::Queue,
    /// Command encoder. Record render passes here.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// The surface texture view for this frame. Use as the color attachment target.
    pub surface_view: &'a wgpu::TextureView,
    /// Texture format for render pipelines. Same as [`GpuContext::render_format()`](crate::GpuContext::render_format).
    pub render_format: wgpu::TextureFormat,
    /// Current window size in pixels.
    pub window_size: PhysicalSize<u32>,
    /// The active camera (read-only). Use for view/projection matrices.
    pub camera: &'a dyn Camera,
}

/// Context for encoding compute passes.
///
/// Passed to [`App::compute`]. Create compute passes on `encoder`.
pub struct ComputeContext<'a> {
    /// The wgpu device.
    pub device: &'a wgpu::Device,
    /// The wgpu queue.
    pub queue: &'a wgpu::Queue,
    /// Command encoder. Record compute passes here.
    pub encoder: &'a mut wgpu::CommandEncoder,
}

/// Context for per-frame logic updates.
///
/// Passed to [`App::update`]. Use for physics, input handling, and GPU data uploads.
pub struct UpdateContext<'a> {
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
    pub camera: &'a mut dyn CameraController,
}

/// The core application trait.
///
/// Implement this trait to build an application on mikage.
/// Pass your type to [`run`](crate::run) to start the event loop.
///
/// # Lifecycle
///
/// ```text
/// init() → [update() → compute() → render() → gui()]* → resize() (on resize)
/// ```
///
/// - [`init`](App::init): Called once after GPU initialization. Create pipelines, buffers, textures.
/// - [`update`](App::update): Called every frame. Run simulation, process input, upload data.
/// - [`compute`](App::compute): Called every frame before rendering. Encode GPU compute passes.
/// - [`render`](App::render): Called every frame. Encode GPU render passes.
/// - [`gui`](App::gui): Called every frame. Build egui UI.
/// - [`resize`](App::resize): Called on window resize. Recreate size-dependent resources.
pub trait App: 'static {
    /// Called once after GPU initialization.
    ///
    /// Create pipelines, buffers, textures, and other GPU resources here.
    /// `size` is the initial window size in pixels.
    fn init(&mut self, ctx: &GpuContext, size: PhysicalSize<u32>);

    /// Called every frame for logic updates.
    ///
    /// Use this for simulation, input handling, and uploading data to the GPU
    /// via `ctx.gpu.queue.write_buffer()`. The camera is available as a mutable
    /// reference through `ctx.camera`.
    fn update(&mut self, ctx: &mut UpdateContext);

    /// Called every frame to encode compute passes, before [`render`](App::render).
    ///
    /// Override this if you use GPU compute shaders. The default does nothing.
    fn compute(&mut self, _ctx: &mut ComputeContext) {}

    /// Called every frame to encode render passes.
    ///
    /// Draw to `ctx.surface_view`. The framework renders egui on top after
    /// [`gui`](App::gui), so you don't need to handle egui rendering here.
    fn render(&mut self, ctx: &mut RenderContext);

    /// Called every frame to build egui UI.
    ///
    /// Use `egui::Window`, `egui::SidePanel`, etc. to create UI elements.
    /// The default does nothing (no UI).
    fn gui(&mut self, _egui_ctx: &egui::Context) {}

    /// Called when the window is resized.
    ///
    /// Recreate size-dependent resources here (e.g., depth textures).
    fn resize(&mut self, ctx: &GpuContext, new_size: PhysicalSize<u32>);

    /// Called for window events not handled internally by the framework.
    ///
    /// Use this to handle file drops, focus changes, scale factor changes,
    /// keyboard shortcuts, etc. The default does nothing.
    fn on_window_event(&mut self, _event: &winit::event::WindowEvent) {}
}
