use dpi::PhysicalSize;

use crate::camera::{Camera, InteractiveCamera};
use crate::context::{GpuContext, RenderTargetConfig};
use crate::runtime::SubmissionToken;

/// A borrowed render destination. `view` is multisampled when MSAA is enabled.
#[derive(Clone, Copy)]
pub struct RenderTarget<'a> {
    pub view: &'a wgpu::TextureView,
    pub resolve_target: Option<&'a wgpu::TextureView>,
    pub depth_view: Option<&'a wgpu::TextureView>,
    pub size: PhysicalSize<u32>,
    pub config: RenderTargetConfig,
}

impl<'a> RenderTarget<'a> {
    /// The single-sampled view used for overlays after scene rendering.
    pub fn resolved_view(&self) -> &'a wgpu::TextureView {
        self.resolve_target.unwrap_or(self.view)
    }
}

/// One simulation tick, independent of rendering, window input, and wall time.
///
/// The runtime submits this encoder immediately after the hook returns. Queue
/// uploads made here therefore precede this tick, including when several ticks
/// execute before one render. Do not submit this encoder yourself.
pub struct TickContext<'a> {
    pub gpu: &'a GpuContext,
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// One-based tick identity; identities never reset when an app resets itself.
    pub tick_id: u64,
    pub dt: f32,
    /// Simulation time at the end of this tick, in seconds.
    pub elapsed: f64,
}

/// View-dependent preparation; this hook never advances simulation.
pub struct RenderUpdateContext<'a, C: Camera> {
    pub gpu: &'a GpuContext,
    pub camera: &'a C,
    pub target_size: PhysicalSize<u32>,
    pub target_config: RenderTargetConfig,
    pub dt: f32,
    pub elapsed: f64,
}

/// Encodes a scene or an overlay into a supplied render destination.
pub struct RenderContext<'a, C: Camera> {
    pub gpu: &'a GpuContext,
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Auxiliary command buffers, submitted in insertion order before `encoder`.
    /// This supports egui's buffer preparation commands.
    pub extra_command_buffers: &'a mut Vec<wgpu::CommandBuffer>,
    pub target: RenderTarget<'a>,
    pub camera: &'a C,
    pub frame_id: u64,
    pub completed_tick: u64,
}

impl<'a, C: Camera> RenderContext<'a, C> {
    pub fn color_attachment(
        &self,
        ops: wgpu::Operations<wgpu::Color>,
    ) -> wgpu::RenderPassColorAttachment<'a> {
        wgpu::RenderPassColorAttachment {
            view: self.target.view,
            resolve_target: self.target.resolve_target,
            depth_slice: None,
            ops,
        }
    }
}

/// Encoder access for an ordered diagnostic command, without a simulation tick.
pub struct CommandContext<'a> {
    pub gpu: &'a GpuContext,
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Most recently submitted simulation tick, or zero before the first tick.
    pub tick_id: u64,
}

/// Host input, available only when the window integration is enabled.
///
/// Convert the input snapshot into app-owned controls and pending actions here.
/// Transient input is delivered once even when zero or several ticks follow.
#[cfg(feature = "window")]
pub struct WindowInputContext<'a, C: InteractiveCamera> {
    pub window: &'a winit::window::Window,
    pub input: &'a crate::input::InputState,
    pub camera: &'a mut C,
}

/// Portable application lifecycle, shared by windowed and headless execution.
///
/// Simulation belongs in `tick`; rendering, GUI, and input never implicitly
/// advance it. A host may execute zero or several ticks between renders.
pub trait App: 'static {
    type Camera: InteractiveCamera;

    fn tick(&mut self, _ctx: &mut TickContext<'_>) {}
    fn prepare_render(&mut self, _ctx: &mut RenderUpdateContext<'_, Self::Camera>) {}
    fn render(&mut self, _ctx: &mut RenderContext<'_, Self::Camera>) {}

    #[cfg(feature = "window")]
    fn on_input(&mut self, _ctx: &mut WindowInputContext<'_, Self::Camera>) {}

    #[cfg(feature = "window")]
    fn on_window_event(&mut self, _event: &winit::event::WindowEvent) {}

    #[cfg(feature = "gui")]
    fn gui(&mut self, _ui: &mut egui::Ui) {}

    fn resize(&mut self, _gpu: &GpuContext, _size: PhysicalSize<u32>) {}

    /// Runs on the thread driving the runtime, immediately after submission.
    fn after_submit(&mut self, _gpu: &GpuContext, _token: &SubmissionToken) {}

    /// Runs on the runtime thread after completion notifications are drained.
    fn after_complete(&mut self, _gpu: &GpuContext, _token: &SubmissionToken) {}

    /// Called once before the runtime releases its GPU and worker resources.
    fn shutdown(&mut self, _gpu: &GpuContext) {}

    /// Register app-owned textures or buffers available for diagnostic capture.
    fn capture_targets(&self, _registry: &mut crate::capture::CaptureRegistry) {}

    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    fn agent_status(&self) -> serde_json::Value {
        serde_json::Value::Null
    }

    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    fn on_agent_command(
        &mut self,
        _payload: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        Err("app.command is not implemented for this app".to_owned())
    }

    /// Encodes a deferred diagnostic using the runtime's submission ordering.
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    fn encode_agent_command(
        &mut self,
        _payload: serde_json::Value,
        _ctx: &mut CommandContext<'_>,
    ) -> Result<serde_json::Value, String> {
        Err("app GPU command is not implemented for this app".to_owned())
    }

    /// Finalizes a deferred response after its GPU endpoint completes and
    /// `after_complete` has run. Drain app-owned readback results here to replace
    /// the provisional JSON returned by `encode_agent_command`.
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    fn complete_agent_command(
        &mut self,
        _gpu: &GpuContext,
        response: serde_json::Value,
        _token: &SubmissionToken,
    ) -> Result<serde_json::Value, String> {
        Ok(response)
    }
}
