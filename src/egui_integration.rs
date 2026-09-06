//! Window GUI adapter. The no-GUI implementation keeps the window driver identical.

#[cfg(feature = "window-gui")]
mod enabled {
    use crate::{App, GpuContext, RenderTargetConfig};
    use dpi::PhysicalSize;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    #[cfg(not(target_family = "wasm"))]
    use std::time::Instant;
    #[cfg(target_family = "wasm")]
    use web_time::Instant;
    use winit::{event::WindowEvent, window::Window};

    pub(crate) struct EguiIntegration {
        ctx: egui::Context,
        state: egui_winit::State,
        renderer: egui_wgpu::Renderer,
        repaint_after: Duration,
        repaint_request: Arc<Mutex<Option<Instant>>>,
    }

    pub(crate) struct PreparedFrame {
        output: egui::FullOutput,
        tris: Vec<egui::ClippedPrimitive>,
    }

    impl Drop for PreparedFrame {
        fn drop(&mut self) {
            self.output.textures_delta.clear();
        }
    }

    impl EguiIntegration {
        pub fn new(
            window: &Window,
            gpu: &GpuContext,
            target: RenderTargetConfig,
            wake: Arc<dyn Fn() + Send + Sync>,
        ) -> Self {
            let ctx = egui::Context::default();
            let repaint_request = Arc::new(Mutex::new(None::<Instant>));
            let repaint_signal = repaint_request.clone();
            ctx.set_request_repaint_callback(move |request| {
                if let Some(deadline) = Instant::now().checked_add(request.delay) {
                    let mut pending = repaint_signal.lock().unwrap();
                    *pending = Some(pending.map_or(deadline, |old| old.min(deadline)));
                    drop(pending);
                    wake();
                }
            });
            let state =
                egui_winit::State::new(ctx.clone(), ctx.viewport_id(), window, None, None, None);
            let renderer = egui_wgpu::Renderer::new(
                &gpu.device,
                target.color_format,
                egui_wgpu::RendererOptions::default(),
            );
            Self {
                ctx,
                state,
                renderer,
                repaint_after: Duration::MAX,
                repaint_request,
            }
        }

        pub fn handle_window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
            self.state.on_window_event(window, event).consumed
        }

        pub fn wants_pointer_input(&self) -> bool {
            self.ctx.egui_wants_pointer_input()
        }
        pub fn repaint_after(&self) -> Duration {
            self.repaint_after
        }
        pub fn take_repaint_request(&self) -> Option<Instant> {
            self.repaint_request.lock().unwrap().take()
        }

        pub fn build<A: App>(&mut self, window: &Window, app: &mut A) -> PreparedFrame {
            let input = self.state.take_egui_input(window);
            let mut output = self.ctx.run_ui(input, |ui| app.gui(ui));
            self.repaint_after = output
                .viewport_output
                .get(&self.ctx.viewport_id())
                .map_or(Duration::MAX, |o| o.repaint_delay);
            self.state
                .handle_platform_output(window, std::mem::take(&mut output.platform_output));
            let tris = self
                .ctx
                .tessellate(std::mem::take(&mut output.shapes), output.pixels_per_point);
            PreparedFrame { output, tris }
        }

        pub fn encode(
            &mut self,
            gpu: &GpuContext,
            encoder: &mut wgpu::CommandEncoder,
            view: &wgpu::TextureView,
            size: PhysicalSize<u32>,
            prepared: &PreparedFrame,
        ) -> Vec<wgpu::CommandBuffer> {
            for (id, deltas) in &prepared.output.textures_delta.set {
                for delta in deltas {
                    self.renderer
                        .update_texture(&gpu.device, &gpu.queue, *id, delta);
                }
            }
            let screen = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [size.width, size.height],
                pixels_per_point: prepared.output.pixels_per_point,
            };
            let extra = self.renderer.update_buffers(
                &gpu.device,
                &gpu.queue,
                encoder,
                &prepared.tris,
                &screen,
            );
            {
                let mut pass = encoder
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("mikage_gui"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    })
                    .forget_lifetime();
                self.renderer.render(&mut pass, &prepared.tris, &screen);
            }
            extra
        }

        // Submitted command buffers may still reference textures scheduled for deletion.
        // wgpu retains those resources; destroy is legal only after encoding/submission.
        pub fn finish(&mut self, mut prepared: PreparedFrame) {
            for id in &prepared.output.textures_delta.free {
                self.renderer.free_texture(id);
            }
            prepared.output.textures_delta.clear();
        }
    }
}

#[cfg(not(feature = "window-gui"))]
mod disabled {
    use crate::{App, GpuContext, RenderTargetConfig};
    use dpi::PhysicalSize;
    use std::time::Duration;
    use winit::{event::WindowEvent, window::Window};
    pub(crate) struct EguiIntegration;
    pub(crate) struct PreparedFrame;
    impl EguiIntegration {
        pub fn new(
            _: &Window,
            _: &GpuContext,
            _: RenderTargetConfig,
            _: std::sync::Arc<dyn Fn() + Send + Sync>,
        ) -> Self {
            Self
        }
        pub fn handle_window_event(&mut self, _: &Window, _: &WindowEvent) -> bool {
            false
        }
        pub fn wants_pointer_input(&self) -> bool {
            false
        }
        pub fn repaint_after(&self) -> Duration {
            Duration::MAX
        }
        #[cfg(not(target_family = "wasm"))]
        pub fn take_repaint_request(&self) -> Option<std::time::Instant> {
            None
        }
        #[cfg(target_family = "wasm")]
        pub fn take_repaint_request(&self) -> Option<web_time::Instant> {
            None
        }
        pub fn build<A: App>(&mut self, _: &Window, _: &mut A) -> PreparedFrame {
            PreparedFrame
        }
        pub fn encode(
            &mut self,
            _: &GpuContext,
            _: &mut wgpu::CommandEncoder,
            _: &wgpu::TextureView,
            _: PhysicalSize<u32>,
            _: &PreparedFrame,
        ) -> Vec<wgpu::CommandBuffer> {
            Vec::new()
        }
        pub fn finish(&mut self, _: PreparedFrame) {}
    }
}
#[cfg(not(feature = "window-gui"))]
pub(crate) use disabled::*;
#[cfg(feature = "window-gui")]
pub(crate) use enabled::*;
