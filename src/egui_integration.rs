use winit::event::WindowEvent;
use winit::window::Window;

use crate::context::GpuContext;

pub struct EguiIntegration {
    ctx: egui::Context,
    state: egui_winit::State,
    pub(crate) renderer: egui_wgpu::Renderer,
    pub(crate) screen_descriptor: egui_wgpu::ScreenDescriptor,
}

/// egui の prepare 結果。render pass に描画するために使う。
pub(crate) struct EguiPreparedFrame {
    pub(crate) tris: Vec<egui::ClippedPrimitive>,
    textures_to_free: Vec<egui::TextureId>,
}

impl EguiPreparedFrame {
    pub(crate) fn finish(self, renderer: &mut egui_wgpu::Renderer) {
        for id in &self.textures_to_free {
            renderer.free_texture(id);
        }
    }
}

impl EguiIntegration {
    pub fn new(window: &Window, gpu: &GpuContext) -> Self {
        let ctx = egui::Context::default();
        let viewport_id = ctx.viewport_id();
        let state = egui_winit::State::new(ctx.clone(), viewport_id, window, None, None, None);
        let renderer = egui_wgpu::Renderer::new(
            &gpu.device,
            gpu.render_format(),
            egui_wgpu::RendererOptions::default(),
        );

        let size = gpu.window_size();
        let pixels_per_point = Self::compute_pixels_per_point(window);
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [size.width, size.height],
            pixels_per_point,
        };

        Self {
            ctx,
            state,
            renderer,
            screen_descriptor,
        }
    }

    /// winit イベントを egui に転送。egui が消費したら true を返す。
    pub fn handle_window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        let response = self.state.on_window_event(window, event);
        response.consumed
    }

    /// egui がポインタ入力を要求しているか。
    pub fn wants_pointer_input(&self) -> bool {
        self.ctx.wants_pointer_input()
    }

    /// egui がキーボード入力を要求しているか。
    pub fn wants_keyboard_input(&self) -> bool {
        self.ctx.wants_keyboard_input()
    }

    /// egui がなんらかの入力を要求しているか。
    pub fn wants_any_input(&self) -> bool {
        self.wants_pointer_input() || self.wants_keyboard_input()
    }

    /// リサイズ時にスクリーン情報を更新。
    pub fn resize(&mut self, width: u32, height: u32, pixels_per_point: f32) {
        self.screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point,
        };
    }

    pub(crate) fn compute_pixels_per_point(window: &Window) -> f32 {
        window.scale_factor() as f32
    }

    /// egui の UI 構築・描画を一括で行う。
    ///
    /// prepare (UI 構築 + GPU バッファ更新) → render pass 作成 → 描画 → テクスチャ解放
    /// をすべてこのメソッド内で完結させる。
    pub(crate) fn render(
        &mut self,
        window: &Window,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        surface_view: &wgpu::TextureView,
        gui_fn: impl FnMut(&egui::Context),
    ) {
        let prepared = self.prepare(window, gpu, encoder, gui_fn);

        {
            let mut render_pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui_render_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: surface_view,
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
                })
                .forget_lifetime();
            self.renderer
                .render(&mut render_pass, &prepared.tris, &self.screen_descriptor);
        }

        prepared.finish(&mut self.renderer);
    }

    /// egui の UI 構築を実行し、GPU バッファを更新する。
    fn prepare(
        &mut self,
        window: &Window,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        gui_fn: impl FnMut(&egui::Context),
    ) -> EguiPreparedFrame {
        // 毎フレーム screen_descriptor をサーフェスの実サイズで更新
        let surface_size = gpu.window_size();
        self.screen_descriptor.size_in_pixels = [surface_size.width, surface_size.height];

        let raw_input = self.state.take_egui_input(window);
        let full_output = self.ctx.run(raw_input, gui_fn);

        self.state
            .handle_platform_output(window, full_output.platform_output);

        let tris = self
            .ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(&gpu.device, &gpu.queue, *id, image_delta);
        }

        self.renderer.update_buffers(
            &gpu.device,
            &gpu.queue,
            encoder,
            &tris,
            &self.screen_descriptor,
        );

        EguiPreparedFrame {
            tris,
            textures_to_free: full_output.textures_delta.free,
        }
    }
}
