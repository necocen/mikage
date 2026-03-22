use mikage::{App, GpuContext, RenderContext, RunConfig, UpdateContext};
use winit::dpi::PhysicalSize;

struct ClearApp {
    time: f64,
}

impl App for ClearApp {
    fn init(&mut self, _ctx: &GpuContext, _size: PhysicalSize<u32>) {
        tracing::info!("ClearApp initialized");
    }

    fn update(&mut self, ctx: &mut UpdateContext) {
        self.time = ctx.elapsed;
    }

    fn render(&mut self, ctx: &mut RenderContext) {
        // Cycle through colors over time
        let t = self.time as f32;
        let r = (t * 0.3).sin() * 0.5 + 0.5;
        let g = (t * 0.5).sin() * 0.5 + 0.5;
        let b = (t * 0.7).sin() * 0.5 + 0.5;

        let _pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.surface_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: r as f64,
                        g: g as f64,
                        b: b as f64,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
    }

    fn resize(&mut self, _ctx: &GpuContext, new_size: PhysicalSize<u32>) {
        tracing::info!("Resized to {}x{}", new_size.width, new_size.height);
    }
}

fn main() {
    mikage::run(
        ClearApp { time: 0.0 },
        RunConfig {
            title: "mikage - clear example".to_string(),
            ..Default::default()
        },
    );
}
