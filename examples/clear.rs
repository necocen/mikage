use mikage::{App, FrameContext, RunConfig, UpdateContext};

struct ClearApp {
    time: f64,
}

impl App for ClearApp {
    fn update(&mut self, ctx: &mut UpdateContext) {
        self.time = ctx.elapsed;
    }

    fn encode(&mut self, ctx: &mut FrameContext) {
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
}

fn main() {
    mikage::run(
        |_ctx, _size| ClearApp { time: 0.0 },
        RunConfig {
            title: "mikage - clear example".to_string(),
            ..Default::default()
        },
    );
}
