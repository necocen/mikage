use mikage::{App, FrameContext, OrbitCamera, RunConfig, UpdateContext};

struct ClearApp {
    time: f64,
}

impl App for ClearApp {
    type Camera = OrbitCamera;

    fn update(&mut self, ctx: &mut UpdateContext<OrbitCamera>) {
        self.time = ctx.elapsed;
    }

    fn encode(&mut self, ctx: &mut FrameContext<OrbitCamera>) {
        // Cycle through colors over time
        let t = self.time as f32;
        let r = (t * 0.3).sin() * 0.5 + 0.5;
        let g = (t * 0.5).sin() * 0.5 + 0.5;
        let b = (t * 0.7).sin() * 0.5 + 0.5;

        let color_attachment = ctx.color_attachment(wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color {
                r: r as f64,
                g: g as f64,
                b: b as f64,
                a: 1.0,
            }),
            store: wgpu::StoreOp::Store,
        });
        let _pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear_pass"),
            color_attachments: &[Some(color_attachment)],
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
