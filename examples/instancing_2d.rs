use mikage::{
    App, Camera, Camera2d, FrameContext, GpuContext, InstanceData, InstanceRenderer,
    InstanceRendererConfig, RegularPolygonMesh, RunConfig, SceneBinding, UpdateContext,
};
use winit::dpi::PhysicalSize;

struct Instancing2dApp {
    renderer: InstanceRenderer,
    scene: SceneBinding,
}

impl Instancing2dApp {
    fn new(ctx: &GpuContext, _size: PhysicalSize<u32>) -> Self {
        let scene = SceneBinding::new(&ctx.device);

        let hex = RegularPolygonMesh::generate(6);
        let renderer = InstanceRenderer::new(
            ctx,
            scene.layout(),
            &hex.positions,
            &hex.normals,
            &hex.indices,
            InstanceRendererConfig::default_2d(),
        );

        Self { renderer, scene }
    }
}

impl App for Instancing2dApp {
    type Camera = Camera2d;

    fn update(&mut self, ctx: &mut UpdateContext<Camera2d>) {
        let aspect = ctx.window_size.width as f32 / ctx.window_size.height.max(1) as f32;
        self.scene
            .update_from_camera(&ctx.gpu.queue, &*ctx.camera, aspect);

        let vp = ctx.camera.view_projection_matrix(aspect);

        // Compute world-space viewport bounds from inverse VP matrix
        let inv_vp = vp.inverse();
        let ndc_min = inv_vp.project_point3(glam::Vec3::new(-1.0, -1.0, 0.0));
        let ndc_max = inv_vp.project_point3(glam::Vec3::new(1.0, 1.0, 0.0));
        let view_min = glam::Vec2::new(ndc_min.x, ndc_min.y);
        let view_max = glam::Vec2::new(ndc_max.x, ndc_max.y);

        // Generate a hex grid of instances covering the viewport
        let hex_radius = 0.12;
        let spacing = hex_radius * 1.05;
        let dx = spacing * 1.5;
        let dy = spacing * 3.0_f32.sqrt();

        let margin = hex_radius * 2.0;
        let col_min = ((view_min.x - margin) / dx).floor() as i32;
        let col_max = ((view_max.x + margin) / dx).ceil() as i32;
        let row_min = ((view_min.y - margin) / dy).floor() as i32;
        let row_max = ((view_max.y + margin) / dy).ceil() as i32;

        let mut instances = Vec::new();
        for col in col_min..=col_max {
            for row in row_min..=row_max {
                let x = col as f32 * dx;
                let y = row as f32 * dy + if col % 2 != 0 { dy * 0.5 } else { 0.0 };

                let r = ((col as f32 * 0.1).sin() * 0.5 + 0.5).clamp(0.2, 1.0);
                let g = ((row as f32 * 0.15).cos() * 0.5 + 0.5).clamp(0.2, 1.0);
                let b = (((col + row) as f32 * 0.08).sin() * 0.5 + 0.5).clamp(0.2, 1.0);

                instances.push(InstanceData {
                    pos_scale: [x, y, 0.0, hex_radius],
                    color: [r, g, b, 1.0],
                });
            }
        }

        self.renderer
            .update_instances(ctx.gpu, &instances);
    }

    fn encode(&mut self, ctx: &mut FrameContext<Camera2d>) {
        let color_attachment = ctx.color_attachment(wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color {
                r: 0.1,
                g: 0.1,
                b: 0.15,
                a: 1.0,
            }),
            store: wgpu::StoreOp::Store,
        });
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("instancing_2d_pass"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_bind_group(0, self.scene.bind_group(), &[]);
        self.renderer.render(&mut pass);
    }

    fn gui(&mut self, egui_ctx: &mikage::egui::Context) {
        mikage::egui::Window::new("Info").show(egui_ctx, |ui| {
            ui.label("2D Instancing Demo");
            ui.label("Left drag: pan");
            ui.label("Scroll: zoom");
        });
    }
}

fn main() {
    let mut camera = Camera2d::default();
    camera.zoom = 2.0;
    camera.damping = 0.85;

    mikage::run(
        Instancing2dApp::new,
        RunConfig {
            title: "mikage - 2D instancing".to_string(),
            ..RunConfig::with_defaults(camera)
        },
    );
}
