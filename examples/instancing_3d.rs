use mikage::dpi::PhysicalSize;
use mikage::{
    App, GpuContext, IcoSphereMesh, InstanceData, InstanceRenderer, InstanceRendererConfig,
    OrbitCamera, RenderContext, RenderTargetConfig, RenderUpdateContext, RunConfig, SceneBinding,
    TickContext, create_depth_texture,
};

struct Instancing3dApp {
    renderer: InstanceRenderer,
    scene: SceneBinding,
    depth_view: wgpu::TextureView,
    target_config: RenderTargetConfig,
    time: f64,
}

impl Instancing3dApp {
    fn new(ctx: &GpuContext, target: RenderTargetConfig, size: PhysicalSize<u32>) -> Self {
        let scene = SceneBinding::new(&ctx.device);

        let sphere = IcoSphereMesh::generate(1);
        let renderer = InstanceRenderer::new(
            ctx,
            target,
            scene.layout(),
            &sphere.positions,
            &sphere.normals,
            &sphere.indices,
            InstanceRendererConfig::default_3d(),
        );

        let (_, depth_view) = create_depth_texture(ctx, size, target);

        Self {
            renderer,
            scene,
            depth_view,
            target_config: target,
            time: 0.0,
        }
    }
}

impl App for Instancing3dApp {
    type Camera = OrbitCamera;

    fn tick(&mut self, ctx: &mut TickContext) {
        self.time = ctx.elapsed;
        // 3D grid of spheres
        let grid = 5;
        let spacing = 2.5;
        let t = self.time as f32;
        let mut instances = Vec::new();

        for x in -grid..=grid {
            for y in -grid..=grid {
                for z in -grid..=grid {
                    let px = x as f32 * spacing;
                    let py = y as f32 * spacing;
                    let pz = z as f32 * spacing;

                    // Animated scale
                    let dist = (px * px + py * py + pz * pz).sqrt();
                    let wave = ((dist * 0.3 - t * 2.0).sin() * 0.3 + 0.7).max(0.1);

                    // Color based on position
                    let r = (x as f32 / grid as f32 * 0.5 + 0.5).clamp(0.1, 1.0);
                    let g = (y as f32 / grid as f32 * 0.5 + 0.5).clamp(0.1, 1.0);
                    let b = (z as f32 / grid as f32 * 0.5 + 0.5).clamp(0.1, 1.0);

                    instances.push(InstanceData {
                        pos_scale: [px, py, pz, wave],
                        color: [r, g, b, 1.0],
                    });
                }
            }
        }

        self.renderer.update_instances(ctx.gpu, &instances);
    }

    fn prepare_render(&mut self, ctx: &mut RenderUpdateContext<OrbitCamera>) {
        let aspect = ctx.target_size.width as f32 / ctx.target_size.height.max(1) as f32;
        self.scene
            .update_from_camera(&ctx.gpu.queue, ctx.camera, aspect);
    }

    fn render(&mut self, ctx: &mut RenderContext<OrbitCamera>) {
        let color_attachment = ctx.color_attachment(wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color {
                r: 0.02,
                g: 0.02,
                b: 0.05,
                a: 1.0,
            }),
            store: wgpu::StoreOp::Store,
        });
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("instancing_3d_pass"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_bind_group(0, self.scene.bind_group(), &[]);
        self.renderer.render(&mut pass);
    }

    #[cfg(feature = "gui")]
    fn gui(&mut self, ui: &mut mikage::egui::Ui) {
        let egui_ctx = ui.ctx();
        mikage::egui::Window::new("Info").show(egui_ctx, |ui| {
            ui.label("3D Instancing Demo");
            ui.label("Left drag: orbit | Right drag: pan | Scroll: zoom");
            ui.label(format!("Instances: {}", (5 * 2 + 1_i32).pow(3)));
        });
    }

    fn resize(&mut self, ctx: &GpuContext, new_size: PhysicalSize<u32>) {
        let (_, depth_view) = create_depth_texture(ctx, new_size, self.target_config);
        self.depth_view = depth_view;
    }
}

fn main() {
    let mut camera = OrbitCamera::default();
    camera.distance = 25.0;
    camera.pitch = 0.5;
    camera.yaw = 0.8;
    camera.damping = 0.85;

    mikage::run(
        Instancing3dApp::new,
        RunConfig {
            title: "mikage - 3D instancing".to_string(),
            camera,
            ..Default::default()
        },
    )
    .expect("mikage application failed");
}
