use mikage::{
    App, DEPTH_FORMAT, GpuContext, IcoSphereMesh, InstanceData, InstanceRenderer,
    InstanceRendererConfig, OrbitCamera, RenderContext, RunConfig, SceneUniform, UpdateContext,
    create_depth_texture,
};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

struct Instancing3dApp {
    renderer: Option<InstanceRenderer>,
    scene_buffer: Option<wgpu::Buffer>,
    scene_bind_group: Option<wgpu::BindGroup>,
    depth_view: Option<wgpu::TextureView>,
    time: f64,
}

impl App for Instancing3dApp {
    fn init(&mut self, ctx: &GpuContext, size: PhysicalSize<u32>) {
        let device = &ctx.device;

        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let scene_uniform = SceneUniform::new(glam::Mat4::IDENTITY, glam::Vec3::ZERO);
        let scene_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene_buffer"),
            contents: bytemuck::bytes_of(&scene_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_bind_group"),
            layout: &scene_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_buffer.as_entire_binding(),
            }],
        });

        let sphere = IcoSphereMesh::generate(1);
        let renderer = InstanceRenderer::new(
            device,
            ctx.render_format(),
            &scene_bgl,
            &sphere.positions,
            &sphere.normals,
            &sphere.indices,
            InstanceRendererConfig::default_3d(),
        );

        let (_, depth_view) = create_depth_texture(device, size, DEPTH_FORMAT);

        self.renderer = Some(renderer);
        self.scene_buffer = Some(scene_buffer);
        self.scene_bind_group = Some(scene_bind_group);
        self.depth_view = Some(depth_view);
    }

    fn update(&mut self, ctx: &mut UpdateContext) {
        self.time = ctx.elapsed;

        let aspect = ctx.window_size.width as f32 / ctx.window_size.height.max(1) as f32;
        let vp = ctx.camera.view_projection_matrix(aspect);
        let scene_uniform = SceneUniform::new(vp, ctx.camera.position());
        ctx.gpu.queue.write_buffer(
            self.scene_buffer.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&scene_uniform),
        );

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

        self.renderer.as_mut().unwrap().update_instances(
            &ctx.gpu.device,
            &ctx.gpu.queue,
            &instances,
        );
    }

    fn render(&mut self, ctx: &mut RenderContext) {
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("instancing_3d_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.surface_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.02,
                        g: 0.02,
                        b: 0.05,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: self.depth_view.as_ref().unwrap(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_bind_group(0, self.scene_bind_group.as_ref().unwrap(), &[]);
        self.renderer.as_ref().unwrap().render(&mut pass);
    }

    fn gui(&mut self, egui_ctx: &mikage::egui::Context) {
        mikage::egui::Window::new("Info").show(egui_ctx, |ui| {
            ui.label("3D Instancing Demo");
            ui.label("Left drag: orbit | Right drag: pan | Scroll: zoom");
            ui.label(format!("Instances: {}", (5 * 2 + 1_i32).pow(3)));
        });
    }

    fn resize(&mut self, ctx: &GpuContext, new_size: PhysicalSize<u32>) {
        let (_, depth_view) = create_depth_texture(&ctx.device, new_size, DEPTH_FORMAT);
        self.depth_view = Some(depth_view);
    }
}

fn main() {
    let mut camera = OrbitCamera::default();
    camera.distance = 25.0;
    camera.pitch = 0.5;
    camera.yaw = 0.8;
    camera.damping = 0.85;

    mikage::run(
        Instancing3dApp {
            renderer: None,
            scene_buffer: None,
            scene_bind_group: None,
            depth_view: None,
            time: 0.0,
        },
        RunConfig {
            title: "mikage - 3D instancing".to_string(),
            camera: Box::new(camera),
            ..Default::default()
        },
    );
}
