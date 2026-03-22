use mikage::{
    App, Camera2d, GpuContext, InstanceData, InstanceRenderer, InstanceRendererConfig,
    RegularPolygonMesh, RenderContext, RunConfig, SceneUniform, UpdateContext,
};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

struct Instancing2dApp {
    renderer: Option<InstanceRenderer>,
    scene_buffer: Option<wgpu::Buffer>,
    scene_bind_group: Option<wgpu::BindGroup>,
}

impl App for Instancing2dApp {
    fn init(&mut self, ctx: &GpuContext, _size: PhysicalSize<u32>) {
        let scene_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let scene_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scene_buffer"),
                contents: bytemuck::bytes_of(&scene_uniform),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let scene_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_bind_group"),
            layout: &scene_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_buffer.as_entire_binding(),
            }],
        });

        let hex = RegularPolygonMesh::generate(6);
        let renderer = InstanceRenderer::new(
            &ctx.device,
            ctx.render_format(),
            &scene_bgl,
            &hex.positions,
            &hex.normals,
            &hex.indices,
            InstanceRendererConfig::default_2d(),
        );

        self.renderer = Some(renderer);
        self.scene_buffer = Some(scene_buffer);
        self.scene_bind_group = Some(scene_bind_group);
    }

    fn update(&mut self, ctx: &mut UpdateContext) {
        let aspect = ctx.window_size.width as f32 / ctx.window_size.height.max(1) as f32;
        let vp = ctx.camera.view_projection_matrix(aspect);
        let scene_uniform = SceneUniform::new(vp, ctx.camera.position());
        ctx.gpu.queue.write_buffer(
            self.scene_buffer.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&scene_uniform),
        );

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

        self.renderer.as_mut().unwrap().update_instances(
            &ctx.gpu.device,
            &ctx.gpu.queue,
            &instances,
        );
    }

    fn render(&mut self, ctx: &mut RenderContext) {
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("instancing_2d_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.surface_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.1,
                        b: 0.15,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_bind_group(0, self.scene_bind_group.as_ref().unwrap(), &[]);
        self.renderer.as_ref().unwrap().render(&mut pass);
    }

    fn gui(&mut self, egui_ctx: &mikage::egui::Context) {
        mikage::egui::Window::new("Info").show(egui_ctx, |ui| {
            ui.label("2D Instancing Demo");
            ui.label("Left drag: pan");
            ui.label("Scroll: zoom");
        });
    }

    fn resize(&mut self, _ctx: &GpuContext, _new_size: PhysicalSize<u32>) {}
}

fn main() {
    let mut camera = Camera2d::default();
    camera.zoom = 2.0;
    camera.damping = 0.85;

    mikage::run(
        Instancing2dApp {
            renderer: None,
            scene_buffer: None,
            scene_bind_group: None,
        },
        RunConfig {
            title: "mikage - 2D instancing".to_string(),
            camera: Box::new(camera),
            ..Default::default()
        },
    );
}
