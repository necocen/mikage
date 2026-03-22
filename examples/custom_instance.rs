//! Demonstrates a custom `InstanceVertex` with per-instance 2D rotation.
//!
//! Uses a 16-byte vertex (position + angle + scale) instead of the default
//! 32-byte `InstanceData`. Color is computed from the rotation angle in the
//! shader, saving bandwidth.

use bytemuck::{Pod, Zeroable};
use mikage::{
    App, Camera2d, GpuContext, InstanceRenderer, InstanceRendererConfig, InstanceVertex,
    RegularPolygonMesh, RenderContext, RunConfig, SceneUniform, ShaderProcessor, UpdateContext,
};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

/// Custom per-instance data: position + rotation + scale (16 bytes).
///
/// - xy: world position
/// - z: rotation angle (radians)
/// - w: uniform scale
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct RotatedInstance {
    pos_angle_scale: [f32; 4],
}

impl InstanceVertex for RotatedInstance {
    fn vertex_attributes() -> Vec<wgpu::VertexAttribute> {
        vec![wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 0,
            shader_location: 2,
        }]
    }
}

const SHADER_SOURCE: &str = include_str!("../assets/shaders/rotated_instancing.wgsl");

struct CustomInstanceApp {
    renderer: Option<InstanceRenderer<RotatedInstance>>,
    scene_buffer: Option<wgpu::Buffer>,
    scene_bind_group: Option<wgpu::BindGroup>,
    time: f64,
}

impl App for CustomInstanceApp {
    fn init(&mut self, ctx: &GpuContext, _size: PhysicalSize<u32>) {
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

        // Resolve shader imports
        let mut sp = ShaderProcessor::new();
        sp.register("mikage::scene_types", mikage::SCENE_TYPES_WGSL);
        let resolved = sp.resolve(SHADER_SOURCE).expect("failed to resolve shader");

        // Pentagon mesh
        let mesh = RegularPolygonMesh::generate(5);
        let renderer = InstanceRenderer::<RotatedInstance>::with_shader(
            device,
            ctx.render_format(),
            &scene_bgl,
            &mesh.positions,
            &mesh.normals,
            &mesh.indices,
            &resolved,
            InstanceRendererConfig {
                vertex_entry: "vertex",
                fragment_entry: "fragment",
                depth: false,
                sample_count: 1,
            },
        );

        self.renderer = Some(renderer);
        self.scene_buffer = Some(scene_buffer);
        self.scene_bind_group = Some(scene_bind_group);
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

        let t = self.time as f32;
        let spacing = 2.8;
        let scale = 1.0;

        // Compute viewport bounds
        let inv_vp = vp.inverse();
        let ndc_min = inv_vp.project_point3(glam::Vec3::new(-1.0, -1.0, 0.0));
        let ndc_max = inv_vp.project_point3(glam::Vec3::new(1.0, 1.0, 0.0));

        let margin = spacing * 2.0;
        let col_min = ((ndc_min.x - margin) / spacing).floor() as i32;
        let col_max = ((ndc_max.x + margin) / spacing).ceil() as i32;
        let row_min = ((ndc_min.y - margin) / spacing).floor() as i32;
        let row_max = ((ndc_max.y + margin) / spacing).ceil() as i32;

        let mut instances = Vec::new();
        for col in col_min..=col_max {
            for row in row_min..=row_max {
                let x = col as f32 * spacing;
                let y = row as f32 * spacing;

                // Each tile rotates at a different speed based on position
                let dist = (x * x + y * y).sqrt();
                let angle = t * (0.5 + (col + row) as f32 * 0.02) + dist * 0.1;

                instances.push(RotatedInstance {
                    pos_angle_scale: [x, y, angle, scale],
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
            label: Some("custom_instance_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.surface_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.08,
                        g: 0.08,
                        b: 0.12,
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
            ui.label("Custom InstanceVertex Demo");
            ui.label("Pentagons with per-instance 2D rotation");
            ui.label("16 bytes/instance (vs 32 for default)");
            ui.separator();
            ui.label("Left drag: pan | Scroll: zoom");
        });
    }

    fn resize(&mut self, _ctx: &GpuContext, _new_size: PhysicalSize<u32>) {}
}

fn main() {
    let mut camera = Camera2d::default();
    camera.zoom = 4.0;
    camera.damping = 0.85;

    mikage::run(
        CustomInstanceApp {
            renderer: None,
            scene_buffer: None,
            scene_bind_group: None,
            time: 0.0,
        },
        RunConfig {
            title: "mikage - custom instance vertex".to_string(),
            camera: Box::new(camera),
            ..Default::default()
        },
    );
}
