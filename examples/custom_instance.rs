//! Demonstrates a custom `InstanceVertex` with per-instance 2D rotation.
//!
//! Uses a 16-byte vertex (position + angle + scale) instead of the default
//! 32-byte `InstanceData`. Color is computed from the rotation angle in the
//! shader, saving bandwidth.

use bytemuck::{Pod, Zeroable};
use mikage::{
    App, Camera, Camera2d, FrameContext, GpuContext, InstanceRenderer, InstanceRendererConfig,
    InstanceVertex, RegularPolygonMesh, RunConfig, SceneBinding, ShaderProcessor, UpdateContext,
};
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

const SHADER_SOURCE: &str = include_str!("shaders/rotated_instancing.wgsl");

struct CustomInstanceApp {
    renderer: InstanceRenderer<RotatedInstance>,
    scene: SceneBinding,
    time: f64,
}

impl CustomInstanceApp {
    fn new(ctx: &GpuContext, _size: PhysicalSize<u32>) -> Self {
        let scene = SceneBinding::new(&ctx.device);

        // Resolve shader imports
        let sp = ShaderProcessor::new();
        let resolved = sp.resolve(SHADER_SOURCE).expect("failed to resolve shader");

        // Pentagon mesh
        let mesh = RegularPolygonMesh::generate(5);
        let renderer = InstanceRenderer::<RotatedInstance>::with_shader(
            ctx,
            scene.layout(),
            &mesh.positions,
            &mesh.normals,
            &mesh.indices,
            &resolved,
            InstanceRendererConfig {
                vertex_entry: "vertex",
                fragment_entry: "fragment",
                depth: false,
                storage_binding: false,
            },
        );

        Self {
            renderer,
            scene,
            time: 0.0,
        }
    }
}

impl App for CustomInstanceApp {
    type Camera = Camera2d;

    fn update(&mut self, ctx: &mut UpdateContext<Camera2d>) {
        self.time = ctx.elapsed;

        let aspect = ctx.window_size.width as f32 / ctx.window_size.height.max(1) as f32;
        self.scene
            .update_from_camera(&ctx.gpu.queue, &*ctx.camera, aspect);

        let vp = ctx.camera.view_projection_matrix(aspect);
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

        self.renderer
            .update_instances(&ctx.gpu.device, &ctx.gpu.queue, &instances);
    }

    fn encode(&mut self, ctx: &mut FrameContext<Camera2d>) {
        let color_attachment = ctx.color_attachment(wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color {
                r: 0.08,
                g: 0.08,
                b: 0.12,
                a: 1.0,
            }),
            store: wgpu::StoreOp::Store,
        });
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("custom_instance_pass"),
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
            ui.label("Custom InstanceVertex Demo");
            ui.label("Pentagons with per-instance 2D rotation");
            ui.label("16 bytes/instance (vs 32 for default)");
            ui.separator();
            ui.label("Left drag: pan | Scroll: zoom");
        });
    }
}

fn main() {
    let mut camera = Camera2d::default();
    camera.zoom = 4.0;
    camera.damping = 0.85;

    mikage::run(
        CustomInstanceApp::new,
        RunConfig {
            title: "mikage - custom instance vertex".to_string(),
            ..RunConfig::with_defaults(camera)
        },
    );
}
