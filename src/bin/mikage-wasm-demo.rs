use mikage::{
    App, DEPTH_FORMAT, GpuContext, OrbitCamera, RenderContext, RunConfig, SceneBinding,
    ShaderProcessor, UpdateContext, create_depth_texture,
};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

struct DemoApp {
    render_pipeline: Option<wgpu::RenderPipeline>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    index_count: u32,
    scene: Option<SceneBinding>,
    depth_view: Option<wgpu::TextureView>,
    time: f64,
}

impl App for DemoApp {
    fn init(&mut self, ctx: &GpuContext, size: PhysicalSize<u32>) {
        let mesh = mikage::IcoSphereMesh::generate(2);

        let mut vertex_data: Vec<f32> = Vec::with_capacity(mesh.positions.len() * 6);
        for i in 0..mesh.positions.len() {
            vertex_data.extend_from_slice(&mesh.positions[i]);
            vertex_data.extend_from_slice(&mesh.normals[i]);
        }

        let vertex_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex_buffer"),
                contents: bytemuck::cast_slice(&vertex_data),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("index_buffer"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        self.index_count = mesh.indices.len() as u32;

        let scene = SceneBinding::new(&ctx.device);

        let mut sp = ShaderProcessor::new();
        sp.register("mikage::scene_types", mikage::SCENE_TYPES_WGSL);
        let resolved = sp
            .resolve(SHADER)
            .expect("failed to resolve demo shader imports");
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sphere_shader"),
                source: wgpu::ShaderSource::Wgsl(resolved.into()),
            });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pipeline_layout"),
                bind_group_layouts: &[scene.layout()],
                push_constant_ranges: &[],
            });

        let render_pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("render_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 24,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 12,
                                shader_location: 1,
                            },
                        ],
                    }],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.render_format(),
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        let (_, depth_view) = create_depth_texture(&ctx.device, size, DEPTH_FORMAT);

        self.render_pipeline = Some(render_pipeline);
        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(index_buffer);
        self.scene = Some(scene);
        self.depth_view = Some(depth_view);
    }

    fn update(&mut self, ctx: &mut UpdateContext) {
        self.time = ctx.elapsed;
        let aspect = ctx.window_size.width as f32 / ctx.window_size.height.max(1) as f32;
        self.scene
            .as_ref()
            .unwrap()
            .update_from_camera(&ctx.gpu.queue, &*ctx.camera, aspect);
    }

    fn render(&mut self, ctx: &mut RenderContext) {
        let mut render_pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.surface_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.05,
                        g: 0.05,
                        b: 0.08,
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

        render_pass.set_pipeline(self.render_pipeline.as_ref().unwrap());
        render_pass.set_bind_group(0, self.scene.as_ref().unwrap().bind_group(), &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.as_ref().unwrap().slice(..));
        render_pass.set_index_buffer(
            self.index_buffer.as_ref().unwrap().slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }

    fn gui(&mut self, egui_ctx: &mikage::egui::Context) {
        mikage::egui::Window::new("Info").show(egui_ctx, |ui| {
            ui.label("mikage WASM Demo");
            ui.label("Drag: orbit | Right drag: pan | Scroll: zoom");
            ui.label(format!("Time: {:.1}s", self.time));
        });
    }

    fn resize(&mut self, ctx: &GpuContext, new_size: PhysicalSize<u32>) {
        let (_, depth_view) = create_depth_texture(&ctx.device, new_size, DEPTH_FORMAT);
        self.depth_view = Some(depth_view);
    }
}

const SHADER: &str = r#"
#import mikage::scene_types
@group(0) @binding(0) var<uniform> scene: SceneUniform;
struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) normal: vec3<f32> };
@vertex fn vs_main(@location(0) pos: vec3<f32>, @location(1) normal: vec3<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = scene.view_proj * vec4<f32>(pos, 1.0);
    out.normal = normal;
    return out;
}
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = max(dot(normalize(in.normal), scene.light_dir.xyz), 0.0);
    let color = vec3<f32>(0.4, 0.6, 0.9);
    return vec4<f32>(color * (scene.ambient.x + d * (1.0 - scene.ambient.x)), 1.0);
}
"#;

fn main() {
    let mut camera = OrbitCamera::default();
    camera.distance = 3.0;
    camera.pitch = 0.4;
    camera.yaw = 0.6;

    mikage::run(
        DemoApp {
            render_pipeline: None,
            vertex_buffer: None,
            index_buffer: None,
            index_count: 0,
            scene: None,
            depth_view: None,
            time: 0.0,
        },
        RunConfig {
            title: "mikage WASM demo".to_string(),
            camera: Box::new(camera),
            ..Default::default()
        },
    );
}
