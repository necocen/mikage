use mikage::{
    App, DEPTH_FORMAT, FrameContext, GpuContext, MeshBuffers, OrbitCamera, POSITION_NORMAL_LAYOUT,
    RunConfig, SceneBinding, ShaderProcessor, UpdateContext, create_depth_texture,
};
use winit::dpi::PhysicalSize;

struct DemoApp {
    render_pipeline: wgpu::RenderPipeline,
    mesh: MeshBuffers,
    scene: SceneBinding,
    depth_view: wgpu::TextureView,
    time: f64,
}

impl DemoApp {
    fn new(ctx: &GpuContext, size: PhysicalSize<u32>) -> Self {
        let sphere = mikage::IcoSphereMesh::generate(2);
        let mesh = MeshBuffers::from_position_normal(
            &ctx.device,
            &sphere.positions,
            &sphere.normals,
            &sphere.indices,
        );

        let scene = SceneBinding::new(&ctx.device);

        let sp = ShaderProcessor::new();
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
                    buffers: &[POSITION_NORMAL_LAYOUT],
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

        Self {
            render_pipeline,
            mesh,
            scene,
            depth_view,
            time: 0.0,
        }
    }
}

impl App for DemoApp {
    type Camera = OrbitCamera;

    fn update(&mut self, ctx: &mut UpdateContext<OrbitCamera>) {
        self.time = ctx.elapsed;
        let aspect = ctx.window_size.width as f32 / ctx.window_size.height.max(1) as f32;
        self.scene
            .update_from_camera(&ctx.gpu.queue, &*ctx.camera, aspect);
    }

    fn encode(&mut self, ctx: &mut FrameContext<OrbitCamera>) {
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
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, self.scene.bind_group(), &[]);
        render_pass.set_vertex_buffer(0, self.mesh.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.mesh.index_count, 0, 0..1);
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
        self.depth_view = depth_view;
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
        DemoApp::new,
        RunConfig {
            title: "mikage WASM demo".to_string(),
            camera,
            ..Default::default()
        },
    );
}
