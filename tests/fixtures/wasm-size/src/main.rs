//! Fixed workload for the WASM size budget. Keep application behavior stable.
use mikage::dpi::PhysicalSize;
use mikage::glam::{Mat4, Vec4};
use mikage::{
    App, CubeMesh, GpuContext, GpuDescriptor, OrbitCamera, RenderContext, RenderTargetConfig,
    RenderUpdateContext, RunConfig, SceneBinding, SolidRenderer, create_depth_texture, wgpu,
};

struct SizeApp {
    scene: SceneBinding,
    renderer: SolidRenderer,
    depth: wgpu::TextureView,
    target: RenderTargetConfig,
    gain: f32,
}

impl SizeApp {
    fn new(gpu: &GpuContext, target: RenderTargetConfig, size: PhysicalSize<u32>) -> Self {
        let scene = SceneBinding::new(&gpu.device);
        let mut renderer = SolidRenderer::new(gpu, target, scene.layout());
        let cube = CubeMesh::generate();
        let object = renderer.add_object(gpu, &cube.positions, &cube.normals, &cube.indices);
        renderer.update_object(gpu, object, Mat4::IDENTITY, Vec4::new(0.3, 0.6, 0.9, 1.0));
        Self {
            scene,
            renderer,
            depth: create_depth_texture(gpu, size, target).1,
            target,
            gain: 1.0,
        }
    }
}

impl App for SizeApp {
    type Camera = OrbitCamera;

    fn prepare_render(&mut self, ctx: &mut RenderUpdateContext<OrbitCamera>) {
        let aspect = ctx.target_size.width as f32 / ctx.target_size.height.max(1) as f32;
        self.scene
            .update_from_camera(&ctx.gpu.queue, ctx.camera, aspect);
    }

    fn render(&mut self, ctx: &mut RenderContext<OrbitCamera>) {
        let color = ctx.color_attachment(wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color {
                r: 0.04 * self.gain as f64,
                g: 0.06 * self.gain as f64,
                b: 0.09 * self.gain as f64,
                a: 1.0,
            }),
            store: wgpu::StoreOp::Store,
        });
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("wasm_size"),
            color_attachments: &[Some(color)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_bind_group(0, self.scene.bind_group(), &[]);
        self.renderer.render(&mut pass);
    }

    fn gui(&mut self, ui: &mut mikage::egui::Ui) {
        mikage::egui::Window::new("WASM size fixture").show(ui.ctx(), |ui| {
            ui.label("Drag to orbit; scroll to zoom.");
            ui.add(mikage::egui::Slider::new(&mut self.gain, 0.0..=1.0).text("Background"));
        });
    }

    fn resize(&mut self, gpu: &GpuContext, size: PhysicalSize<u32>) {
        self.depth = create_depth_texture(gpu, size, self.target).1;
    }
}

fn main() {
    let mut camera = OrbitCamera::default();
    camera.distance = 5.0;
    camera.pitch = 0.4;
    camera.yaw = 0.6;
    mikage::run(
        SizeApp::new,
        RunConfig::new("WASM size fixture")
            .with_camera(camera)
            .with_gpu(GpuDescriptor {
                backends: Some(wgpu::Backends::BROWSER_WEBGPU),
                ..Default::default()
            }),
    )
    .expect("WASM size fixture failed");
}
