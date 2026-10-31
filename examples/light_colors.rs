//! Neutral solid sphere and instanced cubes under cool ambient / warm directional light.
//! cargo run --example light_colors
//! cargo run --example light_colors -- --headless /tmp/light-colors.png

use mikage::{
    App, ColoredSceneBinding, CubeMesh, GpuContext, IcoSphereMesh, InstanceData, InstanceRenderer,
    InstanceRendererConfig, OrbitCamera, RenderContext, RenderTargetConfig, RenderUpdateContext,
    RunConfig, SceneLightColors, SceneLighting, SimulationPolicy, SolidRenderer,
    create_depth_texture,
    dpi::PhysicalSize,
    glam::{Mat4, Vec3, Vec4},
    wgpu,
};

struct LightColorsApp {
    scene: ColoredSceneBinding,
    solid: SolidRenderer,
    instances: InstanceRenderer,
    depth: wgpu::TextureView,
    target: RenderTargetConfig,
    ambient: f32,
    directional: f32,
}

impl LightColorsApp {
    fn new(gpu: &GpuContext, target: RenderTargetConfig, size: PhysicalSize<u32>) -> Self {
        let scene = ColoredSceneBinding::new(&gpu.device);
        scene.update_light_colors(
            &gpu.queue,
            SceneLightColors {
                // These are linear RGB values, not sRGB UI colors.
                ambient: Vec3::new(0.15, 0.35, 1.0),
                directional: Vec3::new(1.0, 0.55, 0.15),
            },
        );
        let mut solid = SolidRenderer::with_light_colors(gpu, target, &scene);
        let sphere = IcoSphereMesh::generate(3);
        let id = solid.add_object(gpu, &sphere.positions, &sphere.normals, &sphere.indices);
        solid.update_object(
            gpu,
            id,
            Mat4::from_translation(Vec3::new(-1.4, 0.0, 0.0)),
            Vec4::new(0.7, 0.7, 0.7, 1.0),
        );
        let cube = CubeMesh::generate();
        let mut instances = InstanceRenderer::with_light_colors(
            gpu,
            target,
            &scene,
            &cube.positions,
            &cube.normals,
            &cube.indices,
            InstanceRendererConfig::default_3d(),
        );
        instances.update_instances(
            gpu,
            &[
                InstanceData {
                    pos_scale: [1.1, -0.5, 0.0, 0.7],
                    color: [0.7, 0.7, 0.7, 1.0],
                },
                InstanceData {
                    pos_scale: [1.5, 0.65, 0.0, 0.5],
                    color: [0.7, 0.7, 0.7, 1.0],
                },
            ],
        );
        let (_, depth) = create_depth_texture(gpu, size, target);
        Self {
            scene,
            solid,
            instances,
            depth,
            target,
            ambient: 0.3,
            directional: 0.9,
        }
    }
}

impl App for LightColorsApp {
    type Camera = OrbitCamera;

    fn prepare_render(&mut self, ctx: &mut RenderUpdateContext<OrbitCamera>) {
        self.scene.update_lighting(
            &ctx.gpu.queue,
            SceneLighting {
                direction: Vec3::new(-1.0, 2.0, 2.0),
                ambient_intensity: self.ambient,
                directional_intensity: self.directional,
            },
        );
        self.scene.update_camera(
            &ctx.gpu.queue,
            ctx.camera,
            ctx.target_size.width as f32 / ctx.target_size.height.max(1) as f32,
        );
    }

    fn render(&mut self, ctx: &mut RenderContext<OrbitCamera>) {
        let color = ctx.color_attachment(wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color {
                r: 0.025,
                g: 0.025,
                b: 0.025,
                a: 1.0,
            }),
            store: wgpu::StoreOp::Store,
        });
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("colored_lights"),
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
        self.solid.render(&mut pass);
        self.instances.render(&mut pass);
    }

    fn resize(&mut self, gpu: &GpuContext, size: PhysicalSize<u32>) {
        let (_, depth) = create_depth_texture(gpu, size, self.target);
        self.depth = depth;
    }

    fn gui(&mut self, ui: &mut mikage::egui::Ui) {
        ui.label("Neutral materials; cool ambient and warm directional light.");
        ui.add(mikage::egui::Slider::new(&mut self.ambient, 0.0..=1.0).text("Ambient gain"));
        ui.add(
            mikage::egui::Slider::new(&mut self.directional, 0.0..=2.0).text("Directional gain"),
        );
        ui.label("Left: SolidRenderer. Right: InstanceRenderer. Simulation is paused.");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let camera = OrbitCamera {
        distance: 7.5,
        yaw: 0.3,
        pitch: 0.2,
        ..Default::default()
    };
    #[cfg(not(target_family = "wasm"))]
    if std::env::args().nth(1).as_deref() == Some("--headless") {
        let output = std::env::args()
            .nth(2)
            .unwrap_or_else(|| "/tmp/light-colors.png".into());
        let size = PhysicalSize::new(800, 400);
        let mut harness = pollster::block_on(mikage::HeadlessHarness::new(
            mikage::GpuDescriptor::default(),
            RenderTargetConfig {
                sample_count: 4,
                ..Default::default()
            },
            size,
            camera,
            LightColorsApp::new,
        ))?;
        harness.render_once()?;
        let capture = harness.capture_named("scene")?;
        assert_eq!(harness.runtime.progress().encoded_ticks, 0);
        image::save_buffer(
            &output,
            &capture.data?,
            size.width,
            size.height,
            image::ColorType::Rgba8,
        )?;
        println!("Saved {output} without advancing simulation.");
        return Ok(());
    }
    let config = RunConfig {
        sample_count: 4,
        ..RunConfig::new("mikage - light colors")
            .with_camera(camera)
            .with_simulation_policy(SimulationPolicy::Manual)
    };
    mikage::run(LightColorsApp::new, config)?;
    Ok(())
}
