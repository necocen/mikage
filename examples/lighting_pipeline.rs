//! Colored light + GPU-generated particles + directional shadows + HDR exposure.
//! cargo run --example lighting_pipeline
//! cargo run --example lighting_pipeline -- --headless /tmp/lighting.png --exposure -1
//! Add --no-shadows for an unshadowed reference. Native primary backends / WebGPU.

use mikage::{
    App, ColoredSceneBinding, CubeMesh, DirectionalShadow, DirectionalShadowConfig, GpuContext,
    GpuDescriptor, HDR_FORMAT, IcoSphereMesh, InstanceRenderer, InstanceRendererConfig,
    OffscreenTarget, OrbitCamera, QuadMesh2d, RenderContext, RenderTargetConfig,
    RenderUpdateContext, RunConfig, SceneLightColors, SceneLighting, ShadowOptions,
    SimulationPolicy, SolidRenderer, SolidShading, ToneMapSource, ToneMapper,
    create_compute_pipeline,
    dpi::PhysicalSize,
    glam::{
        Mat4, Quat, Vec3, Vec4,
        camera::rh::{proj::directx, view},
    },
    storage_buffer_entry, wgpu,
};

struct LightingPipelineApp {
    scene: ColoredSceneBinding,
    solid: SolidRenderer,
    particles: InstanceRenderer,
    shadow: DirectionalShadow,
    hdr: OffscreenTarget,
    mapper: ToneMapper,
    source: ToneMapSource,
    exposure: f32,
    ambient: f32,
    directional: f32,
    shadows_enabled: bool,
}

fn light_direction() -> Vec3 {
    Vec3::new(-4.0, 7.0, 4.0)
}
fn descriptor() -> GpuDescriptor {
    // Compute and half-float rendering are baseline here; WebGL is not targeted.
    GpuDescriptor {
        backends: Some(wgpu::Backends::PRIMARY),
        ..Default::default()
    }
}

impl LightingPipelineApp {
    fn new(gpu: &GpuContext, target: RenderTargetConfig, size: PhysicalSize<u32>) -> Self {
        let scene = ColoredSceneBinding::new(&gpu.device);
        scene.update_light_colors(
            &gpu.queue,
            SceneLightColors {
                ambient: Vec3::new(0.2, 0.35, 1.0),
                directional: Vec3::new(1.0, 0.75, 0.4),
            },
        );
        let light_matrix = directx::orthographic(-4.0, 4.0, -4.0, 4.0, 0.1, 30.0)
            * view::look_at_mat4(light_direction(), Vec3::ZERO, Vec3::Y);
        let shadow = DirectionalShadow::new(
            gpu,
            DirectionalShadowConfig {
                light_view_projection: light_matrix,
                resolution: 1024,
                depth_bias: 2,
                slope_bias: 2.0,
                ..Default::default()
            },
        )
        .expect("valid shadow settings");
        let hdr = OffscreenTarget::new(
            gpu,
            size,
            RenderTargetConfig {
                color_format: HDR_FORMAT,
                ..target
            },
        )
        .expect(
            "HDR example requires renderable/sampleable Rgba16Float and the requested MSAA count",
        );
        let hdr_config = hdr.render_target_config();
        let mut solid =
            SolidRenderer::with_light_colors_and_shadows(gpu, hdr_config, &scene, &shadow);
        let floor = QuadMesh2d::generate();
        let floor_id = solid.add_object(gpu, &floor.positions, &floor.normals, &floor.indices);
        solid.update_object(
            gpu,
            floor_id,
            Mat4::from_scale_rotation_translation(
                Vec3::new(6.0, 6.0, 1.0),
                Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                Vec3::new(0.0, -0.1, 0.0),
            ),
            Vec4::new(0.65, 0.65, 0.65, 1.0),
        );
        solid.set_shadow_options(
            floor_id,
            ShadowOptions {
                cast: false,
                receive: true,
            },
        );
        let sphere = IcoSphereMesh::generate(2);
        let solid_id = solid.add_object(gpu, &sphere.positions, &sphere.normals, &sphere.indices);
        solid.update_object(
            gpu,
            solid_id,
            Mat4::from_scale_rotation_translation(
                Vec3::new(0.45, 0.8, 0.45),
                Quat::IDENTITY,
                Vec3::new(-2.0, 0.7, 0.7),
            ),
            Vec4::new(0.7, 0.7, 0.7, 1.0),
        );
        // All opaque particles render before this transparent solid, in the HDR pass.
        let cube = CubeMesh::generate();
        let wall_id = solid.add_object(gpu, &cube.positions, &cube.normals, &cube.indices);
        solid.update_object(
            gpu,
            wall_id,
            Mat4::from_scale_rotation_translation(
                Vec3::new(0.06, 1.4, 3.0),
                Quat::IDENTITY,
                Vec3::new(1.8, 0.6, 0.0),
            ),
            Vec4::new(0.3, 0.7, 0.8, 0.35),
        );
        solid.set_shading(wall_id, SolidShading::Lit);
        solid.set_shadow_options(
            wall_id,
            ShadowOptions {
                cast: false,
                receive: true,
            },
        );
        let mut particles = InstanceRenderer::with_light_colors_and_shadows(
            gpu,
            hdr_config,
            &scene,
            &shadow,
            &sphere.positions,
            &sphere.normals,
            &sphere.indices,
            InstanceRendererConfig {
                storage_binding: true,
                ..InstanceRendererConfig::default_3d()
            },
        );
        let buffer = particles.prepare_compute(&gpu.device, 25).buffer;
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("particle_init"),
                entries: &[storage_buffer_entry(0, wgpu::ShaderStages::COMPUTE, false)],
            });
        let pipeline =
            create_compute_pipeline(&gpu.device, "particle_init", PARTICLES, &[&layout], "main");
        let binding = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_init"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &binding, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        gpu.queue.submit([encoder.finish()]);
        let mapper = ToneMapper::new(gpu, target).expect("float-compatible display target");
        let source = mapper.source(gpu, hdr.view());
        Self {
            scene,
            solid,
            particles,
            shadow,
            hdr,
            mapper,
            source,
            exposure: 0.0,
            ambient: 0.4,
            directional: 3.0,
            shadows_enabled: true,
        }
    }
}

impl App for LightingPipelineApp {
    type Camera = OrbitCamera;
    fn prepare_render(&mut self, ctx: &mut RenderUpdateContext<OrbitCamera>) {
        self.scene.update_lighting(
            &ctx.gpu.queue,
            SceneLighting {
                direction: light_direction(),
                ambient_intensity: self.ambient,
                directional_intensity: self.directional,
            },
        );
        self.scene.update_camera(
            &ctx.gpu.queue,
            ctx.camera,
            ctx.target_size.width as f32 / ctx.target_size.height.max(1) as f32,
        );
        self.mapper.update_exposure(&ctx.gpu.queue, self.exposure);
        self.shadow
            .set_enabled(&ctx.gpu.queue, self.shadows_enabled);
    }
    fn render(&mut self, ctx: &mut RenderContext<OrbitCamera>) {
        if let Some(mut pass) = self.shadow.begin_pass(ctx.encoder) {
            self.solid.render_shadow(&mut pass);
            self.particles.render_shadow(&mut pass);
        }
        {
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("linear_hdr_scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.hdr.color_view(),
                    resolve_target: self.hdr.resolve_target(),
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.025,
                            g: 0.025,
                            b: 0.025,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.hdr.depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_bind_group(0, self.scene.bind_group(), &[]);
            self.particles.render(&mut pass); // opaque instances first
            self.solid.render(&mut pass); // solid opaque geometry, then transparent wall
        }
        // Sampling the resolved HDR view after the scene pass avoids clipping.
        self.mapper.encode(ctx.encoder, &self.source, ctx.target);
    }
    fn resize(&mut self, gpu: &GpuContext, size: PhysicalSize<u32>) {
        self.hdr.resize(gpu, size).expect("resize HDR target");
        self.source = self.mapper.source(gpu, self.hdr.view());
    }
    fn gui(&mut self, ui: &mut mikage::egui::Ui) {
        ui.label("Colored lighting, directional shadows, linear HDR → display sRGB");
        ui.add(mikage::egui::Slider::new(&mut self.exposure, -4.0..=4.0).text("Exposure (EV)"));
        ui.add(mikage::egui::Slider::new(&mut self.ambient, 0.0..=2.0).text("Ambient gain"));
        ui.add(
            mikage::egui::Slider::new(&mut self.directional, 0.0..=8.0).text("Directional gain"),
        );
        ui.add_enabled(
            self.shadow.is_supported(),
            mikage::egui::Checkbox::new(&mut self.shadows_enabled, "Directional shadows"),
        );
        ui.label("Simulation is paused. Orbiting and exposure changes do not advance ticks.");
    }
}

const PARTICLES: &str = r#"
struct Instance { pos_scale: vec4<f32>, color: vec4<f32> };
@group(0) @binding(0) var<storage, read_write> particles: array<Instance>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= 25u { return; }
    let x = (f32(id.x % 5u) - 2.0) * 0.6;
    let z = (f32(id.x / 5u) - 2.0) * 0.6;
    particles[id.x] = Instance(vec4<f32>(x, 0.35 + 0.1 * sin(f32(id.x)), z, 0.22), vec4<f32>(0.75, 0.75, 0.75, 1.0));
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut camera = OrbitCamera::default();
    camera.distance = 8.0;
    camera.pitch = 0.55;
    camera.yaw = 0.7;
    camera.target = Vec3::new(0.0, 0.2, 0.0);
    #[cfg(not(target_family = "wasm"))]
    {
        let args: Vec<String> = std::env::args().collect();
        if let Some(index) = args.iter().position(|arg| arg == "--headless") {
            let output = args
                .get(index + 1)
                .filter(|arg| !arg.starts_with("--"))
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| std::env::temp_dir().join("lighting-pipeline.png"));
            let ev = args
                .windows(2)
                .find(|pair| pair[0] == "--exposure")
                .map(|pair| pair[1].parse::<f32>())
                .transpose()?
                .unwrap_or(0.0);
            let enabled = !args.iter().any(|arg| arg == "--no-shadows");
            let size = PhysicalSize::new(800, 500);
            let mut harness = pollster::block_on(mikage::HeadlessHarness::new(
                descriptor(),
                RenderTargetConfig {
                    sample_count: 4,
                    ..Default::default()
                },
                size,
                camera,
                |gpu, target, size| {
                    let mut app = LightingPipelineApp::new(gpu, target, size);
                    app.exposure = ev;
                    app.shadows_enabled = enabled;
                    app
                },
            ))?;
            harness.render_once()?;
            let result = harness.capture_named("scene")?;
            assert_eq!(harness.runtime.progress().encoded_ticks, 0);
            image::save_buffer(
                &output,
                &result.data?,
                size.width,
                size.height,
                image::ColorType::Rgba8,
            )?;
            println!(
                "Saved {} at EV {ev}, shadows={enabled}, ticks=0",
                output.display()
            );
            return Ok(());
        }
    }
    mikage::run(
        LightingPipelineApp::new,
        RunConfig {
            gpu: descriptor(),
            sample_count: 4,
            ..RunConfig::new("mikage - lighting pipeline")
                .with_camera(camera)
                .with_simulation_policy(SimulationPolicy::Manual)
        },
    )?;
    Ok(())
}
