//! 2D Boids flocking simulation — 10,000 entities on GPU compute.
//!
//! Demonstrates `InstanceRenderer` with GPU compute shader integration:
//! the compute shader writes directly to the instance buffer with no CPU readback.

use bytemuck::{Pod, Zeroable};
use mikage::{
    App, Camera2d, FrameContext, GpuContext, InstanceRenderer, InstanceRendererConfig,
    InstanceVertex, RegularPolygonMesh, RunConfig, SceneBinding, SceneUniform, ShaderProcessor,
    UniformBuffer, UpdateContext, create_compute_pipeline, create_storage_buffer_init,
    storage_buffer_entry, uniform_buffer_entry,
};
use winit::dpi::PhysicalSize;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

const NUM_BOIDS: u32 = 20_000;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BoidState {
    pos: [f32; 2],
    vel: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BoidParams {
    num_boids: u32,
    dt: f32,
    separation_radius: f32,
    alignment_radius: f32,
    cohesion_radius: f32,
    separation_strength: f32,
    alignment_strength: f32,
    cohesion_strength: f32,
    max_speed: f32,
    min_speed: f32,
    world_size: f32,
    boid_scale: f32,
    fov_cosine: f32, // cos(half_angle); -1.0 = 360°, 0.0 = 180°, 0.5 = 120°
    _pad: [f32; 3],
}

impl Default for BoidParams {
    fn default() -> Self {
        Self {
            num_boids: NUM_BOIDS,
            dt: 0.016,
            separation_radius: 0.8,
            alignment_radius: 2.0,
            cohesion_radius: 3.0,
            separation_strength: 2.5,
            alignment_strength: 1.0,
            cohesion_strength: 0.5,
            max_speed: 5.0,
            min_speed: 1.0,
            world_size: 300.0,
            boid_scale: 0.3,
            fov_cosine: 0.0, // 180°
            _pad: [0.0; 3],
        }
    }
}

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

// ---------------------------------------------------------------------------
// Simple deterministic PRNG (splitmix32)
// ---------------------------------------------------------------------------

fn splitmix32(state: &mut u32) -> u32 {
    *state = state.wrapping_add(0x9e37_79b9);
    let mut z = *state;
    z = (z ^ (z >> 16)).wrapping_mul(0x85eb_ca6b);
    z = (z ^ (z >> 13)).wrapping_mul(0xc2b2_ae35);
    z ^ (z >> 16)
}

fn rand_f32(state: &mut u32) -> f32 {
    (splitmix32(state) as f32) / (u32::MAX as f32)
}

fn generate_initial_boids(count: u32, world_size: f32) -> Vec<BoidState> {
    let mut rng = 42u32;
    (0..count)
        .map(|_| {
            let px = (rand_f32(&mut rng) * 2.0 - 1.0) * world_size;
            let py = (rand_f32(&mut rng) * 2.0 - 1.0) * world_size;
            let angle = rand_f32(&mut rng) * std::f32::consts::TAU;
            let speed = rand_f32(&mut rng) * 3.0 + 1.0;
            BoidState {
                pos: [px, py],
                vel: [angle.cos() * speed, angle.sin() * speed],
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

const RENDER_SHADER: &str = include_str!("shaders/boids_render.wgsl");
const COMPUTE_SHADER: &str = include_str!("shaders/boids_compute.wgsl");

struct BoidsApp {
    renderer: InstanceRenderer<RotatedInstance>,
    tile_scenes: [SceneBinding; 9],
    compute_pipeline: wgpu::ComputePipeline,
    boid_buffers: [wgpu::Buffer; 2],
    params_buffer: UniformBuffer<BoidParams>,
    #[allow(dead_code)] // retained for potential bind group rebuilds
    compute_bind_group_layout: wgpu::BindGroupLayout,
    compute_bind_groups: [wgpu::BindGroup; 2],
    frame_index: u32,

    params: BoidParams,
    fov_degrees: f32, // UI-friendly; synced to params.fov_cosine
    paused: bool,
    reset_requested: bool,
}

impl BoidsApp {
    fn new(ctx: &GpuContext, _size: PhysicalSize<u32>) -> Self {
        let device = &ctx.device;
        let params = BoidParams::default();

        let tile_scenes: [SceneBinding; 9] = std::array::from_fn(|_| SceneBinding::new(device));

        // Render shader (resolve imports)
        let sp = ShaderProcessor::new();
        let resolved_render = sp.resolve(RENDER_SHADER).expect("render shader");

        // Triangle mesh (pointing right = +X)
        let mesh = RegularPolygonMesh::generate(3);
        let mut renderer = InstanceRenderer::<RotatedInstance>::with_shader(
            device,
            ctx.render_format(),
            tile_scenes[0].layout(),
            &mesh.positions,
            &mesh.normals,
            &mesh.indices,
            &resolved_render,
            InstanceRendererConfig {
                vertex_entry: "vertex",
                fragment_entry: "fragment",
                depth: false,
                sample_count: 1,
            },
        );
        renderer.ensure_capacity(device, params.num_boids);
        renderer.set_instance_count(params.num_boids);

        // Boid state buffers (double-buffered)
        let initial = generate_initial_boids(params.num_boids, params.world_size);
        let boid_buffers = [0, 1]
            .map(|i| create_storage_buffer_init(device, &format!("boid_state_{i}"), &initial));

        // Params uniform buffer
        let params_buffer = UniformBuffer::new(device, "boid_params", &params);

        // Compute pipeline
        let cs = wgpu::ShaderStages::COMPUTE;
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("boids_compute_bgl"),
                entries: &[
                    storage_buffer_entry(0, cs, true),  // boids_in
                    storage_buffer_entry(1, cs, false), // boids_out
                    storage_buffer_entry(2, cs, false), // instances
                    uniform_buffer_entry(3, cs),        // params
                ],
            });

        let compute_pipeline = create_compute_pipeline(
            device,
            "boids_compute",
            COMPUTE_SHADER,
            &[&compute_bind_group_layout],
            "update_boids",
        );

        // Build compute bind groups
        let compute_bind_groups = Self::build_compute_bind_groups(
            device,
            &compute_bind_group_layout,
            &boid_buffers,
            params_buffer.buffer(),
            renderer.instance_buffer(),
        );

        Self {
            renderer,
            tile_scenes,
            compute_pipeline,
            boid_buffers,
            params_buffer,
            compute_bind_group_layout,
            compute_bind_groups,
            frame_index: 0,
            params,
            fov_degrees: 180.0,
            paused: false,
            reset_requested: false,
        }
    }

    fn build_compute_bind_groups(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        bufs: &[wgpu::Buffer; 2],
        params_buf: &wgpu::Buffer,
        instance_buf: &wgpu::Buffer,
    ) -> [wgpu::BindGroup; 2] {
        let make_bg = |label: &str, buf_in: &wgpu::Buffer, buf_out: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: instance_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            })
        };

        let bg0 = make_bg("boids_bg_a2b", &bufs[0], &bufs[1]);
        let bg1 = make_bg("boids_bg_b2a", &bufs[1], &bufs[0]);
        [bg0, bg1]
    }
}

impl App for BoidsApp {
    fn update(&mut self, ctx: &mut UpdateContext) {
        // Handle reset
        if self.reset_requested {
            self.reset_requested = false;
            let initial = generate_initial_boids(self.params.num_boids, self.params.world_size);
            let data = bytemuck::cast_slice(&initial);
            for buf in &self.boid_buffers {
                ctx.gpu.queue.write_buffer(buf, 0, data);
            }
            self.frame_index = 0;
        }

        // Update 3x3 tile scene bindings centered on the camera position
        let aspect = ctx.window_size.width as f32 / ctx.window_size.height.max(1) as f32;
        let vp = ctx.camera.view_projection_matrix(aspect);
        let camera_pos = ctx.camera.position();
        let tile_size = self.params.world_size * 2.0;

        // Find the base tile the camera is currently in
        let base_tx = (camera_pos.x / tile_size).round() as i32;
        let base_ty = (camera_pos.y / tile_size).round() as i32;

        let mut idx = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let tx = base_tx + dx;
                let ty = base_ty + dy;
                let offset = glam::Vec3::new(tx as f32 * tile_size, ty as f32 * tile_size, 0.0);
                let vp_tile = vp * glam::Mat4::from_translation(offset);
                let uniform = SceneUniform::new(vp_tile, camera_pos);
                self.tile_scenes[idx].update(&ctx.gpu.queue, &uniform);
                idx += 1;
            }
        }

        // Sync FOV degrees → cosine
        self.params.fov_cosine = (self.fov_degrees.to_radians() / 2.0).cos();

        // Cap dt to prevent instability
        self.params.dt = ctx.dt.min(1.0 / 30.0);
        self.params_buffer.write(&ctx.gpu.queue, &self.params);
    }

    fn encode(&mut self, ctx: &mut FrameContext) {
        // Compute pass
        if !self.paused {
            let bg_index = (self.frame_index % 2) as usize;
            let workgroups = self.params.num_boids.div_ceil(64);

            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("boids_compute_pass"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &self.compute_bind_groups[bg_index], &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
            drop(pass);

            self.frame_index += 1;
        }

        // Render pass
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("boids_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.surface_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.02,
                        g: 0.02,
                        b: 0.06,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Render 3x3 tiles for seamless periodic boundaries
        for tile_scene in &self.tile_scenes {
            pass.set_bind_group(0, tile_scene.bind_group(), &[]);
            self.renderer.render(&mut pass);
        }
    }

    fn gui(&mut self, egui_ctx: &mikage::egui::Context) {
        mikage::egui::SidePanel::left("boids_panel")
            .default_width(220.0)
            .show(egui_ctx, |ui| {
                ui.heading("Boids");
                ui.label(format!("Entities: {}", self.params.num_boids));
                ui.separator();

                ui.horizontal(|ui| {
                    if ui
                        .button(if self.paused { "Resume" } else { "Pause" })
                        .clicked()
                    {
                        self.paused = !self.paused;
                    }
                    if ui.button("Reset").clicked() {
                        self.reset_requested = true;
                    }
                });

                ui.separator();
                ui.label("Separation");
                ui.add(
                    mikage::egui::Slider::new(&mut self.params.separation_radius, 0.1..=5.0)
                        .text("radius"),
                );
                ui.add(
                    mikage::egui::Slider::new(&mut self.params.separation_strength, 0.0..=10.0)
                        .text("strength"),
                );

                ui.separator();
                ui.label("Alignment");
                ui.add(
                    mikage::egui::Slider::new(&mut self.params.alignment_radius, 0.1..=10.0)
                        .text("radius"),
                );
                ui.add(
                    mikage::egui::Slider::new(&mut self.params.alignment_strength, 0.0..=5.0)
                        .text("strength"),
                );

                ui.separator();
                ui.label("Cohesion");
                ui.add(
                    mikage::egui::Slider::new(&mut self.params.cohesion_radius, 0.1..=10.0)
                        .text("radius"),
                );
                ui.add(
                    mikage::egui::Slider::new(&mut self.params.cohesion_strength, 0.0..=5.0)
                        .text("strength"),
                );

                ui.separator();
                ui.label("Speed");
                ui.add(
                    mikage::egui::Slider::new(&mut self.params.max_speed, 1.0..=20.0).text("max"),
                );
                ui.add(
                    mikage::egui::Slider::new(&mut self.params.min_speed, 0.0..=5.0).text("min"),
                );

                ui.separator();
                ui.add(
                    mikage::egui::Slider::new(&mut self.params.boid_scale, 0.05..=1.0)
                        .text("scale"),
                );
                ui.add(mikage::egui::Slider::new(&mut self.fov_degrees, 30.0..=360.0).text("FOV"));
            });
    }
}

fn main() {
    let mut camera = Camera2d::default();
    camera.zoom = 0.02;
    camera.damping = 0.85;

    mikage::run(
        BoidsApp::new,
        RunConfig {
            title: "mikage - GPU boids (20k)".to_string(),
            camera: Box::new(camera),
            ..Default::default()
        },
    );
}
