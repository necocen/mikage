//! Run with `cargo run --example agent_capture --features agent`.
//! POST /command {"op":"app.gpu_command","payload":{"reset":42}} and
//! capture the registered `values` target as raw bytes using scripts/capture.py.

#[cfg(all(feature = "window", feature = "agent", not(target_family = "wasm")))]
mod demo {
    use mikage::wgpu::{self, util::DeviceExt};
    use mikage::{
        App, CaptureRegistry, CommandContext, GpuContext, ReadbackId, ReadbackMetadata,
        ReadbackRing, RenderContext, TickContext,
    };
    use serde_json::{Value, json};
    use std::collections::HashMap;

    pub struct CaptureApp {
        values: wgpu::Buffer,
        pipeline: wgpu::ComputePipeline,
        bind_group: wgpu::BindGroup,
        readbacks: ReadbackRing,
        completed_readbacks: HashMap<ReadbackId, mikage::ReadbackResult>,
        tick: u64,
        last_values: Vec<u32>,
    }

    impl CaptureApp {
        pub fn new(gpu: &GpuContext) -> Self {
            let values = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("agent_capture_values"),
                    contents: bytemuck::cast_slice(&[1u32, 2, 3, 4]),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                });
            let layout = gpu
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("agent_capture_layout"),
                    entries: &[mikage::storage_buffer_entry(
                        0,
                        wgpu::ShaderStages::COMPUTE,
                        false,
                    )],
                });
            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("agent_capture_bind_group"),
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: values.as_entire_binding(),
                }],
            });
            let pipeline = mikage::create_compute_pipeline(
                &gpu.device,
                "agent_capture_increment",
                "@group(0) @binding(0) var<storage,read_write> values: array<u32>; @compute @workgroup_size(4) fn main(@builtin(local_invocation_index) i:u32) { values[i] += 1u; }",
                &[&layout],
                "main",
            );
            Self {
                values,
                pipeline,
                bind_group,
                readbacks: ReadbackRing::default(),
                completed_readbacks: HashMap::new(),
                tick: 0,
                last_values: Vec::new(),
            }
        }
    }

    impl App for CaptureApp {
        type Camera = ();

        fn tick(&mut self, ctx: &mut TickContext<'_>) {
            self.tick = ctx.tick_id;
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        fn render(&mut self, ctx: &mut RenderContext<'_, ()>) {
            let attachment = ctx.color_attachment(wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.025,
                    g: 0.06,
                    b: 0.09,
                    a: 1.0,
                }),
                store: wgpu::StoreOp::Store,
            });
            let _pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("agent_capture_background"),
                color_attachments: &[Some(attachment)],
                ..Default::default()
            });
        }

        #[cfg(feature = "gui")]
        fn gui(&mut self, ui: &mut mikage::egui::Ui) {
            ui.heading("Agent capture example");
            ui.label(format!("Simulation tick: {}", self.tick));
            ui.label(format!(
                "Last completed GPU diagnostic: {:?}",
                self.last_values
            ));
            ui.label("Capture window/scene as PNG, or values as raw bytes.");
        }

        fn capture_targets(&self, registry: &mut CaptureRegistry) {
            registry.register_buffer("values", &self.values, 0, self.values.size());
        }

        fn agent_status(&self) -> Value {
            json!({"tick":self.tick,"last_values":self.last_values})
        }

        fn encode_agent_command(
            &mut self,
            payload: Value,
            ctx: &mut CommandContext<'_>,
        ) -> Result<Value, String> {
            if let Some(seed) = payload.get("reset") {
                let seed = seed
                    .as_u64()
                    .and_then(|seed| u32::try_from(seed).ok())
                    .ok_or("reset must be a u32")?;
                let initial = [seed; 4];
                let source = ctx
                    .gpu
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("agent_capture_reset"),
                        contents: bytemuck::cast_slice(&initial),
                        usage: wgpu::BufferUsages::COPY_SRC,
                    });
                ctx.encoder
                    .copy_buffer_to_buffer(&source, 0, &self.values, 0, self.values.size());
            }
            let id = self
                .readbacks
                .enqueue_buffer(
                    &ctx.gpu.device,
                    ctx.encoder,
                    &self.values,
                    0,
                    self.values.size(),
                    ReadbackMetadata {
                        target: "values".into(),
                        tick_id: Some(ctx.tick_id),
                        ..Default::default()
                    },
                )
                .map_err(|error| error.to_string())?;
            Ok(json!({"readback_id":id.0}))
        }

        fn complete_agent_command(
            &mut self,
            _gpu: &GpuContext,
            response: Value,
            token: &mikage::SubmissionToken,
        ) -> Result<Value, String> {
            let id = ReadbackId(
                response["readback_id"]
                    .as_u64()
                    .ok_or("missing readback ID")?,
            );
            for result in self.readbacks.take_ready() {
                self.completed_readbacks.insert(result.id, result);
            }
            let result = self
                .completed_readbacks
                .remove(&id)
                .ok_or("diagnostic readback is not ready")?;
            let bytes = result.data.map_err(|error| error.to_string())?;
            self.last_values = bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            Ok(
                json!({"values":self.last_values,"tick_id":result.metadata.tick_id,"submission_id":token.id}),
            )
        }
    }
}

#[cfg(all(feature = "window", feature = "agent", not(target_family = "wasm")))]
fn main() {
    let mut policy = mikage::SimulationPolicy::PerRedraw;
    let mut agent = mikage::AgentConfig::default();
    let mut sample_count = 1;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--fixed" => {
                policy = mikage::SimulationPolicy::fixed(std::time::Duration::from_millis(10));
            }
            "--manual" => policy = mikage::SimulationPolicy::Manual,
            "--msaa4" => sample_count = 4,
            "--port" => {
                let port = args
                    .next()
                    .expect("--port requires a port number")
                    .parse::<u16>()
                    .expect("--port must be between 0 and 65535");
                agent.bind_addr.set_port(port);
            }
            "--connection-file" => {
                agent.write_connection_file = Some(
                    args.next()
                        .expect("--connection-file requires a path")
                        .into(),
                );
            }
            "--help" => {
                println!(
                    "agent_capture [--fixed|--manual] [--msaa4] [--port PORT] [--connection-file PATH]"
                );
                return;
            }
            _ => panic!("unknown argument: {arg}"),
        }
    }
    mikage::run_with_agent(
        |gpu, _target, _size| demo::CaptureApp::new(gpu),
        mikage::RunConfig {
            title: "mikage - agent capture".into(),
            simulation_policy: policy,
            sample_count,
            redraw_policy: mikage::RedrawPolicy::Reactive,
            ..Default::default()
        },
        agent,
    )
    .expect("mikage application failed");
}

#[cfg(not(all(feature = "window", feature = "agent", not(target_family = "wasm"))))]
fn main() {
    eprintln!("This example requires native window and agent features.");
}
