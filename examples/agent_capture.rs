//! Run with `cargo run --example agent_capture --features agent`.
//! POST /command {"op":"app.gpu_command","payload":{"reset":42}} and
//! capture the registered `values` target as raw bytes using scripts/capture.py.

#[cfg(all(feature = "window", feature = "agent", not(target_family = "wasm")))]
mod demo {
    use mikage::wgpu::{self, util::DeviceExt};
    use mikage::{
        App, CaptureRegistry, CommandContext, GpuContext, ReadbackId, ReadbackMetadata,
        ReadbackRing, RedrawRequest, RenderContext, RenderUpdateContext, SimulationControl,
        SubmissionToken, TickContext,
    };
    use serde_json::{Value, json};
    use std::collections::HashMap;
    use std::sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    };
    use std::time::{Duration, Instant};

    pub struct CaptureApp {
        values: wgpu::Buffer,
        pipeline: wgpu::ComputePipeline,
        bind_group: wgpu::BindGroup,
        readbacks: ReadbackRing,
        completed_readbacks: HashMap<ReadbackId, mikage::ReadbackResult>,
        tick: u64,
        last_values: Vec<u32>,
        cpu_tick_work: Duration,
        command_probe: u64,
        gui_input_count: u64,
        simulation_control: Option<SimulationControl>,
        control_installs: u64,
        redraw_request: Option<RedrawRequest>,
        pause_on_next_tick: bool,
        pause_on_next_complete: bool,
        pause_hook: Option<&'static str>,
        pause_tick: Option<u64>,
        resumed_at: Option<Instant>,
        first_tick_delay_micros: Option<u128>,
        worker_snapshot: Arc<AtomicU64>,
        rendered_snapshot: u64,
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
                cpu_tick_work: Duration::ZERO,
                command_probe: 0,
                gui_input_count: 0,
                simulation_control: None,
                control_installs: 0,
                redraw_request: None,
                pause_on_next_tick: false,
                pause_on_next_complete: false,
                pause_hook: None,
                pause_tick: None,
                resumed_at: None,
                first_tick_delay_micros: None,
                worker_snapshot: Arc::new(AtomicU64::new(0)),
                rendered_snapshot: 0,
            }
        }
    }

    impl App for CaptureApp {
        type Camera = ();

        fn tick(&mut self, ctx: &mut TickContext<'_>) {
            self.tick = ctx.tick_id;
            if let Some(resumed_at) = self.resumed_at.take() {
                self.first_tick_delay_micros = Some(resumed_at.elapsed().as_micros());
            }
            if std::mem::take(&mut self.pause_on_next_tick) {
                self.simulation_control.as_ref().unwrap().pause();
                self.pause_hook = Some("tick");
                self.pause_tick = Some(self.tick);
            }
            if !self.cpu_tick_work.is_zero() {
                // Exercise CPU-only apps that take longer than the fixed dt.
                // Leave the encoder empty: its fast completion can arrive
                // during the next tick and wake the native event loop again.
                let started = Instant::now();
                let mut work = self.tick;
                while started.elapsed() < self.cpu_tick_work {
                    for _ in 0..256 {
                        work = std::hint::black_box(work)
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1);
                    }
                }
                std::hint::black_box(work);
                return;
            }
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        fn set_simulation_control(&mut self, control: SimulationControl) {
            self.control_installs += 1;
            self.simulation_control = Some(control);
        }

        fn set_redraw_request(&mut self, request: RedrawRequest) {
            self.redraw_request = Some(request);
        }

        fn after_complete(&mut self, _gpu: &GpuContext, token: &SubmissionToken) {
            if token.tick_id.is_some() && std::mem::take(&mut self.pause_on_next_complete) {
                self.simulation_control.as_ref().unwrap().pause();
                self.pause_hook = Some("after_complete");
                self.pause_tick = Some(self.tick);
            }
        }

        fn prepare_render(&mut self, _ctx: &mut RenderUpdateContext<'_, ()>) {
            self.rendered_snapshot = self.worker_snapshot.load(Ordering::Acquire);
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
            if !self.cpu_tick_work.is_zero() && ui.button("Input probe").clicked() {
                self.gui_input_count += 1;
            }
        }

        fn capture_targets(&self, registry: &mut CaptureRegistry) {
            registry.register_buffer("values", &self.values, 0, self.values.size());
        }

        fn agent_status(&self) -> Value {
            json!({"tick":self.tick,"last_values":self.last_values,
                "cpu_tick_micros":self.cpu_tick_work.as_micros(),
                "command_probe":self.command_probe,
                "gui_input_count":self.gui_input_count,
                "control_installs":self.control_installs,
                "simulation_paused":self.simulation_control.as_ref().is_some_and(SimulationControl::is_paused),
                "pause_hook":self.pause_hook,"pause_tick":self.pause_tick,
                "first_tick_delay_micros":self.first_tick_delay_micros,
                "worker_snapshot":self.worker_snapshot.load(Ordering::Acquire),
                "rendered_snapshot":self.rendered_snapshot})
        }

        fn on_agent_command(&mut self, payload: Value) -> Result<Value, String> {
            if let Some(micros) = payload.get("cpu_tick_micros") {
                let micros = micros
                    .as_u64()
                    .filter(|micros| *micros <= 100_000)
                    .ok_or("cpu_tick_micros must be between 0 and 100000")?;
                self.cpu_tick_work = Duration::from_micros(micros);
            }
            if let Some(probe) = payload.get("probe") {
                self.command_probe = probe.as_u64().ok_or("probe must be a u64")?;
            }
            if let Some(operation) = payload.get("simulation") {
                let operation = operation.as_str().ok_or("simulation must be a string")?;
                let control = self
                    .simulation_control
                    .as_ref()
                    .ok_or("simulation control was not installed")?;
                self.pause_on_next_tick = false;
                self.pause_on_next_complete = false;
                self.pause_hook = None;
                self.pause_tick = None;
                self.resumed_at = None;
                match operation {
                    "pause" => control.pause(),
                    "resume" | "pause_on_tick" | "pause_on_complete" | "pause_resume" => {
                        match operation {
                            "pause_on_tick" => self.pause_on_next_tick = true,
                            "pause_on_complete" => self.pause_on_next_complete = true,
                            "pause_resume" => control.pause(),
                            _ => {}
                        }
                        self.resumed_at = Some(Instant::now());
                        self.first_tick_delay_micros = None;
                        control.resume();
                    }
                    _ => return Err("unknown simulation operation".into()),
                }
            }
            if let Some(snapshot) = payload.get("worker_snapshot") {
                let revision = snapshot.as_u64().ok_or("worker_snapshot must be a u64")?;
                let snapshot = self.worker_snapshot.clone();
                let redraw = self
                    .redraw_request
                    .clone()
                    .ok_or("redraw request was not installed")?;
                std::thread::spawn(move || {
                    // Let the command's own invalidation settle first, then
                    // emulate a later CPU-worker result that only redraws.
                    std::thread::sleep(Duration::from_millis(150));
                    snapshot.store(revision, Ordering::Release);
                    redraw.request_redraw();
                });
            }
            Ok(self.agent_status())
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
    let mut presentation_policy = mikage::PresentationPolicy::Immediate;
    let mut max_in_flight_submissions = 8;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--fixed" => {
                policy = mikage::SimulationPolicy::fixed(std::time::Duration::from_millis(10));
            }
            "--manual" => policy = mikage::SimulationPolicy::Manual,
            "--vsync" => presentation_policy = mikage::PresentationPolicy::Vsync,
            "--tick-micros" => {
                let micros = args
                    .next()
                    .expect("--tick-micros requires a duration")
                    .parse::<u64>()
                    .expect("invalid tick duration");
                let dt = std::time::Duration::from_micros(micros);
                policy = mikage::SimulationPolicy::Fixed {
                    dt,
                    max_ticks_per_wake: 4,
                    max_backlog: dt.saturating_mul(4),
                };
            }
            "--submission-slots" => {
                max_in_flight_submissions = args
                    .next()
                    .expect("--submission-slots requires a count")
                    .parse()
                    .expect("invalid slot count");
            }
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
                    "agent_capture [--fixed|--manual|--tick-micros N] [--vsync] [--submission-slots N] [--msaa4] [--port PORT] [--connection-file PATH]"
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
            presentation_policy,
            max_in_flight_submissions,
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
