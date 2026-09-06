//! Exercises the production app lifecycle without a window or event loop.
#![cfg(not(target_family = "wasm"))]

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};
use std::time::Duration;

use mikage::app::{App, RenderContext, RenderTarget, RenderUpdateContext, TickContext};
use mikage::context::{GpuContext, GpuDescriptor, OffscreenTarget, RenderTargetConfig};
use mikage::runtime::{AppRuntime, RuntimeConfig, RuntimeError, SubmissionToken};
use mikage::wgpu;
use wgpu::util::DeviceExt;

struct CounterApp {
    parameters: wgpu::Buffer,
    result: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    ticks: u64,
    prepared: u64,
    renders: u64,
    completions: Vec<u64>,
    owner: std::thread::ThreadId,
    shutdowns: Arc<AtomicUsize>,
}

impl CounterApp {
    fn new(gpu: &GpuContext, shutdowns: Arc<AtomicUsize>) -> Self {
        let parameters = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("runtime_test_parameters"),
                contents: &[0; 16],
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let result = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("runtime_test_counter"),
                contents: &[0; 4],
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("runtime_test_counter_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    "\
                struct Parameters { value: vec4<u32> };\n\
                @group(0) @binding(0) var<uniform> parameters: Parameters;\n\
                @group(0) @binding(1) var<storage, read_write> result: array<u32>;\n\
                @compute @workgroup_size(1) fn main() { result[0] += parameters.value.x; }\n"
                        .into(),
                ),
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("runtime_test_pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let bindings = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: parameters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: result.as_entire_binding(),
                },
            ],
        });
        Self {
            parameters,
            result,
            pipeline,
            bindings,
            ticks: 0,
            prepared: 0,
            renders: 0,
            completions: Vec::new(),
            owner: std::thread::current().id(),
            shutdowns,
        }
    }
}

impl App for CounterApp {
    type Camera = ();

    fn tick(&mut self, ctx: &mut TickContext<'_>) {
        self.ticks += 1;
        let parameters = [self.ticks as u32, 0, 0, 0];
        ctx.gpu
            .queue
            .write_buffer(&self.parameters, 0, bytemuck::cast_slice(&parameters));
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bindings, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    fn prepare_render(&mut self, _ctx: &mut RenderUpdateContext<'_, ()>) {
        self.prepared += 1;
    }

    fn render(&mut self, ctx: &mut RenderContext<'_, ()>) {
        self.renders += 1;
        let attachment = ctx.color_attachment(wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color {
                r: self.ticks as f64 / 10.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            }),
            store: wgpu::StoreOp::Store,
        });
        let _pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }

    fn after_complete(&mut self, _gpu: &GpuContext, token: &SubmissionToken) {
        assert_eq!(
            std::thread::current().id(),
            self.owner,
            "App hooks must stay on their owner thread"
        );
        self.completions.push(token.id);
    }

    fn shutdown(&mut self, _gpu: &GpuContext) {
        self.shutdowns.fetch_add(1, Ordering::SeqCst);
    }

    fn capture_targets(&self, registry: &mut mikage::CaptureRegistry) {
        registry.register_buffer("counter", &self.result, 0, 4);
    }
}

fn runtime() -> Option<(AppRuntime<CounterApp>, Arc<AtomicUsize>)> {
    let gpu = match pollster::block_on(GpuContext::headless(GpuDescriptor::default())) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skipping runtime GPU test: {error}");
            return None;
        }
    };
    let shutdowns = Arc::new(AtomicUsize::new(0));
    let app = CounterApp::new(&gpu, shutdowns.clone());
    Some((AppRuntime::new(gpu, app, ()), shutdowns))
}

fn counter_value(runtime: &mut AppRuntime<CounterApp>) -> u32 {
    let readback = runtime.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("runtime_test_readback"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let (sender, receiver) = mpsc::channel();
    let (_, token) = runtime
        .submit_command(|app, ctx| {
            ctx.encoder
                .copy_buffer_to_buffer(&app.result, 0, &readback, 0, 4);
            ctx.encoder
                .map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |result| {
                    let _ = sender.send(result);
                });
        })
        .unwrap();
    runtime
        .wait_for_timeout(&token, Some(Duration::from_secs(10)))
        .unwrap();
    receiver
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let mapped = readback.get_mapped_range(..).unwrap();
    let value = u32::from_le_bytes(mapped[..4].try_into().unwrap());
    drop(mapped);
    readback.unmap();
    value
}

#[test]
fn exact_ticks_preserve_per_tick_uploads_and_completion_identity() {
    let Some((mut runtime, shutdowns)) = runtime() else {
        return;
    };
    runtime
        .set_config(RuntimeConfig {
            max_in_flight_submissions: 2,
        })
        .unwrap();
    let wakes = Arc::new(AtomicUsize::new(0));
    let wake_count = wakes.clone();
    runtime.set_waker(Arc::new(move || {
        wake_count.fetch_add(1, Ordering::SeqCst);
    }));

    assert!(
        runtime
            .advance_ticks(0, Duration::from_millis(10))
            .unwrap()
            .is_none()
    );
    assert_eq!(runtime.progress().submitted_submissions, 0);
    let endpoint = runtime
        .advance_ticks(7, Duration::from_millis(10))
        .unwrap()
        .unwrap();
    runtime
        .wait_for_timeout(&endpoint, Some(Duration::from_secs(10)))
        .unwrap();
    assert_eq!(runtime.app.ticks, 7);
    assert_eq!(runtime.progress().completed_ticks, 7);
    assert_eq!(runtime.progress().submitted_frames, 0);
    assert_eq!(runtime.progress().presented_frames, 0);
    assert!((runtime.progress().elapsed - 0.07).abs() < 1e-12);
    // A single batched submission with repeated uniform uploads would yield 49.
    assert_eq!(counter_value(&mut runtime), 28);
    assert_eq!(runtime.progress().completed_ticks, 7);
    assert_eq!(runtime.app.completions, (1..=8).collect::<Vec<_>>());
    assert!(wakes.load(Ordering::SeqCst) > 0);
    runtime.shutdown();
    runtime.shutdown();
    drop(runtime);
    assert_eq!(shutdowns.load(Ordering::SeqCst), 1);
}

#[test]
fn zero_tick_offscreen_render_and_diagnostics_do_not_advance_simulation() {
    let Some((mut runtime, _)) = runtime() else {
        return;
    };
    let endpoint = runtime
        .advance_ticks(1, Duration::from_millis(5))
        .unwrap()
        .unwrap();
    runtime
        .wait_for_timeout(&endpoint, Some(Duration::from_secs(10)))
        .unwrap();
    let target = OffscreenTarget::new(
        &runtime.gpu,
        dpi::PhysicalSize::new(4, 4),
        RenderTargetConfig::default(),
    )
    .unwrap();
    let destination = RenderTarget {
        view: target.color_view(),
        resolve_target: target.resolve_target(),
        depth_view: Some(target.depth_view()),
        size: target.size(),
        config: target.render_target_config(),
    };
    for _ in 0..2 {
        let token = runtime.render(destination).unwrap();
        runtime
            .wait_for_timeout(&token, Some(Duration::from_secs(10)))
            .unwrap();
    }
    assert_eq!(runtime.app.ticks, 1);
    assert_eq!(runtime.app.prepared, 2);
    assert_eq!(runtime.app.renders, 2);
    assert_eq!(runtime.progress().completed_frames, 2);
    assert_eq!(runtime.progress().presented_frames, 0);
    assert_eq!(counter_value(&mut runtime), 1);

    let invalid = RenderTarget {
        size: dpi::PhysicalSize::new(0, 4),
        ..destination
    };
    assert!(matches!(
        runtime.render(invalid),
        Err(RuntimeError::InvalidTarget)
    ));
    assert_eq!(
        runtime.app.prepared, 2,
        "invalid target must not call the app"
    );
}

#[test]
fn headless_harness_captures_named_buffers_and_the_prior_rendered_scene() {
    let shutdowns = Arc::new(AtomicUsize::new(0));
    let mut harness = match pollster::block_on(mikage::HeadlessHarness::new(
        GpuDescriptor::default(),
        RenderTargetConfig::default(),
        dpi::PhysicalSize::new(4, 4),
        (),
        |gpu, _, _| CounterApp::new(gpu, shutdowns),
    )) {
        Ok(harness) => harness,
        Err(mikage::HeadlessError::Gpu(error)) => {
            eprintln!("skipping headless GPU test: {error}");
            return;
        }
        Err(error) => panic!("headless initialization failed: {error}"),
    };
    assert!(harness.offscreen_target().is_none());
    assert!(matches!(
        harness.capture_named("scene"),
        Err(mikage::HeadlessError::SceneNotRendered)
    ));
    let endpoint = harness
        .advance_ticks(4, Duration::from_millis(1))
        .unwrap()
        .unwrap();
    harness
        .runtime
        .wait_for_timeout(&endpoint, Some(Duration::from_secs(10)))
        .unwrap();
    let counter = harness.capture_named("counter").unwrap();
    assert_eq!(
        u32::from_le_bytes(counter.data.unwrap().try_into().unwrap()),
        10
    );
    assert_eq!(counter.metadata.tick_id, Some(4));
    assert_eq!(counter.metadata.frame_id, None);
    assert!(
        harness.offscreen_target().is_none(),
        "buffer capture must not allocate a render target"
    );

    let frame = harness.render_once().unwrap();
    harness.advance_ticks(1, Duration::from_millis(1)).unwrap();
    let scene = harness.capture_named("scene").unwrap();
    assert_eq!(scene.metadata.frame_id, frame.frame_id);
    assert_eq!(
        scene.metadata.tick_id,
        Some(4),
        "scene metadata must retain its rendered checkpoint"
    );
    assert_eq!(scene.data.unwrap().len(), 4 * 4 * 4);
    assert_eq!(harness.runtime.app.ticks, 5);
    assert_eq!(harness.runtime.app.renders, 1);
    assert_eq!(harness.runtime.progress().presented_frames, 0);
}

#[test]
fn destroyed_device_rejects_new_work_and_shutdown_remains_idempotent() {
    let Some((mut runtime, shutdowns)) = runtime() else {
        return;
    };
    runtime.gpu.device.destroy();
    let _ = runtime.gpu.device.poll(wgpu::PollType::Poll);
    assert!(matches!(
        runtime.poll_completions(),
        Err(RuntimeError::DeviceLost(_))
    ));
    assert!(matches!(
        runtime.try_tick(Duration::from_millis(1)),
        Err(RuntimeError::DeviceLost(_))
    ));
    assert_eq!(runtime.app.ticks, 0);
    runtime.shutdown();
    drop(runtime);
    assert_eq!(shutdowns.load(Ordering::SeqCst), 1);
}
