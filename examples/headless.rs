//! Run with: cargo run --no-default-features --example headless

#[cfg(not(target_family = "wasm"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use mikage::{
        App, CaptureRegistry, GpuDescriptor, HeadlessHarness, RenderContext, RenderTargetConfig,
        TickContext, wgpu,
    };
    use std::time::Duration;

    struct Counter {
        ticks: u32,
        buffer: wgpu::Buffer,
    }
    impl App for Counter {
        type Camera = ();
        fn tick(&mut self, ctx: &mut TickContext<'_>) {
            self.ticks += 1;
            ctx.gpu
                .queue
                .write_buffer(&self.buffer, 0, &self.ticks.to_le_bytes());
        }
        fn render(&mut self, ctx: &mut RenderContext<'_, ()>) {
            let color = ctx.color_attachment(wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.2,
                    g: 0.4,
                    b: 0.8,
                    a: 1.0,
                }),
                store: wgpu::StoreOp::Store,
            });
            let _pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("headless_scene"),
                color_attachments: &[Some(color)],
                ..Default::default()
            });
        }
        fn capture_targets(&self, registry: &mut CaptureRegistry) {
            registry.register_buffer("counter", &self.buffer, 0, 4);
        }
    }

    let mut harness = pollster::block_on(HeadlessHarness::new(
        GpuDescriptor::default(),
        RenderTargetConfig::default(),
        mikage::dpi::PhysicalSize::new(64, 64),
        (),
        |gpu, _target, _size| Counter {
            ticks: 0,
            buffer: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("counter"),
                size: 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
        },
    ))?;
    let endpoint = harness
        .advance_ticks(120, Duration::from_secs_f64(1.0 / 60.0))?
        .unwrap();
    harness.runtime.wait_for(&endpoint)?;
    let counter = harness.capture_named("counter")?;
    let ticks = u32::from_le_bytes(counter.data?.try_into().unwrap());
    assert_eq!(ticks, 120);

    harness.render_once()?;
    let scene = harness.capture_named("scene")?;
    println!(
        "Completed {ticks} ticks; captured {} scene bytes at frame {:?}",
        scene.data?.len(),
        scene.metadata.frame_id
    );
    Ok(())
}

#[cfg(target_family = "wasm")]
fn main() {}
