//! An external consumer which owns its CAMetalLayer and frame loop.
//! No winit, egui, NSWindow, or EventLoop is used.

#[cfg(target_os = "macos")]
fn main() {
    objc2::rc::autoreleasepool(|_| run());
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("The native external-surface fixture requires macOS.");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn run() {
    use mikage::{GpuContext, GpuDescriptor, SurfaceDescriptor, dpi::PhysicalSize, wgpu};
    use objc2::rc::Retained;
    use objc2_core_foundation::{CGPoint, CGRect, CGSize};
    use objc2_quartz_core::CAMetalLayer;

    // This owner plays the role of ScreenSaverView's host-owned layer. It must
    // outlive the surface and all acquired frames, including error paths.
    let layer = CAMetalLayer::new();
    layer.setBounds(CGRect::new(CGPoint::new(0.0, 0.0), CGSize::new(64.0, 64.0)));
    layer.setDrawableSize(CGSize::new(64.0, 64.0));
    let raw_layer = Retained::as_ptr(&layer).cast_mut().cast();
    let descriptor = SurfaceDescriptor {
        size: PhysicalSize::new(64, 64),
        sample_count: 4,
        ..Default::default()
    };
    // SAFETY: layer stays retained until after surface and acquired frames drop.
    let (gpu, mut surface) = pollster::block_on(unsafe {
        GpuContext::for_surface_unsafe(
            wgpu::SurfaceTargetUnsafe::CoreAnimationLayer(raw_layer),
            GpuDescriptor {
                backends: Some(wgpu::Backends::METAL),
                ..Default::default()
            },
            descriptor,
        )
    })
    .expect("external layer GPU initialization");
    assert_eq!(gpu.adapter_info().backend, wgpu::Backend::Metal);
    render(&gpu, &surface);

    surface.resize(&gpu, PhysicalSize::new(0, 0)).unwrap();
    assert!(surface.is_suspended());
    assert!(surface.acquire_surface_texture().is_none());
    layer.setDrawableSize(CGSize::new(96.0, 48.0));
    surface.resize(&gpu, PhysicalSize::new(96, 48)).unwrap();
    assert!(!surface.is_suspended());
    render(&gpu, &surface);

    // Reattachment exercises the path used after Surface loss without replacing
    // the logical device. Release the previous surface before recreating it.
    let descriptor = surface.descriptor().clone();
    drop(surface);
    // SAFETY: the same retained host layer remains valid throughout this scope.
    let surface = unsafe {
        gpu.attach_surface_unsafe(
            wgpu::SurfaceTargetUnsafe::CoreAnimationLayer(raw_layer),
            descriptor,
        )
    }
    .expect("reattach external layer");
    render(&gpu, &surface);
    drop(surface);
    drop(gpu);
    drop(layer);
    println!("external CAMetalLayer: render, suspend, resize, reattach, teardown passed");
}

#[cfg(target_os = "macos")]
fn render(gpu: &mikage::GpuContext, surface: &mikage::SurfaceContext<'_>) {
    use mikage::wgpu;
    let texture = match surface
        .acquire_surface_texture()
        .expect("surface is active")
    {
        wgpu::CurrentSurfaceTexture::Success(texture)
        | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
        other => panic!("external surface acquisition failed: {other:?}"),
    };
    let view = surface.create_view(&texture);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("external_surface_fixture"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: surface.msaa_view().unwrap_or(&view),
                resolve_target: surface.msaa_view().map(|_| &view),
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.3,
                        b: 0.7,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
    }
    let submission = gpu.queue.submit([encoder.finish()]);
    gpu.queue.present(texture);
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(std::time::Duration::from_secs(10)),
        })
        .expect("external frame completion");
}
