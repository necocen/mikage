//! Headless GPU integration tests.
//!
//! These tests require a GPU (or software adapter). If no adapter is available,
//! tests are skipped rather than failing.

use std::path::Path;

use image::{ImageBuffer, Rgba, RgbaImage};
use mikage::wgpu;
use mikage::{
    CubeMesh, DEPTH_FORMAT, IcoSphereMesh, InstanceData, InstanceRenderer, InstanceRendererConfig,
    PlaneMesh, QuadMesh2d, RegularPolygonMesh, SceneBinding, SceneUniform, ShaderProcessor,
    SolidRenderer, UniformBuffer, create_compute_pipeline, create_storage_buffer_init,
    storage_buffer_entry,
};
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Headless GPU setup
// ---------------------------------------------------------------------------

struct GpuTest {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// Render format used across all headless tests (no surface, so we pick a common sRGB format).
const RENDER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn setup_gpu() -> Option<GpuTest> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("test_device"),
        ..Default::default()
    }))
    .ok()?;
    Some(GpuTest { device, queue })
}

/// Run `body` if a GPU is available; skip otherwise.
macro_rules! gpu_test {
    ($body:expr) => {
        let Some(gpu) = setup_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        $body(gpu);
    };
}

// ---------------------------------------------------------------------------
// Helpers: create textures and read back pixels
// ---------------------------------------------------------------------------

fn create_color_texture(
    device: &wgpu::Device,
    w: u32,
    h: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_color"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: RENDER_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}

fn create_depth_texture(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_depth"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&Default::default())
}

/// Read back RGBA pixels from a texture. Returns a Vec<u8> of length w*h*4.
fn readback_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    w: u32,
    h: u32,
) -> Vec<u8> {
    let bytes_per_row = align_to(w * 4, 256);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (bytes_per_row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .ok();
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    // Remove row padding
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for row in 0..h {
        let start = (row * bytes_per_row) as usize;
        let end = start + (w * 4) as usize;
        pixels.extend_from_slice(&data[start..end]);
    }
    pixels
}

fn align_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

// ===========================================================================
// 1. Shader compilation tests
// ===========================================================================

#[test]
fn builtin_shaders_compile() {
    gpu_test!(|gpu: GpuTest| {
        let sp = ShaderProcessor::new();

        // Resolve and compile each built-in shader
        let instancing_src = include_str!("../assets/shaders/instancing.wgsl");
        let solid_src = include_str!("../assets/shaders/solid.wgsl");

        for (name, src) in [("instancing", instancing_src), ("solid", solid_src)] {
            let resolved = sp.resolve(src).unwrap_or_else(|e| {
                panic!("Failed to resolve {name}: {e}");
            });
            // This will panic if the WGSL is invalid
            let _ = gpu
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(resolved.into()),
                });
        }
    });
}

#[test]
fn utility_shaders_compile_standalone() {
    gpu_test!(|gpu: GpuTest| {
        let sp = ShaderProcessor::new();

        // The math and color_utils modules should compile as standalone modules
        // (after import resolution)
        let math_src =
            "#import mikage::math\n@compute @workgroup_size(1) fn main() { let x = PI; }";
        let color_src = "#import mikage::color_utils\n@compute @workgroup_size(1) fn main() { let c = hsv2rgb(0.0, 1.0, 1.0); }";

        for (name, src) in [("math", math_src), ("color_utils", color_src)] {
            let resolved = sp.resolve(src).unwrap_or_else(|e| {
                panic!("Failed to resolve {name}: {e}");
            });
            let _ = gpu
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(resolved.into()),
                });
        }
    });
}

// ===========================================================================
// 2. Buffer and pipeline creation tests
// ===========================================================================

#[test]
fn scene_binding_creation() {
    gpu_test!(|gpu: GpuTest| {
        let scene = SceneBinding::new(&gpu.device);

        // Update with a real uniform — should not panic
        let uniform = SceneUniform::new(glam::Mat4::IDENTITY, glam::Vec3::ZERO);
        scene.update(&gpu.queue, &uniform);
    });
}

#[test]
fn uniform_buffer_write() {
    gpu_test!(|gpu: GpuTest| {
        let uniform = SceneUniform::new(glam::Mat4::IDENTITY, glam::Vec3::ZERO);
        let buf = UniformBuffer::new(&gpu.device, "test_uniform", &uniform);

        let updated = SceneUniform::with_light(
            glam::Mat4::IDENTITY,
            glam::Vec3::new(1.0, 2.0, 3.0),
            glam::Vec3::Y,
            0.5,
        );
        buf.write(&gpu.queue, &updated);
        // No panic = success. The buffer was created and written to.
    });
}

#[test]
fn storage_buffer_creation() {
    gpu_test!(|gpu: GpuTest| {
        let data: Vec<[f32; 4]> = vec![[1.0, 0.0, 0.0, 1.0]; 100];
        let _buf = create_storage_buffer_init(&gpu.device, "test_storage", &data);
    });
}

#[test]
fn instance_renderer_2d_creation() {
    gpu_test!(|gpu: GpuTest| {
        let scene = SceneBinding::new(&gpu.device);
        let quad = QuadMesh2d::generate();
        let _renderer = InstanceRenderer::new(
            &gpu.device,
            RENDER_FORMAT,
            scene.layout(),
            &quad.positions,
            &quad.normals,
            &quad.indices,
            InstanceRendererConfig::default_2d(),
        );
    });
}

#[test]
fn instance_renderer_3d_creation() {
    gpu_test!(|gpu: GpuTest| {
        let scene = SceneBinding::new(&gpu.device);
        let sphere = IcoSphereMesh::generate(1);
        let _renderer = InstanceRenderer::new(
            &gpu.device,
            RENDER_FORMAT,
            scene.layout(),
            &sphere.positions,
            &sphere.normals,
            &sphere.indices,
            InstanceRendererConfig::default_3d(),
        );
    });
}

#[test]
fn solid_renderer_creation() {
    gpu_test!(|gpu: GpuTest| {
        let scene = SceneBinding::new(&gpu.device);
        let _solid = SolidRenderer::new(&gpu.device, RENDER_FORMAT, scene.layout());
    });
}

#[test]
fn solid_renderer_add_and_update_object() {
    gpu_test!(|gpu: GpuTest| {
        let scene = SceneBinding::new(&gpu.device);
        let mut solid = SolidRenderer::new(&gpu.device, RENDER_FORMAT, scene.layout());
        let cube = CubeMesh::generate();
        let id = solid.add_object(&gpu.device, &cube.positions, &cube.normals, &cube.indices);
        solid.update_object(
            &gpu.queue,
            id,
            glam::Mat4::from_scale(glam::Vec3::splat(2.0)),
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
        );
    });
}

// ===========================================================================
// 3. Render-to-texture tests (pixel verification)
// ===========================================================================

#[test]
fn clear_to_color() {
    gpu_test!(|gpu: GpuTest| {
        let (tex, view) = create_color_texture(&gpu.device, 64, 64);

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
        }
        gpu.queue.submit([encoder.finish()]);

        let pixels = readback_texture(&gpu.device, &gpu.queue, &tex, 64, 64);
        // Center pixel should be red (sRGB encoded: linear 1.0 → sRGB 255)
        let idx = (32 * 64 + 32) * 4;
        assert_eq!(pixels[idx], 255, "R");
        assert_eq!(pixels[idx + 1], 0, "G");
        assert_eq!(pixels[idx + 2], 0, "B");
        assert_eq!(pixels[idx + 3], 255, "A");
    });
}

#[test]
fn instance_renderer_draws_pixels() {
    gpu_test!(|gpu: GpuTest| {
        let w = 64u32;
        let h = 64u32;
        let scene = SceneBinding::new(&gpu.device);

        // Use a quad that fills the screen in NDC
        // Identity view-projection: NDC coordinates map directly
        let uniform = SceneUniform::new(glam::Mat4::IDENTITY, glam::Vec3::ZERO);
        scene.update(&gpu.queue, &uniform);

        // Create a quad from -1 to 1 (fills entire viewport in NDC with identity VP)
        let positions: Vec<[f32; 3]> = vec![
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 1.0]; 4];
        let indices: Vec<u32> = vec![0, 1, 2, 0, 2, 3];

        let mut renderer = InstanceRenderer::new(
            &gpu.device,
            RENDER_FORMAT,
            scene.layout(),
            &positions,
            &normals,
            &indices,
            InstanceRendererConfig::default_2d(),
        );

        // One green instance at origin, scale 1
        let instances = vec![InstanceData {
            pos_scale: [0.0, 0.0, 0.0, 1.0],
            color: [0.0, 1.0, 0.0, 1.0],
        }];
        renderer.update_instances(&gpu.device, &gpu.queue, &instances);

        let (tex, color_view) = create_color_texture(&gpu.device, w, h);

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("instance_render"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_bind_group(0, scene.bind_group(), &[]);
            renderer.render(&mut pass);
        }
        gpu.queue.submit([encoder.finish()]);

        let pixels = readback_texture(&gpu.device, &gpu.queue, &tex, w, h);

        // Center pixel should be green (not black)
        let idx = (h / 2 * w + w / 2) as usize * 4;
        assert!(
            pixels[idx + 1] > 0,
            "center pixel G channel should be non-zero, got {:?}",
            &pixels[idx..idx + 4]
        );
        assert_eq!(
            pixels[idx], 0,
            "center pixel R should be 0, got {}",
            pixels[idx]
        );
    });
}

#[test]
fn solid_renderer_draws_pixels() {
    gpu_test!(|gpu: GpuTest| {
        let w = 64u32;
        let h = 64u32;
        let scene = SceneBinding::new(&gpu.device);

        let uniform = SceneUniform::new(glam::Mat4::IDENTITY, glam::Vec3::ZERO);
        scene.update(&gpu.queue, &uniform);

        let mut solid = SolidRenderer::new(&gpu.device, RENDER_FORMAT, scene.layout());

        // Full-screen quad facing camera
        let positions: Vec<[f32; 3]> = vec![
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 1.0]; 4];
        let indices: Vec<u32> = vec![0, 1, 2, 0, 2, 3];

        let id = solid.add_object(&gpu.device, &positions, &normals, &indices);
        // Transparent (alpha < 1.0) so it uses the unlit pipeline (direct color)
        solid.update_object(
            &gpu.queue,
            id,
            glam::Mat4::IDENTITY,
            glam::Vec4::new(0.0, 0.0, 1.0, 0.5),
        );

        let (tex, color_view) = create_color_texture(&gpu.device, w, h);
        let depth_view = create_depth_texture(&gpu.device, w, h);

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("solid_render"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_bind_group(0, scene.bind_group(), &[]);
            solid.render(&mut pass);
        }
        gpu.queue.submit([encoder.finish()]);

        let pixels = readback_texture(&gpu.device, &gpu.queue, &tex, w, h);

        // Center pixel should have blue (not pure black)
        let idx = (h / 2 * w + w / 2) as usize * 4;
        assert!(
            pixels[idx + 2] > 0,
            "center pixel B channel should be non-zero, got {:?}",
            &pixels[idx..idx + 4]
        );
    });
}

#[test]
fn instance_renderer_update_and_regrow() {
    gpu_test!(|gpu: GpuTest| {
        let scene = SceneBinding::new(&gpu.device);
        let hex = RegularPolygonMesh::generate(6);
        let mut renderer = InstanceRenderer::new(
            &gpu.device,
            RENDER_FORMAT,
            scene.layout(),
            &hex.positions,
            &hex.normals,
            &hex.indices,
            InstanceRendererConfig::default_2d(),
        );

        // Start with a small batch
        let small: Vec<InstanceData> = (0..10)
            .map(|i| InstanceData {
                pos_scale: [i as f32 * 0.1, 0.0, 0.0, 0.1],
                color: [1.0, 1.0, 1.0, 1.0],
            })
            .collect();
        renderer.update_instances(&gpu.device, &gpu.queue, &small);

        // Grow past initial capacity (1024)
        let large: Vec<InstanceData> = (0..2000)
            .map(|i| InstanceData {
                pos_scale: [i as f32 * 0.001, 0.0, 0.0, 0.01],
                color: [1.0, 0.0, 0.0, 1.0],
            })
            .collect();
        renderer.update_instances(&gpu.device, &gpu.queue, &large);
        assert!(renderer.instance_capacity() >= 2000);
    });
}

// ===========================================================================
// 4. Snapshot image comparison
// ===========================================================================

const SNAPSHOT_DIR: &str = "tests/snapshots";

/// Compare rendered pixels against a reference PNG snapshot.
///
/// - `channel_tolerance`: max allowed difference per RGBA channel (e.g., 2)
/// - `min_match_percent`: minimum percentage of matching pixels (e.g., 99.0)
///
/// If the reference file doesn't exist, the current render is saved as the
/// new reference and the test passes (first-run behavior).
/// On mismatch, saves `<name>_actual.png` next to the reference for debugging.
fn assert_snapshot(
    name: &str,
    pixels: &[u8],
    w: u32,
    h: u32,
    channel_tolerance: u8,
    min_match_percent: f64,
) {
    let snapshot_dir = Path::new(SNAPSHOT_DIR);
    std::fs::create_dir_all(snapshot_dir).expect("failed to create snapshot dir");

    let ref_path = snapshot_dir.join(format!("{name}.png"));
    let actual_img: RgbaImage =
        ImageBuffer::from_raw(w, h, pixels.to_vec()).expect("pixel data size mismatch");

    if !ref_path.exists() {
        actual_img
            .save(&ref_path)
            .expect("failed to save reference");
        eprintln!(
            "snapshot: created new reference '{}' ({}x{})",
            ref_path.display(),
            w,
            h
        );
        return;
    }

    let ref_img = image::open(&ref_path)
        .unwrap_or_else(|e| panic!("failed to load reference '{}': {e}", ref_path.display()))
        .to_rgba8();

    assert_eq!(
        (ref_img.width(), ref_img.height()),
        (w, h),
        "snapshot '{}': size mismatch (reference {}x{}, actual {}x{})",
        name,
        ref_img.width(),
        ref_img.height(),
        w,
        h,
    );

    let total_pixels = (w * h) as usize;
    let mut matching = 0usize;
    for (actual_px, ref_px) in actual_img.pixels().zip(ref_img.pixels()) {
        let ok = actual_px
            .0
            .iter()
            .zip(ref_px.0.iter())
            .all(|(&a, &r)| a.abs_diff(r) <= channel_tolerance);
        if ok {
            matching += 1;
        }
    }

    let match_percent = matching as f64 / total_pixels as f64 * 100.0;

    if match_percent < min_match_percent {
        let actual_path = snapshot_dir.join(format!("{name}_actual.png"));
        actual_img
            .save(&actual_path)
            .expect("failed to save actual image");

        // Also save a diff image for debugging
        let mut diff_img = RgbaImage::new(w, h);
        for (x, y, diff_px) in diff_img.enumerate_pixels_mut() {
            let a = actual_img.get_pixel(x, y);
            let r = ref_img.get_pixel(x, y);
            let ok =
                a.0.iter()
                    .zip(r.0.iter())
                    .all(|(&av, &rv)| av.abs_diff(rv) <= channel_tolerance);
            *diff_px = if ok {
                Rgba([0, 0, 0, 255])
            } else {
                Rgba([255, 0, 0, 255])
            };
        }
        let diff_path = snapshot_dir.join(format!("{name}_diff.png"));
        diff_img
            .save(&diff_path)
            .expect("failed to save diff image");

        panic!(
            "snapshot '{}': match {:.1}% < {:.1}% threshold ({} / {} pixels)\n  \
             reference: {}\n  actual:    {}\n  diff:      {}",
            name,
            match_percent,
            min_match_percent,
            matching,
            total_pixels,
            ref_path.display(),
            actual_path.display(),
            diff_path.display(),
        );
    }
}

// ---------------------------------------------------------------------------
// Render helpers for snapshot tests
// ---------------------------------------------------------------------------

/// Render with SolidRenderer and return pixels.
fn render_solid(
    gpu: &GpuTest,
    w: u32,
    h: u32,
    setup: impl FnOnce(&mut SolidRenderer, &SceneBinding, &wgpu::Device, &wgpu::Queue),
) -> Vec<u8> {
    let scene = SceneBinding::new(&gpu.device);
    let mut solid = SolidRenderer::new(&gpu.device, RENDER_FORMAT, scene.layout());
    setup(&mut solid, &scene, &gpu.device, &gpu.queue);

    let (tex, color_view) = create_color_texture(&gpu.device, w, h);
    let depth_view = create_depth_texture(&gpu.device, w, h);

    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("solid_snapshot"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_bind_group(0, scene.bind_group(), &[]);
        solid.render(&mut pass);
    }
    gpu.queue.submit([encoder.finish()]);
    readback_texture(&gpu.device, &gpu.queue, &tex, w, h)
}

/// Render with InstanceRenderer (2D, no depth) and return pixels.
fn render_instances_2d(
    gpu: &GpuTest,
    w: u32,
    h: u32,
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    indices: &[u32],
    instances: &[InstanceData],
    scene_uniform: &SceneUniform,
) -> Vec<u8> {
    let scene = SceneBinding::new(&gpu.device);
    scene.update(&gpu.queue, scene_uniform);

    let mut renderer = InstanceRenderer::new(
        &gpu.device,
        RENDER_FORMAT,
        scene.layout(),
        positions,
        normals,
        indices,
        InstanceRendererConfig::default_2d(),
    );
    renderer.update_instances(&gpu.device, &gpu.queue, instances);

    let (tex, color_view) = create_color_texture(&gpu.device, w, h);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("instance_snapshot"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_bind_group(0, scene.bind_group(), &[]);
        renderer.render(&mut pass);
    }
    gpu.queue.submit([encoder.finish()]);
    readback_texture(&gpu.device, &gpu.queue, &tex, w, h)
}

// ===========================================================================
// 5. Mesh snapshot tests
// ===========================================================================

const SNAP_SIZE: u32 = 128;
const SNAP_TOLERANCE: u8 = 3;
const SNAP_MATCH: f64 = 98.0;

#[test]
fn snapshot_solid_cube() {
    gpu_test!(|gpu: GpuTest| {
        let pixels = render_solid(&gpu, SNAP_SIZE, SNAP_SIZE, |solid, scene, device, queue| {
            // Perspective camera looking at origin
            let view = glam::Mat4::look_at_rh(
                glam::Vec3::new(2.0, 2.0, 2.0),
                glam::Vec3::ZERO,
                glam::Vec3::Y,
            );
            let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
            let uniform = SceneUniform::new(proj * view, glam::Vec3::new(2.0, 2.0, 2.0));
            scene.update(queue, &uniform);

            let cube = CubeMesh::generate();
            let id = solid.add_object(device, &cube.positions, &cube.normals, &cube.indices);
            solid.update_object(
                queue,
                id,
                glam::Mat4::IDENTITY,
                glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            );
        });
        assert_snapshot(
            "solid_cube",
            &pixels,
            SNAP_SIZE,
            SNAP_SIZE,
            SNAP_TOLERANCE,
            SNAP_MATCH,
        );
    });
}

#[test]
fn snapshot_solid_sphere() {
    gpu_test!(|gpu: GpuTest| {
        let pixels = render_solid(&gpu, SNAP_SIZE, SNAP_SIZE, |solid, scene, device, queue| {
            let view = glam::Mat4::look_at_rh(
                glam::Vec3::new(0.0, 0.0, 3.0),
                glam::Vec3::ZERO,
                glam::Vec3::Y,
            );
            let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
            let uniform = SceneUniform::new(proj * view, glam::Vec3::new(0.0, 0.0, 3.0));
            scene.update(queue, &uniform);

            let sphere = IcoSphereMesh::generate(2);
            let id = solid.add_object(device, &sphere.positions, &sphere.normals, &sphere.indices);
            solid.update_object(
                queue,
                id,
                glam::Mat4::IDENTITY,
                glam::Vec4::new(0.2, 0.6, 1.0, 1.0),
            );
        });
        assert_snapshot(
            "solid_sphere",
            &pixels,
            SNAP_SIZE,
            SNAP_SIZE,
            SNAP_TOLERANCE,
            SNAP_MATCH,
        );
    });
}

#[test]
fn snapshot_solid_plane() {
    gpu_test!(|gpu: GpuTest| {
        let pixels = render_solid(&gpu, SNAP_SIZE, SNAP_SIZE, |solid, scene, device, queue| {
            let view = glam::Mat4::look_at_rh(
                glam::Vec3::new(0.0, 2.0, 1.0),
                glam::Vec3::ZERO,
                glam::Vec3::Y,
            );
            let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
            let uniform = SceneUniform::new(proj * view, glam::Vec3::new(0.0, 2.0, 1.0));
            scene.update(queue, &uniform);

            let plane = PlaneMesh::generate();
            let id = solid.add_object(device, &plane.positions, &plane.normals, &plane.indices);
            solid.update_object(
                queue,
                id,
                glam::Mat4::from_scale(glam::Vec3::splat(3.0)),
                glam::Vec4::new(0.0, 0.8, 0.2, 1.0),
            );
        });
        assert_snapshot(
            "solid_plane",
            &pixels,
            SNAP_SIZE,
            SNAP_SIZE,
            SNAP_TOLERANCE,
            SNAP_MATCH,
        );
    });
}

#[test]
fn snapshot_instanced_hexagons() {
    gpu_test!(|gpu: GpuTest| {
        let hex = RegularPolygonMesh::generate(6);
        let uniform = SceneUniform::new(glam::Mat4::IDENTITY, glam::Vec3::ZERO);

        let instances = vec![
            InstanceData {
                pos_scale: [-0.5, -0.5, 0.0, 0.4],
                color: [1.0, 0.0, 0.0, 1.0],
            },
            InstanceData {
                pos_scale: [0.5, -0.5, 0.0, 0.4],
                color: [0.0, 1.0, 0.0, 1.0],
            },
            InstanceData {
                pos_scale: [0.0, 0.5, 0.0, 0.4],
                color: [0.0, 0.0, 1.0, 1.0],
            },
        ];

        let pixels = render_instances_2d(
            &gpu,
            SNAP_SIZE,
            SNAP_SIZE,
            &hex.positions,
            &hex.normals,
            &hex.indices,
            &instances,
            &uniform,
        );
        assert_snapshot(
            "instanced_hexagons",
            &pixels,
            SNAP_SIZE,
            SNAP_SIZE,
            SNAP_TOLERANCE,
            SNAP_MATCH,
        );
    });
}

#[test]
fn snapshot_instanced_quad_grid() {
    gpu_test!(|gpu: GpuTest| {
        let quad = QuadMesh2d::generate();
        let uniform = SceneUniform::new(glam::Mat4::IDENTITY, glam::Vec3::ZERO);

        // 3x3 grid of colored quads
        let mut instances = Vec::new();
        for row in 0..3 {
            for col in 0..3 {
                let x = (col as f32 - 1.0) * 0.6;
                let y = (row as f32 - 1.0) * 0.6;
                let r = col as f32 / 2.0;
                let g = row as f32 / 2.0;
                instances.push(InstanceData {
                    pos_scale: [x, y, 0.0, 0.25],
                    color: [r, g, 0.5, 1.0],
                });
            }
        }

        let pixels = render_instances_2d(
            &gpu,
            SNAP_SIZE,
            SNAP_SIZE,
            &quad.positions,
            &quad.normals,
            &quad.indices,
            &instances,
            &uniform,
        );
        assert_snapshot(
            "instanced_quad_grid",
            &pixels,
            SNAP_SIZE,
            SNAP_SIZE,
            SNAP_TOLERANCE,
            SNAP_MATCH,
        );
    });
}

#[test]
fn snapshot_solid_transparent() {
    gpu_test!(|gpu: GpuTest| {
        let pixels = render_solid(&gpu, SNAP_SIZE, SNAP_SIZE, |solid, scene, device, queue| {
            let uniform = SceneUniform::new(glam::Mat4::IDENTITY, glam::Vec3::ZERO);
            scene.update(queue, &uniform);

            // Full-screen quad
            let positions: Vec<[f32; 3]> = vec![
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ];
            let normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 1.0]; 4];
            let indices: Vec<u32> = vec![0, 1, 2, 0, 2, 3];

            let id = solid.add_object(device, &positions, &normals, &indices);
            // Semi-transparent blue (uses unlit pipeline)
            solid.update_object(
                queue,
                id,
                glam::Mat4::IDENTITY,
                glam::Vec4::new(0.0, 0.5, 1.0, 0.5),
            );
        });
        assert_snapshot(
            "solid_transparent",
            &pixels,
            SNAP_SIZE,
            SNAP_SIZE,
            SNAP_TOLERANCE,
            SNAP_MATCH,
        );
    });
}

// ===========================================================================
// 6. Compute shader integration test
// ===========================================================================

#[test]
fn compute_double_values() {
    gpu_test!(|gpu: GpuTest| {
        let sp = ShaderProcessor::new();
        let shader_src = sp
            .resolve(
                r#"
#import mikage::math

@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i < arrayLength(&data) {
        data[i] = data[i] * 2.0;
    }
}
"#,
            )
            .unwrap();

        let n = 256u32;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();

        let data_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("data"),
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[storage_buffer_entry(0, wgpu::ShaderStages::COMPUTE, false)],
            });

        let pipeline = create_compute_pipeline(&gpu.device, "double", &shader_src, &[&bgl], "main");

        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: data_buf.as_entire_binding(),
            }],
        });

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
        }

        // Copy result to a staging buffer for readback
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (n as u64) * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&data_buf, 0, &staging, 0, (n as u64) * 4);
        gpu.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
        rx.recv().unwrap().unwrap();

        let result: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        for (i, &val) in result.iter().enumerate() {
            let expected = i as f32 * 2.0;
            assert!(
                (val - expected).abs() < 1e-5,
                "data[{i}] = {val}, expected {expected}"
            );
        }
    });
}

// ===========================================================================
// 7. HSV color utility test
// ===========================================================================

/// CPU reference implementation of hsv2rgb matching the WGSL shader logic.
fn hsv2rgb_reference(h: f32, s: f32, v: f32) -> [f32; 3] {
    let pi = std::f32::consts::PI;
    let c = v * s;
    let hp = h * (3.0 / pi);
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = if hp < 1.0 {
        (c, x, 0.0)
    } else if hp < 2.0 {
        (x, c, 0.0)
    } else if hp < 3.0 {
        (0.0, c, x)
    } else if hp < 4.0 {
        (0.0, x, c)
    } else if hp < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    [r + m, g + m, b + m]
}

/// Convert linear [0,1] to sRGB [0,255] (matches GPU's Rgba8UnormSrgb write).
fn linear_to_srgb_u8(c: f32) -> u8 {
    let s = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[test]
fn snapshot_hsv_gradient() {
    gpu_test!(|gpu: GpuTest| {
        let w = 128u32;
        let h = 64u32;

        let sp = ShaderProcessor::new();
        let shader_src = sp
            .resolve(
                r#"
#import mikage::color_utils

// Output: RGBA per pixel, stored as 4 x f32 per pixel
@group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;

struct Params {
    width: u32,
    height: u32,
}
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let idx = id.y * params.width + id.x;
    let u = f32(id.x) / f32(params.width - 1u);   // 0..1 → hue
    let v = f32(id.y) / f32(params.height - 1u);   // 0..1 → saturation
    let h = u * 2.0 * PI;                           // hue in radians
    let rgb = hsv2rgb(h, v, 1.0);                   // value = 1.0
    pixels[idx] = vec4<f32>(rgb, 1.0);
}
"#,
            )
            .unwrap();

        let pixel_count = (w * h) as u64;
        let buf_size = pixel_count * 16; // 4 x f32 per pixel

        let pixel_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hsv_pixels"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            width: u32,
            height: u32,
        }
        let params_buf = UniformBuffer::new(
            &gpu.device,
            "params",
            &Params {
                width: w,
                height: h,
            },
        );

        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    storage_buffer_entry(0, wgpu::ShaderStages::COMPUTE, false),
                    mikage::uniform_buffer_entry(1, wgpu::ShaderStages::COMPUTE),
                ],
            });

        let pipeline =
            create_compute_pipeline(&gpu.device, "hsv_test", &shader_src, &[&bgl], "main");

        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pixel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.buffer().as_entire_binding(),
                },
            ],
        });

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }

        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&pixel_buf, 0, &staging, 0, buf_size);
        gpu.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
        rx.recv().unwrap().unwrap();

        let gpu_data: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();

        // Compare GPU results against CPU reference + save as snapshot image
        let mut rgba_pixels = Vec::with_capacity((w * h * 4) as usize);
        let mut max_diff: f32 = 0.0;
        let mut mismatches = 0u32;

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let gpu_r = gpu_data[idx * 4];
                let gpu_g = gpu_data[idx * 4 + 1];
                let gpu_b = gpu_data[idx * 4 + 2];

                let u = x as f32 / (w - 1) as f32;
                let v = y as f32 / (h - 1) as f32;
                let hue = u * 2.0 * std::f32::consts::PI;
                let [exp_r, exp_g, exp_b] = hsv2rgb_reference(hue, v, 1.0);

                let dr = (gpu_r - exp_r).abs();
                let dg = (gpu_g - exp_g).abs();
                let db = (gpu_b - exp_b).abs();
                let pixel_diff = dr.max(dg).max(db);
                max_diff = max_diff.max(pixel_diff);

                if pixel_diff > 0.01 {
                    mismatches += 1;
                }

                // Convert linear → sRGB u8 for the snapshot image
                rgba_pixels.push(linear_to_srgb_u8(gpu_r));
                rgba_pixels.push(linear_to_srgb_u8(gpu_g));
                rgba_pixels.push(linear_to_srgb_u8(gpu_b));
                rgba_pixels.push(255);
            }
        }

        // Save as snapshot for visual inspection
        assert_snapshot(
            "hsv_gradient",
            &rgba_pixels,
            w,
            h,
            SNAP_TOLERANCE,
            SNAP_MATCH,
        );

        // Verify GPU matches CPU reference
        let total = w * h;
        let match_pct = (total - mismatches) as f64 / total as f64 * 100.0;
        assert!(
            match_pct >= 99.0,
            "HSV mismatch: {mismatches}/{total} pixels differ > 0.01 (max_diff={max_diff:.6}), match={match_pct:.1}%"
        );
    });
}

/// Test that hsv2rgb produces continuous output for angles spanning [-PI, PI],
/// i.e. the full atan2 range. This catches the bug where negative hue values
/// cause discontinuous color jumps.
#[test]
fn compute_hsv_negative_angle_continuity() {
    gpu_test!(|gpu: GpuTest| {
        let sp = ShaderProcessor::new();
        let shader_src = sp
            .resolve(
                r#"
#import mikage::color_utils
#import mikage::math

struct Result {
    r: f32,
    g: f32,
    b: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read_write> results: array<Result>;

struct Params { count: u32 }
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.count { return; }

    // Sweep angle from -PI to PI (simulating atan2 range)
    let t = f32(i) / f32(params.count - 1u);
    let angle = mix(-PI, PI, t);

    // Apply the same wrapping fix as the boids shader
    let hue = (angle + TAU) % TAU;
    let rgb = hsv2rgb(hue, 0.6, 1.0);
    results[i] = Result(rgb.x, rgb.y, rgb.z, 0.0);
}
"#,
            )
            .unwrap();

        let n = 512u32;

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            count: u32,
        }

        let result_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("results"),
            size: (n as u64) * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params_buf = UniformBuffer::new(&gpu.device, "params", &Params { count: n });

        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    storage_buffer_entry(0, wgpu::ShaderStages::COMPUTE, false),
                    mikage::uniform_buffer_entry(1, wgpu::ShaderStages::COMPUTE),
                ],
            });

        let pipeline =
            create_compute_pipeline(&gpu.device, "hsv_continuity", &shader_src, &[&bgl], "main");

        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: result_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.buffer().as_entire_binding(),
                },
            ],
        });

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
        }
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (n as u64) * 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&result_buf, 0, &staging, 0, (n as u64) * 16);
        gpu.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
        rx.recv().unwrap().unwrap();

        let data: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();

        // Check continuity: adjacent samples should not jump too much
        let pi = std::f32::consts::PI;
        let mut max_jump: f32 = 0.0;
        let mut worst_i = 0;

        for i in 1..n as usize {
            let prev_r = data[(i - 1) * 4];
            let prev_g = data[(i - 1) * 4 + 1];
            let prev_b = data[(i - 1) * 4 + 2];
            let cur_r = data[i * 4];
            let cur_g = data[i * 4 + 1];
            let cur_b = data[i * 4 + 2];

            let dr = (cur_r - prev_r).abs();
            let dg = (cur_g - prev_g).abs();
            let db = (cur_b - prev_b).abs();
            let jump = dr.max(dg).max(db);

            if jump > max_jump {
                max_jump = jump;
                worst_i = i;
            }
        }

        // With 512 samples over [−π, π], the angle step is ~0.012 rad.
        // The maximum color change per step should be small.
        // A discontinuity (the old bug) would produce a jump > 0.5.
        let t = worst_i as f32 / (n - 1) as f32;
        let worst_angle = -pi + t * 2.0 * pi;
        assert!(
            max_jump < 0.05,
            "color discontinuity detected: max_jump={max_jump:.4} at sample {worst_i} \
             (angle={worst_angle:.4} rad). This suggests hsv2rgb is not handling \
             negative angles correctly."
        );

        // Also verify against CPU reference at each point
        for i in 0..n as usize {
            let t = i as f32 / (n - 1) as f32;
            let angle = -pi + t * 2.0 * pi;
            let hue = (angle + std::f32::consts::TAU) % std::f32::consts::TAU;
            let [exp_r, exp_g, exp_b] = hsv2rgb_reference(hue, 0.6, 1.0);

            let gpu_r = data[i * 4];
            let gpu_g = data[i * 4 + 1];
            let gpu_b = data[i * 4 + 2];

            let diff = (gpu_r - exp_r)
                .abs()
                .max((gpu_g - exp_g).abs())
                .max((gpu_b - exp_b).abs());
            assert!(
                diff < 0.01,
                "HSV mismatch at sample {i} (angle={angle:.4}): \
                 GPU=({gpu_r:.4},{gpu_g:.4},{gpu_b:.4}), \
                 expected=({exp_r:.4},{exp_g:.4},{exp_b:.4}), diff={diff:.6}"
            );
        }
    });
}
