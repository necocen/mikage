//! Headless GPU integration tests.
//!
//! These tests require a GPU (or software adapter). If no adapter is available,
//! tests are skipped rather than failing.

use mikage::wgpu;
use mikage::{
    CubeMesh, DEPTH_FORMAT, IcoSphereMesh, InstanceData, InstanceRenderer, InstanceRendererConfig,
    QuadMesh2d, RegularPolygonMesh, SceneBinding, SceneUniform, ShaderProcessor, SolidRenderer,
    UniformBuffer, create_storage_buffer_init,
};

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
