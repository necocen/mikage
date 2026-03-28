use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

use crate::camera::Camera;

/// Creates a depth texture and its view.
///
/// Typically called in the factory closure and
/// [`App::resize`](crate::App::resize). Use [`DEPTH_FORMAT`] as the format.
/// The sample count is automatically matched to
/// [`GpuContext::sample_count`](crate::GpuContext::sample_count).
pub fn create_depth_texture(
    gpu: &crate::GpuContext,
    size: PhysicalSize<u32>,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_texture"),
        size: wgpu::Extent3d {
            width: size.width.max(1),
            height: size.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: gpu.sample_count(),
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Common depth format constant.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Scene-global uniform: camera matrices + basic lighting.
///
/// Intended for simple visualization use cases.
/// Contains view-projection matrix, camera position, directional light, and ambient.
///
/// Used as `@group(0) @binding(0)` by [`SolidRenderer`](crate::SolidRenderer)
/// and custom shaders. Create a bind group layout with a single uniform buffer
/// entry and share it across pipelines that use SceneUniform.
///
/// `#[repr(C)]` + `bytemuck::Pod`, so it can be written directly to a
/// uniform buffer via `bytemuck::bytes_of()`.
///
/// Corresponding WGSL struct:
/// ```wgsl
/// struct SceneUniform {
///     view_proj: mat4x4<f32>,
///     camera_pos: vec4<f32>,
///     light_dir: vec4<f32>,   // xyz = normalized direction, w = unused
///     ambient: vec4<f32>,     // x = ambient intensity, yzw = unused
/// };
/// ```
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniform {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    /// Directional light direction (normalized). w is unused.
    pub light_dir: [f32; 4],
    /// x = ambient intensity (0.0–1.0). yzw unused.
    pub ambient: [f32; 4],
}

impl SceneUniform {
    /// Creates a new SceneUniform with default lighting.
    ///
    /// Default light: direction `(1, 2, 1)` normalized, ambient `0.3`.
    pub fn new(view_proj: glam::Mat4, camera_pos: glam::Vec3) -> Self {
        let dir = glam::Vec3::new(1.0, 2.0, 1.0).normalize();
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
            light_dir: [dir.x, dir.y, dir.z, 0.0],
            ambient: [0.3, 0.0, 0.0, 0.0],
        }
    }

    /// Creates a SceneUniform with custom lighting.
    pub fn with_light(
        view_proj: glam::Mat4,
        camera_pos: glam::Vec3,
        light_dir: glam::Vec3,
        ambient: f32,
    ) -> Self {
        let dir = light_dir.normalize_or(glam::Vec3::Y);
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
            light_dir: [dir.x, dir.y, dir.z, 0.0],
            ambient: [ambient, 0.0, 0.0, 0.0],
        }
    }
}

/// Bundles a [`SceneUniform`] buffer, bind group layout, and bind group.
///
/// Eliminates the boilerplate of creating these three resources separately.
/// The layout is compatible with `@group(0) @binding(0) var<uniform> scene: SceneUniform`.
///
/// # Example
///
/// ```ignore
/// let scene = SceneBinding::new(&device);
///
/// // Pass the layout to renderers:
/// let renderer = InstanceRenderer::new(&ctx, scene.layout(), ...);
///
/// // Update each frame:
/// scene.update_from_camera(&queue, &camera, aspect);
///
/// // Bind in render pass:
/// pass.set_bind_group(0, scene.bind_group(), &[]);
/// ```
pub struct SceneBinding {
    layout: wgpu::BindGroupLayout,
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl SceneBinding {
    /// Creates a new `SceneBinding` with an identity view-projection matrix.
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform = SceneUniform::new(glam::Mat4::IDENTITY, glam::Vec3::ZERO);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene_uniform_buffer"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_bind_group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        Self {
            layout,
            buffer,
            bind_group,
        }
    }

    /// Returns the bind group layout for pipeline construction.
    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    /// Returns the bind group for use in render passes.
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Writes a [`SceneUniform`] to the GPU buffer.
    pub fn update(&self, queue: &wgpu::Queue, uniform: &SceneUniform) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(uniform));
    }

    /// Convenience: computes the view-projection matrix from a camera and writes
    /// a [`SceneUniform`] with default lighting.
    pub fn update_from_camera(
        &self,
        queue: &wgpu::Queue,
        camera: &(impl Camera + ?Sized),
        aspect: f32,
    ) {
        let vp = camera.view_projection_matrix(aspect);
        let uniform = SceneUniform::new(vp, camera.position());
        self.update(queue, &uniform);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec3};

    macro_rules! assert_approx {
        ($a:expr, $b:expr) => {
            assert!(
                ($a - $b).abs() < 1e-6,
                "expected {} to be approximately {}, diff = {}",
                $a,
                $b,
                ($a - $b).abs()
            );
        };
    }

    #[test]
    fn new_default_lighting() {
        let u = SceneUniform::new(Mat4::IDENTITY, Vec3::new(1.0, 2.0, 3.0));
        let dir = Vec3::new(1.0, 2.0, 1.0).normalize();
        assert_approx!(u.light_dir[0], dir.x);
        assert_approx!(u.light_dir[1], dir.y);
        assert_approx!(u.light_dir[2], dir.z);
        assert_approx!(u.light_dir[3], 0.0);
        assert_approx!(u.ambient[0], 0.3);
        assert_approx!(u.camera_pos[0], 1.0);
        assert_approx!(u.camera_pos[1], 2.0);
        assert_approx!(u.camera_pos[2], 3.0);
        assert_approx!(u.camera_pos[3], 1.0);
    }

    #[test]
    fn with_light_custom() {
        let u = SceneUniform::with_light(Mat4::IDENTITY, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.5);
        assert_approx!(u.light_dir[0], 0.0);
        assert_approx!(u.light_dir[1], 1.0);
        assert_approx!(u.light_dir[2], 0.0);
        assert_approx!(u.light_dir[3], 0.0);
        assert_approx!(u.ambient[0], 0.5);
    }

    #[test]
    fn with_light_zero_vector_falls_back_to_y() {
        let u = SceneUniform::with_light(Mat4::IDENTITY, Vec3::ZERO, Vec3::ZERO, 0.5);
        // Zero light direction should fall back to Vec3::Y, not produce NaN
        assert_approx!(u.light_dir[0], 0.0);
        assert_approx!(u.light_dir[1], 1.0);
        assert_approx!(u.light_dir[2], 0.0);
    }

    #[test]
    fn camera_pos_w_is_one() {
        for pos in [
            Vec3::ZERO,
            Vec3::new(100.0, -50.0, 0.3),
            Vec3::new(-1.0, -2.0, -3.0),
        ] {
            let u = SceneUniform::new(Mat4::IDENTITY, pos);
            assert_approx!(u.camera_pos[3], 1.0);
        }
    }
}
