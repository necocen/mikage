use dpi::PhysicalSize;
use wgpu::util::DeviceExt;

use crate::camera::Camera;

/// Creates a depth texture and its view.
///
/// Typically called in the factory closure and
/// [`App::resize`](crate::App::resize). Format and sample count match the explicit target.
pub fn create_depth_texture(
    gpu: &crate::GpuContext,
    size: PhysicalSize<u32>,
    target: crate::RenderTargetConfig,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_texture"),
        size: wgpu::Extent3d {
            width: size.width.max(1),
            height: size.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: target.sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: target.depth_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Common depth format constant.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Independent, unitless light gains applied to linear material RGB.
///
/// Lit color is `color * (ambient_intensity + directional_intensity *
/// max(dot(normal, direction), 0))`. Zero disables a contribution; values above
/// one are allowed. There is no automatic exposure or tone mapping, so an
/// unorm render target clamps values outside its range. Negative or non-finite
/// intensities are treated as zero when uploaded.
///
/// ```
/// use mikage::{SceneLighting, SceneUniform, glam::{Mat4, Vec3}};
/// let uniform = SceneUniform::with_lighting(Mat4::IDENTITY, Vec3::ZERO, SceneLighting {
///     direction: Vec3::Z,
///     ambient_intensity: 0.2,
///     directional_intensity: 0.8,
/// });
/// assert_eq!(uniform.ambient, [0.2, 0.8, 1.0, 0.0]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneLighting {
    /// Direction **toward** the light; normalized on upload. A zero or invalid
    /// direction falls back to +Y. Use zero intensity to disable the light.
    pub direction: glam::Vec3,
    pub ambient_intensity: f32,
    pub directional_intensity: f32,
}

impl Default for SceneLighting {
    fn default() -> Self {
        Self {
            direction: glam::Vec3::new(1.0, 2.0, 1.0),
            ambient_intensity: 0.3,
            directional_intensity: 0.7,
        }
    }
}

impl SceneLighting {
    fn uniform_fields(self) -> [[f32; 4]; 2] {
        let dir = self.direction.normalize_or(glam::Vec3::Y);
        let gain = |value: f32| {
            if value.is_finite() {
                value.max(0.0)
            } else {
                0.0
            }
        };
        [
            [dir.x, dir.y, dir.z, 0.0],
            [
                gain(self.ambient_intensity),
                gain(self.directional_intensity),
                1.0,
                0.0,
            ],
        ]
    }
}

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
///     ambient: vec4<f32>,     // x = ambient, y = directional, z = independent mode
/// };
/// ```
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniform {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    /// Directional light direction (normalized). w is unused.
    pub light_dir: [f32; 4],
    /// x = ambient intensity. When z is 1, y is independent directional
    /// intensity; when z is 0, directional intensity is `1 - x` (legacy).
    /// w is reserved. Prefer [`Self::set_lighting`] to encode these fields.
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

    /// Creates a SceneUniform with legacy lighting: ambient plus directional
    /// intensity `1 - ambient`. Use [`Self::with_lighting`] for independent gains.
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

    /// Creates a scene with independent ambient and directional gains.
    pub fn with_lighting(
        view_proj: glam::Mat4,
        camera_pos: glam::Vec3,
        lighting: SceneLighting,
    ) -> Self {
        let mut uniform = Self::new(view_proj, camera_pos);
        uniform.set_lighting(lighting);
        uniform
    }

    /// Changes lighting while preserving camera fields and the 112-byte layout.
    pub fn set_lighting(&mut self, lighting: SceneLighting) {
        [self.light_dir, self.ambient] = lighting.uniform_fields();
    }
}

/// Bundles a [`SceneUniform`] buffer, bind group layout, and bind group.
///
/// Eliminates the boilerplate of creating these three resources separately.
/// The layout is compatible with `@group(0) @binding(0) var<uniform> scene: SceneUniform`.
///
/// # Example
///
/// ```no_run
/// use mikage::{Camera, SceneBinding, SceneLighting, wgpu};
/// # fn update_scene(device: &wgpu::Device, queue: &wgpu::Queue, camera: &impl Camera) {
/// let scene = SceneBinding::new(device);
/// scene.update_lighting(queue, SceneLighting {
///     ambient_intensity: 0.2,
///     directional_intensity: 0.8,
///     ..Default::default()
/// });
/// // Each frame, retain the lighting while moving the camera:
/// scene.update_camera(queue, camera, 16.0 / 9.0);
/// // Use scene.layout() for pipelines and scene.bind_group() at group 0.
/// # }
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

    /// Updates only camera matrices and position, preserving uploaded lighting.
    ///
    /// This also preserves lighting supplied through [`Self::update`], including
    /// legacy [`SceneUniform::with_light`] settings, without a CPU-side cache.
    pub fn update_camera(&self, queue: &wgpu::Queue, camera: &(impl Camera + ?Sized), aspect: f32) {
        let uniform = SceneUniform::new(camera.view_projection_matrix(aspect), camera.position());
        let camera_end = std::mem::offset_of!(SceneUniform, light_dir);
        queue.write_buffer(&self.buffer, 0, &bytemuck::bytes_of(&uniform)[..camera_end]);
    }

    /// Updates only lighting, preserving the uploaded camera matrices and position.
    pub fn update_lighting(&self, queue: &wgpu::Queue, lighting: SceneLighting) {
        queue.write_buffer(
            &self.buffer,
            std::mem::offset_of!(SceneUniform, light_dir) as u64,
            bytemuck::bytes_of(&lighting.uniform_fields()),
        );
    }

    /// Convenience: computes the view-projection matrix from a camera and writes
    /// a [`SceneUniform`] with default lighting.
    ///
    /// This resets lighting for compatibility. Use [`Self::update_camera`] to
    /// preserve configured lighting.
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
    fn lighting_retains_uniform_layout_and_legacy_encoding() {
        assert_eq!(std::mem::size_of::<SceneUniform>(), 112);
        assert_eq!(std::mem::offset_of!(SceneUniform, camera_pos), 64);
        assert_eq!(std::mem::offset_of!(SceneUniform, light_dir), 80);
        assert_eq!(std::mem::offset_of!(SceneUniform, ambient), 96);
        assert_eq!(
            SceneUniform::new(Mat4::IDENTITY, Vec3::ZERO).ambient,
            [0.3, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            SceneUniform::with_light(Mat4::IDENTITY, Vec3::ZERO, Vec3::Z, 0.8).ambient,
            [0.8, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn independent_lighting_preserves_camera_and_normalizes_direction() {
        let vp = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let mut u = SceneUniform::new(vp, Vec3::new(4.0, 5.0, 6.0));
        u.set_lighting(SceneLighting {
            direction: Vec3::new(0.0, 0.0, 5.0),
            ambient_intensity: 0.2,
            directional_intensity: 1.5,
        });
        assert_eq!(u.view_proj, vp.to_cols_array_2d());
        assert_eq!(u.camera_pos, [4.0, 5.0, 6.0, 1.0]);
        assert_eq!(u.light_dir, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(u.ambient, [0.2, 1.5, 1.0, 0.0]);
    }

    #[test]
    fn independent_lighting_handles_disabled_and_invalid_inputs() {
        for direction in [
            Vec3::ZERO,
            Vec3::splat(f32::NAN),
            Vec3::splat(f32::INFINITY),
        ] {
            for gain in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let u = SceneUniform::with_lighting(
                    Mat4::IDENTITY,
                    Vec3::ZERO,
                    SceneLighting {
                        direction,
                        ambient_intensity: gain,
                        directional_intensity: gain,
                    },
                );
                assert_eq!(u.light_dir, [0.0, 1.0, 0.0, 0.0]);
                assert_eq!(u.ambient, [0.0, 0.0, 1.0, 0.0]);
            }
        }
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
