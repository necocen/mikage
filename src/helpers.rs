use crate::camera::Camera;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

/// Creates a depth texture and its view.
///
/// Typically called in [`App::init`](crate::App::init) and
/// [`App::resize`](crate::App::resize). Use [`DEPTH_FORMAT`] as the format.
pub fn create_depth_texture(
    device: &wgpu::Device,
    size: PhysicalSize<u32>,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_texture"),
        size: wgpu::Extent3d {
            width: size.width.max(1),
            height: size.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
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
        let dir = light_dir.normalize();
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
/// let renderer = InstanceRenderer::new(&device, fmt, scene.layout(), ...);
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

/// Unit cube mesh data (vertices + indices).
///
/// Centered at origin, side length 1.0 (-0.5 to 0.5 on each axis).
/// 24 vertices with per-face normals, 36 indices (12 triangles).
/// Scale via model matrix to get the desired size.
pub struct CubeMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl CubeMesh {
    /// Generates a unit cube with per-face normals.
    #[rustfmt::skip]
    pub fn generate() -> Self {
        let positions = vec![
            // +Y (top)
            [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
            // -Y (bottom)
            [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
            // +X (right)
            [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5], [ 0.5, -0.5,  0.5],
            // -X (left)
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5], [-0.5, -0.5, -0.5],
            // +Z (front)
            [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
            // -Z (back)
            [ 0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
        ];
        let normals = vec![
            // +Y
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            // -Y
            [0.0,-1.0, 0.0], [0.0,-1.0, 0.0], [0.0,-1.0, 0.0], [0.0,-1.0, 0.0],
            // +X
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            // -X
            [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            // +Z
            [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
            // -Z
            [0.0, 0.0,-1.0], [0.0, 0.0,-1.0], [0.0, 0.0,-1.0], [0.0, 0.0,-1.0],
        ];
        let indices = vec![
             0,  1,  2,   0,  2,  3,  // +Y
             4,  5,  6,   4,  6,  7,  // -Y
             8,  9, 10,   8, 10, 11,  // +X
            12, 13, 14,  12, 14, 15,  // -X
            16, 17, 18,  16, 18, 19,  // +Z
            20, 21, 22,  20, 22, 23,  // -Z
        ];
        Self { positions, normals, indices }
    }
}

/// Plane (quad) mesh data (vertices + indices).
///
/// Lies in the XZ plane, centered at origin, size 1.0 x 1.0.
/// Normal points in +Y direction. 4 vertices, 6 indices (2 triangles).
pub struct PlaneMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl PlaneMesh {
    /// Generates a unit plane in the XZ plane with +Y normal.
    pub fn generate() -> Self {
        let positions = vec![
            [-0.5, 0.0, -0.5],
            [0.5, 0.0, -0.5],
            [0.5, 0.0, 0.5],
            [-0.5, 0.0, 0.5],
        ];
        let normals = vec![
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];
        Self {
            positions,
            normals,
            indices,
        }
    }
}

/// Icosphere mesh data (vertices + indices).
///
/// Use [`generate`](IcoSphereMesh::generate) to create a sphere mesh with the
/// desired subdivision level. Positions and normals are normalized (unit sphere).
pub struct IcoSphereMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

/// Unit quad mesh in the XY plane (for 2D rendering).
///
/// Centered at origin, size 1.0 x 1.0 (-0.5 to 0.5).
/// Normal points in +Z direction. 4 vertices, 6 indices (2 triangles).
pub struct QuadMesh2d {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl QuadMesh2d {
    /// Generates a unit quad in the XY plane with +Z normal.
    pub fn generate() -> Self {
        let positions = vec![
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ];
        let normals = vec![
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];
        Self {
            positions,
            normals,
            indices,
        }
    }
}

/// Regular polygon mesh in the XY plane.
///
/// Centered at origin, circumradius 1.0. Normal points in +Z direction.
/// Use `sides = 6` for a hexagon, `3` for a triangle, etc.
pub struct RegularPolygonMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl RegularPolygonMesh {
    /// Generates a regular polygon with the given number of sides.
    ///
    /// The polygon lies in the XY plane with circumradius 1.0 and +Z normals.
    /// Vertices are arranged starting from the +X axis, going counter-clockwise.
    pub fn generate(sides: u32) -> Self {
        assert!(sides >= 3, "A polygon must have at least 3 sides");

        let mut positions = Vec::with_capacity(sides as usize + 1);
        let mut normals = Vec::with_capacity(sides as usize + 1);

        // Center vertex
        positions.push([0.0, 0.0, 0.0]);
        normals.push([0.0, 0.0, 1.0]);

        // Outer vertices
        for i in 0..sides {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / sides as f32;
            positions.push([angle.cos(), angle.sin(), 0.0]);
            normals.push([0.0, 0.0, 1.0]);
        }

        // Triangle fan from center
        let mut indices = Vec::with_capacity(sides as usize * 3);
        for i in 0..sides {
            indices.push(0); // center
            indices.push(i + 1);
            indices.push(if i + 1 < sides { i + 2 } else { 1 });
        }

        Self {
            positions,
            normals,
            indices,
        }
    }
}

impl IcoSphereMesh {
    /// Generates an icosphere with the given subdivision level.
    ///
    /// - `subdivisions=0`: icosahedron (12 vertices, 20 faces)
    /// - `subdivisions=1`: 42 vertices, 80 faces
    /// - `subdivisions=2`: 162 vertices, 320 faces
    pub fn generate(subdivisions: u32) -> Self {
        let t = (1.0 + 5.0_f32.sqrt()) / 2.0;

        let mut positions: Vec<glam::Vec3> = vec![
            glam::Vec3::new(-1.0, t, 0.0).normalize(),
            glam::Vec3::new(1.0, t, 0.0).normalize(),
            glam::Vec3::new(-1.0, -t, 0.0).normalize(),
            glam::Vec3::new(1.0, -t, 0.0).normalize(),
            glam::Vec3::new(0.0, -1.0, t).normalize(),
            glam::Vec3::new(0.0, 1.0, t).normalize(),
            glam::Vec3::new(0.0, -1.0, -t).normalize(),
            glam::Vec3::new(0.0, 1.0, -t).normalize(),
            glam::Vec3::new(t, 0.0, -1.0).normalize(),
            glam::Vec3::new(t, 0.0, 1.0).normalize(),
            glam::Vec3::new(-t, 0.0, -1.0).normalize(),
            glam::Vec3::new(-t, 0.0, 1.0).normalize(),
        ];

        let mut indices: Vec<[u32; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];

        use std::collections::HashMap;

        fn get_midpoint(
            a: u32,
            b: u32,
            positions: &mut Vec<glam::Vec3>,
            cache: &mut HashMap<(u32, u32), u32>,
        ) -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&idx) = cache.get(&key) {
                return idx;
            }
            let mid = (positions[a as usize] + positions[b as usize]).normalize();
            let idx = positions.len() as u32;
            positions.push(mid);
            cache.insert(key, idx);
            idx
        }

        let mut midpoint_cache: HashMap<(u32, u32), u32> = HashMap::new();

        for _ in 0..subdivisions {
            let mut new_indices = Vec::with_capacity(indices.len() * 4);
            for tri in &indices {
                let a = get_midpoint(tri[0], tri[1], &mut positions, &mut midpoint_cache);
                let b = get_midpoint(tri[1], tri[2], &mut positions, &mut midpoint_cache);
                let c = get_midpoint(tri[2], tri[0], &mut positions, &mut midpoint_cache);
                new_indices.push([tri[0], a, c]);
                new_indices.push([tri[1], b, a]);
                new_indices.push([tri[2], c, b]);
                new_indices.push([a, b, c]);
            }
            indices = new_indices;
            midpoint_cache.clear();
        }

        let out_positions: Vec<[f32; 3]> = positions.iter().map(|p| p.to_array()).collect();
        let out_normals: Vec<[f32; 3]> =
            positions.iter().map(|p| p.normalize().to_array()).collect();
        let out_indices: Vec<u32> = indices.iter().flat_map(|t| t.iter().copied()).collect();

        IcoSphereMesh {
            positions: out_positions,
            normals: out_normals,
            indices: out_indices,
        }
    }
}
