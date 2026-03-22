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
