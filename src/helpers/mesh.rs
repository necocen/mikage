use wgpu::util::DeviceExt;

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

/// Interleaved position+normal vertex buffer, index buffer, and index count.
///
/// Use [`from_position_normal`](MeshBuffers::from_position_normal) to create
/// from separate position/normal slices (the common mesh data layout in mikage).
pub struct MeshBuffers {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

impl MeshBuffers {
    /// Creates mesh GPU buffers by interleaving `positions` and `normals`.
    ///
    /// Each vertex is stored as `[f32; 3] position + [f32; 3] normal` (24 bytes).
    /// Use [`POSITION_NORMAL_LAYOUT`] for the corresponding `VertexBufferLayout`.
    pub fn from_position_normal(
        device: &wgpu::Device,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
    ) -> Self {
        assert_eq!(
            positions.len(),
            normals.len(),
            "positions and normals must have the same length"
        );

        let mut vertex_data: Vec<f32> = Vec::with_capacity(positions.len() * 6);
        for i in 0..positions.len() {
            vertex_data.extend_from_slice(&positions[i]);
            vertex_data.extend_from_slice(&normals[i]);
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_vertex_buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_index_buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
        }
    }
}

/// Vertex buffer layout for interleaved position (Float32x3) + normal (Float32x3).
///
/// - `location 0`: position
/// - `location 1`: normal
/// - stride: 24 bytes
pub const POSITION_NORMAL_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: 24,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &[
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 12,
            shader_location: 1,
        },
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_counts() {
        let mesh = CubeMesh::generate();
        assert_eq!(mesh.positions.len(), 24);
        assert_eq!(mesh.normals.len(), 24);
        assert_eq!(mesh.indices.len(), 36);
    }

    #[test]
    fn cube_normals_unit_length() {
        let mesh = CubeMesh::generate();
        for n in &mesh.normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "normal length {len} is not unit");
        }
    }

    #[test]
    fn cube_positions_in_range() {
        let mesh = CubeMesh::generate();
        for p in &mesh.positions {
            for &c in p {
                assert!(
                    (-0.5..=0.5).contains(&c),
                    "position component {c} out of [-0.5, 0.5]"
                );
            }
        }
    }

    #[test]
    fn plane_counts() {
        let mesh = PlaneMesh::generate();
        assert_eq!(mesh.positions.len(), 4);
        assert_eq!(mesh.normals.len(), 4);
        assert_eq!(mesh.indices.len(), 6);
    }

    #[test]
    fn plane_normals_are_up() {
        let mesh = PlaneMesh::generate();
        for n in &mesh.normals {
            assert_eq!(*n, [0.0, 1.0, 0.0]);
        }
    }

    #[test]
    fn icosphere_0_counts() {
        let mesh = IcoSphereMesh::generate(0);
        assert_eq!(mesh.positions.len(), 12);
        assert_eq!(mesh.indices.len(), 60);
    }

    #[test]
    fn icosphere_1_counts() {
        let mesh = IcoSphereMesh::generate(1);
        assert_eq!(mesh.positions.len(), 42);
        assert_eq!(mesh.indices.len(), 240);
    }

    #[test]
    fn icosphere_positions_unit_length() {
        for subdivisions in [0, 2] {
            let mesh = IcoSphereMesh::generate(subdivisions);
            for p in &mesh.positions {
                let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                assert!(
                    (len - 1.0).abs() < 1e-5,
                    "icosphere({subdivisions}) position length {len} is not unit"
                );
            }
        }
    }

    #[test]
    fn quad2d_counts() {
        let mesh = QuadMesh2d::generate();
        assert_eq!(mesh.positions.len(), 4);
        assert_eq!(mesh.normals.len(), 4);
        assert_eq!(mesh.indices.len(), 6);
    }

    #[test]
    fn quad2d_z_is_zero() {
        let mesh = QuadMesh2d::generate();
        for p in &mesh.positions {
            assert_eq!(p[2], 0.0, "Z component should be 0.0, got {}", p[2]);
        }
    }

    #[test]
    fn regular_polygon_triangle() {
        let mesh = RegularPolygonMesh::generate(3);
        assert_eq!(mesh.positions.len(), 4); // center + 3
        assert_eq!(mesh.indices.len(), 9);
    }

    #[test]
    fn regular_polygon_hexagon() {
        let mesh = RegularPolygonMesh::generate(6);
        assert_eq!(mesh.positions.len(), 7);
        assert_eq!(mesh.indices.len(), 18);
    }

    #[test]
    fn regular_polygon_unit_circle() {
        let mesh = RegularPolygonMesh::generate(6);
        // Skip index 0 (center vertex), check outer vertices
        for p in &mesh.positions[1..] {
            let len = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "outer vertex XY length {len} is not unit"
            );
        }
    }

    #[test]
    #[should_panic]
    fn regular_polygon_panics_below_3() {
        RegularPolygonMesh::generate(2);
    }
}
