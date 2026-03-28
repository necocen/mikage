//! Renders solid-colored meshes with SceneUniform lighting.
//!
//! Provides opaque (lit) and transparent (unlit) pipelines for rendering
//! solid-colored meshes. Uses [`SceneUniform`](crate::SceneUniform) at
//! `@group(0)` (the same bind group layout the application already has)
//! and per-object model matrix + RGBA color at `@group(1)`.
//!
//! # Example
//!
//! ```ignore
//! let mut solid = SolidRenderer::new(&ctx, &scene_bind_group_layout);
//!
//! let cube = mikage::CubeMesh::generate();
//! let id = solid.add_object(&device, &cube.positions, &cube.normals, &cube.indices);
//! solid.update_object(&queue, id, Mat4::IDENTITY, Vec4::ONE);
//!
//! // In render pass (scene bind group already set at group 0):
//! solid.render(&mut pass);
//! ```

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec4};
use wgpu::util::DeviceExt;

use crate::helpers::{MeshBuffers, POSITION_NORMAL_LAYOUT, uniform_buffer_entry};
use crate::shader_processor::mikage_shader_processor;

const SHADER_SOURCE: &str = include_str!("../assets/shaders/solid.wgsl");

/// Per-object model matrix + RGBA color uniform (80 bytes).
///
/// Written to the GPU via [`SolidRenderer::update_object`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ModelUniform {
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
}

impl ModelUniform {
    /// Creates a new ModelUniform from a model matrix and RGBA color.
    pub fn new(model: Mat4, color: Vec4) -> Self {
        Self {
            model: model.to_cols_array_2d(),
            color: color.to_array(),
        }
    }
}

/// A handle to a registered object in the [`SolidRenderer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SolidObjectId(usize);

/// Per-object GPU resources.
struct ObjectData {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Current alpha value; determines opaque vs. transparent pipeline.
    alpha: f32,
}

/// Renders solid-colored meshes with SceneUniform lighting.
///
/// Objects with `alpha >= 1.0` are drawn with the opaque (lit) pipeline.
/// Objects with `alpha < 1.0` are drawn with the transparent (unlit) pipeline.
/// Opaque objects are always drawn first, then transparent objects.
///
/// The scene bind group (`@group(0)`) must be set by the caller before
/// calling [`render`](SolidRenderer::render).
pub struct SolidRenderer {
    opaque_pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    model_bind_group_layout: wgpu::BindGroupLayout,
    objects: Vec<ObjectData>,
}

impl SolidRenderer {
    /// Creates a new SolidRenderer.
    ///
    /// `scene_bind_group_layout` is the layout for `@group(0)` containing
    /// a [`SceneUniform`](crate::SceneUniform) buffer. This must match the
    /// bind group you set at group 0 before calling [`render`](SolidRenderer::render).
    pub fn new(gpu: &crate::GpuContext, scene_bind_group_layout: &wgpu::BindGroupLayout) -> Self {
        Self::from_parts(
            &gpu.device,
            gpu.render_format(),
            scene_bind_group_layout,
            gpu.sample_count(),
        )
    }

    /// Low-level constructor. Prefer [`new`](Self::new) which takes `&GpuContext`.
    #[doc(hidden)]
    pub fn from_parts(
        device: &wgpu::Device,
        render_format: wgpu::TextureFormat,
        scene_bind_group_layout: &wgpu::BindGroupLayout,
        sample_count: u32,
    ) -> Self {
        // Model bind group layout: single uniform buffer at binding 0
        let model_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("solid_model_bind_group_layout"),
                entries: &[uniform_buffer_entry(
                    0,
                    wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                )],
            });

        let resolved_source = mikage_shader_processor()
            .resolve(SHADER_SOURCE)
            .expect("failed to resolve solid shader imports");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("solid_shader"),
            source: wgpu::ShaderSource::Wgsl(resolved_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("solid_pipeline_layout"),
            bind_group_layouts: &[scene_bind_group_layout, &model_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Opaque pipeline (lit fragment, depth write, back-face culling)
        let opaque_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("solid_opaque_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[POSITION_NORMAL_LAYOUT],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fragment_lit"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: render_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        // Transparent pipeline (unlit fragment, alpha blend, no depth write, no culling)
        let transparent_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("solid_transparent_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[POSITION_NORMAL_LAYOUT],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // show both faces for transparent objects
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fragment_unlit"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: render_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        Self {
            opaque_pipeline,
            transparent_pipeline,
            model_bind_group_layout,
            objects: Vec::new(),
        }
    }

    /// Adds an object with the given mesh data.
    ///
    /// `positions` and `normals` must have the same length. They are
    /// interleaved into a single vertex buffer (position + normal per vertex).
    /// Returns a [`SolidObjectId`] handle for later updates.
    ///
    /// The object starts with an identity model matrix and white color.
    /// Call [`update_object`](SolidRenderer::update_object) to set the
    /// actual transform and color.
    pub fn add_object(
        &mut self,
        device: &wgpu::Device,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
    ) -> SolidObjectId {
        let mesh = MeshBuffers::from_position_normal(device, positions, normals, indices);

        let uniform = ModelUniform::new(Mat4::IDENTITY, Vec4::ONE);
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("solid_model_uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("solid_model_bind_group"),
            layout: &self.model_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let id = SolidObjectId(self.objects.len());
        self.objects.push(ObjectData {
            vertex_buffer: mesh.vertex_buffer,
            index_buffer: mesh.index_buffer,
            index_count: mesh.index_count,
            uniform_buffer,
            bind_group,
            alpha: 1.0,
        });

        id
    }

    /// Updates an object's model matrix and RGBA color.
    ///
    /// The `color.w` (alpha) value determines which pipeline is used:
    /// - `alpha >= 1.0` -> opaque (lit) pipeline
    /// - `alpha < 1.0` -> transparent (unlit) pipeline
    pub fn update_object(
        &mut self,
        queue: &wgpu::Queue,
        id: SolidObjectId,
        model: Mat4,
        color: Vec4,
    ) {
        let obj = &mut self.objects[id.0];
        obj.alpha = color.w;
        let uniform = ModelUniform::new(model, color);
        queue.write_buffer(&obj.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// Renders all objects.
    ///
    /// Opaque objects (alpha >= 1.0) are drawn first with the lit pipeline,
    /// then transparent objects (alpha < 1.0) with the unlit pipeline.
    ///
    /// The caller must set the scene bind group at `@group(0)` before calling
    /// this method (via `pass.set_bind_group(0, &scene_bind_group, &[])`).
    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        // Draw opaque objects first
        pass.set_pipeline(&self.opaque_pipeline);
        for obj in &self.objects {
            if obj.alpha >= 1.0 {
                pass.set_vertex_buffer(0, obj.vertex_buffer.slice(..));
                pass.set_index_buffer(obj.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_bind_group(1, &obj.bind_group, &[]);
                pass.draw_indexed(0..obj.index_count, 0, 0..1);
            }
        }

        // Then draw transparent objects
        pass.set_pipeline(&self.transparent_pipeline);
        for obj in &self.objects {
            if obj.alpha < 1.0 {
                pass.set_vertex_buffer(0, obj.vertex_buffer.slice(..));
                pass.set_index_buffer(obj.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_bind_group(1, &obj.bind_group, &[]);
                pass.draw_indexed(0..obj.index_count, 0, 0..1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec3, Vec4};

    #[test]
    fn model_uniform_identity() {
        let u = ModelUniform::new(Mat4::IDENTITY, Vec4::ONE);
        assert_eq!(u.model, Mat4::IDENTITY.to_cols_array_2d());
        assert_eq!(u.color, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn model_uniform_transform() {
        let transform = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let color = Vec4::new(0.5, 0.0, 1.0, 0.8);
        let u = ModelUniform::new(transform, color);
        assert_eq!(u.model, transform.to_cols_array_2d());
        assert_eq!(u.color, [0.5, 0.0, 1.0, 0.8]);
    }
}
