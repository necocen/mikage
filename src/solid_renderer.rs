//! Renders solid-colored meshes with SceneUniform lighting.
//!
//! Provides opaque and transparent pipelines with selectable shading for rendering
//! solid-colored meshes. Uses [`SceneUniform`](crate::SceneUniform) at
//! `@group(0)` (the same bind group layout the application already has)
//! and per-object model matrix + RGBA color at `@group(1)`.
//!
//! # Transparency policy
//!
//! Pipeline selection is driven by the alpha component of the color passed
//! to [`SolidRenderer::update_object`]:
//!
//! - **alpha >= 1.0** — drawn with the **opaque (lit)** pipeline: back-face
//!   culling, depth write enabled, directional + ambient lighting.
//! - **alpha < 1.0** — drawn with the **transparent (unlit)** pipeline:
//!   no culling, depth write disabled, alpha blending, flat color only.
//!
//! [`SolidRenderer::set_shading`] overrides lighting independently of alpha.
//! [`SolidShading::Lit`] enables Lambert lighting on transparent objects;
//! [`SolidShading::Unlit`] also works on opaque objects. Alpha still determines
//! blending, culling, depth writes, and opaque-before-transparent ordering.
//!
//! Opaque objects are always rendered before transparent objects. However,
//! **transparent objects are not depth-sorted**; overlapping transparent
//! geometry may produce order-dependent blending artifacts. If correct
//! transparency ordering is required, sort objects externally or use an
//! OIT (order-independent transparency) technique.
//!
//! Objects cannot be removed after registration. [`SolidObjectId`] values are
//! stable indices into the renderer's internal object list; create a new
//! renderer if you need to rebuild the set.
//!
//! # Example
//!
//! ```ignore
//! let mut solid = SolidRenderer::new(&ctx, &scene_bind_group_layout);
//!
//! let cube = mikage::CubeMesh::generate();
//! let id = solid.add_object(&ctx, &cube.positions, &cube.normals, &cube.indices);
//! solid.update_object(&ctx, id, Mat4::IDENTITY, Vec4::ONE);
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
///
/// Obtained from [`SolidRenderer::add_object`]. Only valid for the
/// renderer that created it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SolidObjectId(usize);

/// Per-object shading, independent of opacity and depth behavior.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum SolidShading {
    /// Preserve the default: opaque objects are lit, transparent objects unlit.
    #[default]
    Auto,
    /// Lambert lighting from [`SceneUniform`](crate::SceneUniform), at any alpha.
    Lit,
    /// Pass material RGB through without lighting, at any alpha.
    Unlit,
}

impl SolidShading {
    fn is_lit(self, alpha: f32) -> bool {
        match self {
            Self::Auto => alpha >= 1.0,
            Self::Lit => true,
            Self::Unlit => false,
        }
    }
}

/// Per-object GPU resources.
struct ObjectData {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Current alpha value; determines opaque vs. transparent pipeline.
    alpha: f32,
    shading: SolidShading,
}

/// Renders solid-colored meshes with SceneUniform lighting.
///
/// Objects with `alpha >= 1.0` are drawn with the opaque (lit) pipeline.
/// Objects with `alpha < 1.0` are automatically switched to the transparent
/// (unlit) pipeline — no API call is needed to change pipelines.
/// Opaque objects are always drawn first, then transparent objects.
/// Use [`Self::set_shading`] to override lighting independently of alpha.
///
/// **Limitation:** transparent objects are not depth-sorted; overlapping
/// transparent geometry may exhibit order-dependent blending artifacts.
///
/// The scene bind group (`@group(0)`) must be set by the caller before
/// calling [`render`](SolidRenderer::render).
pub struct SolidRenderer {
    opaque_pipeline: wgpu::RenderPipeline,
    opaque_unlit_pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    transparent_lit_pipeline: wgpu::RenderPipeline,
    model_bind_group_layout: wgpu::BindGroupLayout,
    objects: Vec<ObjectData>,
}

impl SolidRenderer {
    /// Creates a new SolidRenderer.
    ///
    /// `scene_bind_group_layout` is the layout for `@group(0)` containing
    /// a [`SceneUniform`](crate::SceneUniform) buffer. This must match the
    /// bind group you set at group 0 before calling [`render`](SolidRenderer::render).
    pub fn new(
        gpu: &crate::GpuContext,
        target: crate::RenderTargetConfig,
        scene_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let device = &gpu.device;
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
            bind_group_layouts: &[
                Some(scene_bind_group_layout),
                Some(&model_bind_group_layout),
            ],
            immediate_size: 0,
        });

        // Opaque pipeline (lit fragment, depth write, back-face culling)
        let mut opaque_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("solid_opaque_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[Some(POSITION_NORMAL_LAYOUT)],
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
                format: target.depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: target.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fragment_lit"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target.color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        };
        let opaque_pipeline = device.create_render_pipeline(&opaque_descriptor);
        opaque_descriptor.label = Some("solid_opaque_unlit_pipeline");
        opaque_descriptor.fragment.as_mut().unwrap().entry_point = Some("fragment_unlit");
        let opaque_unlit_pipeline = device.create_render_pipeline(&opaque_descriptor);

        // Transparent pipeline (unlit fragment, alpha blend, no depth write, no culling)
        let mut transparent_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("solid_transparent_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[Some(POSITION_NORMAL_LAYOUT)],
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
                format: target.depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: target.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fragment_unlit"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target.color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        };
        let transparent_pipeline = device.create_render_pipeline(&transparent_descriptor);
        transparent_descriptor.label = Some("solid_transparent_lit_pipeline");
        transparent_descriptor
            .fragment
            .as_mut()
            .unwrap()
            .entry_point = Some("fragment_lit");
        let transparent_lit_pipeline = device.create_render_pipeline(&transparent_descriptor);

        Self {
            opaque_pipeline,
            opaque_unlit_pipeline,
            transparent_pipeline,
            transparent_lit_pipeline,
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
    /// Registers a new mesh object and returns its handle.
    ///
    /// Call [`update_object`](SolidRenderer::update_object) to set the
    /// actual transform and color.
    pub fn add_object(
        &mut self,
        gpu: &crate::GpuContext,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
    ) -> SolidObjectId {
        let device = &gpu.device;
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
            shading: SolidShading::Auto,
        });

        id
    }

    /// Updates an object's model matrix and RGBA color.
    ///
    /// The `color.w` (alpha) value determines which pipeline is used:
    /// - `alpha >= 1.0` — opaque (lit) pipeline with directional + ambient lighting
    /// - `alpha < 1.0` — transparent (unlit) pipeline with alpha blending,
    ///   no depth write, and no back-face culling
    ///
    /// The switch is automatic and takes effect on the next
    /// [`render`](SolidRenderer::render) call. Note that transparent objects
    /// are not depth-sorted; see the [module-level docs](self) for details.
    /// An explicit [`Self::set_shading`] choice is preserved by this update.
    pub fn update_object(
        &mut self,
        gpu: &crate::GpuContext,
        id: SolidObjectId,
        model: Mat4,
        color: Vec4,
    ) {
        let obj = self
            .objects
            .get_mut(id.0)
            .expect("invalid SolidObjectId: object not found");
        obj.alpha = color.w;
        let uniform = ModelUniform::new(model, color);
        gpu.queue
            .write_buffer(&obj.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// Selects lighting without changing color, alpha, transform, or draw order.
    ///
    /// `Lit` transparent objects retain alpha blending, no culling, and no depth
    /// writes. Both faces use their mesh normals; back-face normals are not flipped.
    /// The choice persists across [`Self::update_object`] calls. `Auto` restores
    /// the original alpha-based shading choice.
    pub fn set_shading(&mut self, id: SolidObjectId, shading: SolidShading) {
        self.objects
            .get_mut(id.0)
            .expect("invalid SolidObjectId: object not found")
            .shading = shading;
    }

    /// Renders all objects.
    ///
    /// Opaque objects (alpha >= 1.0) are drawn first, then transparent objects
    /// (alpha < 1.0), with each object's selected shading. Insertion order is
    /// preserved within each group, including when shading differs.
    ///
    /// The caller must set the scene bind group at `@group(0)` before calling
    /// this method (via `pass.set_bind_group(0, &scene_bind_group, &[])`).
    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        // Draw opaque objects first
        for obj in &self.objects {
            if obj.alpha >= 1.0 {
                pass.set_pipeline(if obj.shading.is_lit(obj.alpha) {
                    &self.opaque_pipeline
                } else {
                    &self.opaque_unlit_pipeline
                });
                pass.set_vertex_buffer(0, obj.vertex_buffer.slice(..));
                pass.set_index_buffer(obj.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_bind_group(1, &obj.bind_group, &[]);
                pass.draw_indexed(0..obj.index_count, 0, 0..1);
            }
        }

        // Then draw transparent objects
        for obj in &self.objects {
            if obj.alpha < 1.0 {
                pass.set_pipeline(if obj.shading.is_lit(obj.alpha) {
                    &self.transparent_lit_pipeline
                } else {
                    &self.transparent_pipeline
                });
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
