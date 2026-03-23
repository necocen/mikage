//! GPU-instanced rendering for large numbers of identical meshes.
//!
//! [`InstanceRenderer`] draws a single base mesh many times, each with
//! per-instance data provided via a type implementing [`InstanceVertex`].
//! The default vertex type [`InstanceData`] provides position, scale, and color.
//!
//! # Example (default vertex)
//!
//! ```ignore
//! let hex = mikage::RegularPolygonMesh::generate(6);
//! let mut renderer = InstanceRenderer::new(
//!     &device, render_format, &scene_bgl,
//!     &hex.positions, &hex.normals, &hex.indices,
//!     InstanceRendererConfig::default_2d(),
//! );
//!
//! renderer.update_instances(&device, &queue, &instances);
//!
//! // In render pass (scene bind group already set at group 0):
//! renderer.render(&mut pass);
//! ```
//!
//! # Custom vertex types
//!
//! Implement [`InstanceVertex`] for your own type and use
//! [`InstanceRenderer::with_shader`]:
//!
//! ```ignore
//! #[repr(C)]
//! #[derive(Clone, Copy, Pod, Zeroable)]
//! struct TileInstance {
//!     pos_angle: [f32; 4], // xyz + rotation angle
//! }
//!
//! impl InstanceVertex for TileInstance {
//!     fn vertex_attributes() -> Vec<wgpu::VertexAttribute> {
//!         vec![wgpu::VertexAttribute {
//!             format: wgpu::VertexFormat::Float32x4,
//!             offset: 0,
//!             shader_location: 2,
//!         }]
//!     }
//! }
//!
//! let renderer = InstanceRenderer::<TileInstance>::with_shader(
//!     &device, render_format, &scene_bgl,
//!     &mesh.positions, &mesh.normals, &mesh.indices,
//!     &resolved_shader_source,
//!     InstanceRendererConfig::default_2d(),
//! );
//! ```

use std::marker::PhantomData;

use bytemuck::{Pod, Zeroable};

use crate::helpers::{MeshBuffers, POSITION_NORMAL_LAYOUT};
use crate::shader_processor::mikage_shader_processor;

const SHADER_SOURCE: &str = include_str!("../assets/shaders/instancing.wgsl");

/// Trait for per-instance vertex data.
///
/// Implementations must be `#[repr(C)]` and derive [`Pod`] + [`Zeroable`].
/// Vertex attributes should use shader locations starting at 2
/// (locations 0 and 1 are reserved for mesh position and normal).
pub trait InstanceVertex: Pod + Zeroable {
    /// Returns the vertex attributes for the instance buffer layout.
    fn vertex_attributes() -> Vec<wgpu::VertexAttribute>;
}

/// Per-instance data: position + uniform scale + RGBA color (32 bytes).
///
/// - `pos_scale`: xyz = world position, w = uniform scale
/// - `color`: RGBA
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct InstanceData {
    pub pos_scale: [f32; 4],
    pub color: [f32; 4],
}

impl InstanceVertex for InstanceData {
    fn vertex_attributes() -> Vec<wgpu::VertexAttribute> {
        vec![
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 16,
                shader_location: 3,
            },
        ]
    }
}

/// Configuration for creating an [`InstanceRenderer`].
pub struct InstanceRendererConfig<'a> {
    /// Vertex shader entry point name.
    pub vertex_entry: &'a str,
    /// Fragment shader entry point name.
    pub fragment_entry: &'a str,
    /// Enable depth testing/writing.
    pub depth: bool,
    /// Multisample count (1 = no MSAA).
    pub sample_count: u32,
}

impl InstanceRendererConfig<'_> {
    /// Configuration for 3D rendering: lit shading, depth enabled, no MSAA.
    ///
    /// Uses the built-in instancing shader entry points (`vertex` / `fragment_lit`).
    pub fn default_3d() -> InstanceRendererConfig<'static> {
        InstanceRendererConfig {
            vertex_entry: "vertex",
            fragment_entry: "fragment_lit",
            depth: true,
            sample_count: 1,
        }
    }

    /// Configuration for 2D rendering: unlit shading, no depth, no MSAA.
    ///
    /// Uses the built-in instancing shader entry points (`vertex` / `fragment_unlit`).
    pub fn default_2d() -> InstanceRendererConfig<'static> {
        InstanceRendererConfig {
            vertex_entry: "vertex",
            fragment_entry: "fragment_unlit",
            depth: false,
            sample_count: 1,
        }
    }
}

/// Renders many copies of a single mesh with per-instance data.
///
/// Generic over the instance vertex type `V`. The default type [`InstanceData`]
/// provides position + scale + color. Use [`with_shader`](InstanceRenderer::with_shader)
/// for custom vertex types with your own shader.
///
/// The scene bind group (`@group(0)`) must be set by the caller before
/// calling [`render`](InstanceRenderer::render).
pub struct InstanceRenderer<V: InstanceVertex = InstanceData> {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    instance_buffer: wgpu::Buffer,
    instance_capacity: u32,
    instance_count: u32,
    _phantom: PhantomData<V>,
}

const INITIAL_INSTANCE_CAPACITY: u32 = 1024;

impl InstanceRenderer<InstanceData> {
    /// Creates an `InstanceRenderer` using the built-in instancing shader.
    ///
    /// This is a convenience constructor for the default [`InstanceData`] layout
    /// (position + scale + color). For custom vertex types, use
    /// [`with_shader`](InstanceRenderer::with_shader).
    pub fn new(
        device: &wgpu::Device,
        render_format: wgpu::TextureFormat,
        scene_bind_group_layout: &wgpu::BindGroupLayout,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
        config: InstanceRendererConfig,
    ) -> Self {
        let resolved = mikage_shader_processor()
            .resolve(SHADER_SOURCE)
            .expect("failed to resolve instancing shader imports");
        Self::with_shader(
            device,
            render_format,
            scene_bind_group_layout,
            positions,
            normals,
            indices,
            &resolved,
            config,
        )
    }
}

impl<V: InstanceVertex> InstanceRenderer<V> {
    /// Creates an `InstanceRenderer` with a custom shader and vertex type.
    ///
    /// `shader_source` must be import-resolved WGSL. Use [`ShaderProcessor`](crate::ShaderProcessor)
    /// to resolve `#import` directives before passing the source here.
    ///
    /// The shader must declare vertex attributes matching `V::vertex_attributes()`
    /// at the expected shader locations (starting at 2).
    #[allow(clippy::too_many_arguments)]
    pub fn with_shader(
        device: &wgpu::Device,
        render_format: wgpu::TextureFormat,
        scene_bind_group_layout: &wgpu::BindGroupLayout,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
        shader_source: &str,
        config: InstanceRendererConfig,
    ) -> Self {
        let mesh = MeshBuffers::from_position_normal(device, positions, normals, indices);

        let instance_stride = std::mem::size_of::<V>() as u64;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_data_buffer"),
            size: (INITIAL_INSTANCE_CAPACITY as u64) * instance_stride,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("instancing_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("instancing_pipeline_layout"),
            bind_group_layouts: &[scene_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Instance buffer layout from the vertex type
        let instance_attributes = V::vertex_attributes();
        let instance_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: instance_stride,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &instance_attributes,
        };

        let depth_stencil = if config.depth {
            Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            })
        } else {
            None
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("instancing_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some(config.vertex_entry),
                compilation_options: Default::default(),
                buffers: &[POSITION_NORMAL_LAYOUT, instance_buffer_layout],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil,
            multisample: wgpu::MultisampleState {
                count: config.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some(config.fragment_entry),
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
            pipeline,
            vertex_buffer: mesh.vertex_buffer,
            index_buffer: mesh.index_buffer,
            index_count: mesh.index_count,
            instance_buffer,
            instance_capacity: INITIAL_INSTANCE_CAPACITY,
            instance_count: 0,
            _phantom: PhantomData,
        }
    }

    /// Updates instance data. Resizes the GPU buffer if needed.
    pub fn update_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[V],
    ) {
        self.instance_count = instances.len() as u32;
        if self.instance_count == 0 {
            return;
        }

        let instance_stride = std::mem::size_of::<V>() as u64;

        // Grow buffer if needed (power-of-two capacity)
        if self.instance_count > self.instance_capacity {
            let new_cap = self
                .instance_count
                .max(1024)
                .checked_next_power_of_two()
                .unwrap_or(self.instance_count);
            self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instance_data_buffer"),
                size: (new_cap as u64) * instance_stride,
                usage: wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_cap;
        }

        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));
    }

    /// Returns a reference to the instance buffer.
    ///
    /// Use this to bind the buffer as storage in a compute shader that writes
    /// instance data directly on the GPU (e.g. particle-to-instance conversion).
    pub fn instance_buffer(&self) -> &wgpu::Buffer {
        &self.instance_buffer
    }

    /// Returns the current instance buffer capacity (number of instances).
    pub fn instance_capacity(&self) -> u32 {
        self.instance_capacity
    }

    /// Sets the number of instances to draw.
    ///
    /// Use this instead of [`update_instances`](InstanceRenderer::update_instances)
    /// when a compute shader writes instance data directly to the buffer.
    pub fn set_instance_count(&mut self, count: u32) {
        self.instance_count = count;
    }

    /// Ensures the instance buffer has at least `required` capacity.
    ///
    /// Returns `true` if the buffer was reallocated (callers may need to
    /// rebuild bind groups that reference the buffer).
    pub fn ensure_capacity(&mut self, device: &wgpu::Device, required: u32) -> bool {
        if required <= self.instance_capacity {
            return false;
        }
        let instance_stride = std::mem::size_of::<V>() as u64;
        let new_cap = required
            .max(1024)
            .checked_next_power_of_two()
            .unwrap_or(required);
        self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_data_buffer"),
            size: (new_cap as u64) * instance_stride,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        self.instance_capacity = new_cap;
        true
    }

    /// Renders all instances.
    ///
    /// The caller must set the scene bind group at `@group(0)` before calling
    /// this method (via `pass.set_bind_group(0, &scene_bind_group, &[])`).
    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.instance_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..self.instance_count);
    }
}
