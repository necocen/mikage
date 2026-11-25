//! GPU-instanced rendering for large numbers of identical meshes.
//!
//! [`InstanceRenderer`] draws a single base mesh many times, each with
//! per-instance data provided via a type implementing [`InstanceVertex`].
//! The default vertex type [`InstanceData`] provides position, scale, and color.
//!
//! # Example (default vertex)
//!
//! ```no_run
//! use mikage::{
//!     GpuContext, InstanceData, InstanceRenderer, InstanceRendererConfig, RegularPolygonMesh,
//!     RenderTargetConfig, SceneBinding, wgpu,
//! };
//! # fn example(
//! #     gpu: &GpuContext,
//! #     target: RenderTargetConfig,
//! #     scene: &SceneBinding,
//! #     instances: &[InstanceData],
//! #     encoder: &mut wgpu::CommandEncoder,
//! #     view: &wgpu::TextureView,
//! # ) {
//! let hex = RegularPolygonMesh::generate(6);
//! let mut renderer = InstanceRenderer::new(
//!     gpu,
//!     target,
//!     scene.layout(),
//!     &hex.positions,
//!     &hex.normals,
//!     &hex.indices,
//!     InstanceRendererConfig::default_2d(),
//! );
//!
//! renderer.update_instances(gpu, instances);
//!
//! // In render pass (scene bind group already set at group 0):
//! # let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
//! #     color_attachments: &[Some(wgpu::RenderPassColorAttachment {
//! #         view,
//! #         resolve_target: None,
//! #         depth_slice: None,
//! #         ops: wgpu::Operations::default(),
//! #     })],
//! #     ..Default::default()
//! # });
//! pass.set_bind_group(0, scene.bind_group(), &[]);
//! renderer.render(&mut pass);
//! # }
//! ```
//!
//! # Custom vertex types
//!
//! Implement [`InstanceVertex`] for your own type and use
//! [`InstanceRenderer::with_shader`]:
//!
//! ```no_run
//! use bytemuck::{Pod, Zeroable};
//! use mikage::{
//!     GpuContext, InstanceRenderer, InstanceRendererConfig, InstanceVertex, RegularPolygonMesh,
//!     RenderTargetConfig, SceneBinding, wgpu,
//! };
//!
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
//! # fn create(
//! #     gpu: &GpuContext,
//! #     target: RenderTargetConfig,
//! #     scene: &SceneBinding,
//! #     resolved_shader_source: &str,
//! # ) -> InstanceRenderer<TileInstance> {
//! let mesh = RegularPolygonMesh::generate(4);
//! InstanceRenderer::<TileInstance>::with_shader(
//!     gpu,
//!     target,
//!     scene.layout(),
//!     &mesh.positions,
//!     &mesh.normals,
//!     &mesh.indices,
//!     resolved_shader_source,
//!     InstanceRendererConfig::default_2d(),
//! )
//! # }
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
pub struct InstanceRendererConfig {
    /// Vertex shader entry point name.
    pub vertex_entry: &'static str,
    /// Fragment shader entry point name.
    pub fragment_entry: &'static str,
    /// Enable depth testing/writing.
    pub depth: bool,
    /// Add `STORAGE` usage to the instance buffer so compute shaders can write to it.
    /// Disable this for WebGL2 compatibility. Default: `false`.
    pub storage_binding: bool,
}

impl InstanceRendererConfig {
    /// Configuration for 3D rendering: lit shading, depth enabled.
    ///
    /// Uses the built-in instancing shader entry points (`vertex` / `fragment_lit`).
    pub fn default_3d() -> InstanceRendererConfig {
        InstanceRendererConfig {
            vertex_entry: "vertex",
            fragment_entry: "fragment_lit",
            depth: true,
            storage_binding: false,
        }
    }

    /// Configuration for 2D rendering: unlit shading, no depth.
    ///
    /// Uses the built-in instancing shader entry points (`vertex` / `fragment_unlit`).
    pub fn default_2d() -> InstanceRendererConfig {
        InstanceRendererConfig {
            vertex_entry: "vertex",
            fragment_entry: "fragment_unlit",
            depth: false,
            storage_binding: false,
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
    instance_buffer_usage: wgpu::BufferUsages,
    instance_capacity: u32,
    instance_count: u32,
    shadows: Option<InstanceShadowPipelines>,
    shadow_options: crate::ShadowOptions,
    _phantom: PhantomData<V>,
}

struct InstanceShadowPipelines {
    caster: wgpu::RenderPipeline,
    receiver: Option<wgpu::RenderPipeline>,
    bind_group: wgpu::BindGroup,
}

const INITIAL_INSTANCE_CAPACITY: u32 = 1024;

/// Result of [`InstanceRenderer::prepare_compute`].
pub struct ComputeBufferState<'a> {
    /// The instance buffer to bind in compute shader bind groups.
    pub buffer: &'a wgpu::Buffer,
    /// `true` if the buffer was reallocated. Callers must rebuild any
    /// bind groups that reference the instance buffer when this is `true`.
    pub reallocated: bool,
}

/// Builder for [`InstanceRenderer`].
///
/// Unifies built-in 2D/3D, colored-light, shadowed, compute-storage, and custom-shader
/// construction without multi-argument constructor combinations.
///
/// ```no_run
/// use mikage::{
///     ColoredSceneBinding, DirectionalShadow, GpuContext, IcoSphereMesh, InstanceRenderer,
///     RenderTargetConfig,
/// };
/// # fn create(
/// #     gpu: &GpuContext,
/// #     target: RenderTargetConfig,
/// #     scene: &ColoredSceneBinding,
/// #     shadow: &DirectionalShadow,
/// # ) -> InstanceRenderer {
/// let sphere = IcoSphereMesh::generate(2);
/// InstanceRenderer::builder(
///     target,
///     scene,
///     &sphere.positions,
///     &sphere.normals,
///     &sphere.indices,
/// )
/// .with_storage_binding(true)
/// .with_shadows(shadow)
/// .build(gpu)
/// # }
/// ```
pub struct InstanceRendererBuilder<'a, V: InstanceVertex = InstanceData> {
    target: crate::RenderTargetConfig,
    scene: crate::RendererScene<'a>,
    positions: &'a [[f32; 3]],
    normals: &'a [[f32; 3]],
    indices: &'a [u32],
    config: InstanceRendererConfig,
    shader_source: Option<&'a str>,
    shadows: Option<&'a crate::DirectionalShadow>,
    _phantom: PhantomData<V>,
}

impl<'a, V: InstanceVertex> InstanceRendererBuilder<'a, V> {
    /// Replaces the renderer configuration (for example `InstanceRendererConfig::default_2d()`).
    pub fn with_config(mut self, config: InstanceRendererConfig) -> Self {
        self.config = config;
        self
    }

    /// Enables or disables depth testing and depth writes.
    pub fn with_depth(mut self, depth: bool) -> Self {
        self.config.depth = depth;
        self
    }

    /// Adds `STORAGE` usage to the instance buffer for GPU compute writes.
    pub fn with_storage_binding(mut self, storage_binding: bool) -> Self {
        self.config.storage_binding = storage_binding;
        self
    }

    /// Overrides the vertex and fragment shader entry points.
    pub fn with_entry_points(
        mut self,
        vertex_entry: &'static str,
        fragment_entry: &'static str,
    ) -> Self {
        self.config.vertex_entry = vertex_entry;
        self.config.fragment_entry = fragment_entry;
        self
    }

    /// Builds the [`InstanceRenderer`].
    pub fn build(self, gpu: &crate::GpuContext) -> InstanceRenderer<V> {
        if let Some(shader_source) = self.shader_source {
            return InstanceRenderer::create(
                gpu,
                self.target,
                self.scene.layout(),
                self.positions,
                self.normals,
                self.indices,
                shader_source,
                self.config,
                None,
            );
        }

        let mut config = self.config;
        if self.scene.is_colored() && config.fragment_entry == "fragment_lit" {
            config.fragment_entry = "fragment_colored";
        }
        let shadows_supported = self
            .shadows
            .is_some_and(crate::DirectionalShadow::is_supported);
        let mut source = SHADER_SOURCE.to_owned();
        if config.fragment_entry == "fragment_colored" || shadows_supported {
            source.push_str(include_str!(
                "../assets/shaders/instancing_light_colors.wgsl"
            ));
        }
        if shadows_supported {
            source.push_str(include_str!("../assets/shaders/instancing_shadows.wgsl"));
        }
        let resolved = mikage_shader_processor()
            .resolve(&source)
            .expect("failed to resolve instancing shader imports");
        InstanceRenderer::create(
            gpu,
            self.target,
            self.scene.layout(),
            self.positions,
            self.normals,
            self.indices,
            &resolved,
            config,
            self.shadows,
        )
    }
}

impl<'a> InstanceRendererBuilder<'a, InstanceData> {
    /// Enables directional shadow casting and receiving using the given shadow map.
    ///
    /// Applies to built-in instancing shaders created via [`InstanceRenderer::builder`].
    /// Unsupported shadow sampling falls back to ordinary lighting.
    pub fn with_shadows(mut self, shadows: &'a crate::DirectionalShadow) -> Self {
        self.shadows = Some(shadows);
        self
    }
}

impl InstanceRenderer<InstanceData> {
    /// Starts a builder for [`InstanceRenderer`] using the built-in instancing shader.
    ///
    /// Defaults to [`InstanceRendererConfig::default_3d()`]. Accepts `&SceneBinding`,
    /// `&ColoredSceneBinding`, or `&wgpu::BindGroupLayout`; passing `&ColoredSceneBinding`
    /// automatically selects colored lighting for lit entry points.
    pub fn builder<'a>(
        target: crate::RenderTargetConfig,
        scene: impl Into<crate::RendererScene<'a>>,
        positions: &'a [[f32; 3]],
        normals: &'a [[f32; 3]],
        indices: &'a [u32],
    ) -> InstanceRendererBuilder<'a, InstanceData> {
        InstanceRendererBuilder {
            target,
            scene: scene.into(),
            positions,
            normals,
            indices,
            config: InstanceRendererConfig::default_3d(),
            shader_source: None,
            shadows: None,
            _phantom: PhantomData,
        }
    }

    /// Reuses the current instance buffer (including GPU-written data) for
    /// shadows. Alpha below one is discarded in the caster fragment shader.
    #[allow(clippy::too_many_arguments)]
    pub fn with_shadows(
        gpu: &crate::GpuContext,
        target: crate::RenderTargetConfig,
        scene_layout: &wgpu::BindGroupLayout,
        shadow: &crate::DirectionalShadow,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
        config: InstanceRendererConfig,
    ) -> Self {
        Self::builder(target, scene_layout, positions, normals, indices)
            .with_config(config)
            .with_shadows(shadow)
            .build(gpu)
    }

    /// Composes RGB light colors and shadows. Bind the colored scene at group 0.
    #[allow(clippy::too_many_arguments)]
    pub fn with_light_colors_and_shadows(
        gpu: &crate::GpuContext,
        target: crate::RenderTargetConfig,
        scene: &crate::ColoredSceneBinding,
        shadow: &crate::DirectionalShadow,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
        config: InstanceRendererConfig,
    ) -> Self {
        Self::builder(target, scene, positions, normals, indices)
            .with_config(config)
            .with_shadows(shadow)
            .build(gpu)
    }

    /// Creates an `InstanceRenderer` using the built-in instancing shader.
    ///
    /// This is a convenience constructor for the default [`InstanceData`] layout
    /// (position + scale + color). For custom vertex types, use
    /// [`with_shader`](InstanceRenderer::with_shader) or
    /// [`builder_with_shader`](InstanceRenderer::builder_with_shader).
    pub fn new(
        gpu: &crate::GpuContext,
        target: crate::RenderTargetConfig,
        scene_bind_group_layout: &wgpu::BindGroupLayout,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
        config: InstanceRendererConfig,
    ) -> Self {
        Self::builder(target, scene_bind_group_layout, positions, normals, indices)
            .with_config(config)
            .build(gpu)
    }

    /// Uses independent linear RGB light colors from a [`ColoredSceneBinding`](crate::ColoredSceneBinding).
    ///
    /// `fragment_lit` in `config` selects the colored Lambert entry point;
    /// `fragment_unlit` still passes material RGBA through unchanged. Bind the
    /// colored scene at group 0 before rendering. Instance buffer ownership and
    /// compute access are unchanged.
    pub fn with_light_colors(
        gpu: &crate::GpuContext,
        target: crate::RenderTargetConfig,
        scene: &crate::ColoredSceneBinding,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
        config: InstanceRendererConfig,
    ) -> Self {
        Self::builder(target, scene, positions, normals, indices)
            .with_config(config)
            .build(gpu)
    }
}

impl<V: InstanceVertex> InstanceRenderer<V> {
    /// Starts a builder for [`InstanceRenderer`] using a custom import-resolved WGSL shader.
    ///
    /// Defaults to [`InstanceRendererConfig::default_3d()`]. Accepts `&SceneBinding`,
    /// `&ColoredSceneBinding`, or `&wgpu::BindGroupLayout`.
    pub fn builder_with_shader<'a>(
        target: crate::RenderTargetConfig,
        scene: impl Into<crate::RendererScene<'a>>,
        positions: &'a [[f32; 3]],
        normals: &'a [[f32; 3]],
        indices: &'a [u32],
        shader_source: &'a str,
    ) -> InstanceRendererBuilder<'a, V> {
        InstanceRendererBuilder {
            target,
            scene: scene.into(),
            positions,
            normals,
            indices,
            config: InstanceRendererConfig::default_3d(),
            shader_source: Some(shader_source),
            shadows: None,
            _phantom: PhantomData,
        }
    }

    /// Creates an `InstanceRenderer` with a custom shader and vertex type.
    ///
    /// `shader_source` must be import-resolved WGSL. Use [`ShaderProcessor`](crate::ShaderProcessor)
    /// to resolve `#import` directives before passing the source here.
    ///
    /// The shader must declare vertex attributes matching `V::vertex_attributes()`
    /// at the expected shader locations (starting at 2).
    #[allow(clippy::too_many_arguments)] // Explicit target configuration keeps pipeline compatibility visible.
    pub fn with_shader(
        gpu: &crate::GpuContext,
        target: crate::RenderTargetConfig,
        scene_bind_group_layout: &wgpu::BindGroupLayout,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
        shader_source: &str,
        config: InstanceRendererConfig,
    ) -> Self {
        Self::builder_with_shader(
            target,
            scene_bind_group_layout,
            positions,
            normals,
            indices,
            shader_source,
        )
        .with_config(config)
        .build(gpu)
    }

    #[allow(clippy::too_many_arguments)]
    fn create(
        gpu: &crate::GpuContext,
        target: crate::RenderTargetConfig,
        scene_bind_group_layout: &wgpu::BindGroupLayout,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
        shader_source: &str,
        config: InstanceRendererConfig,
        shadow: Option<&crate::DirectionalShadow>,
    ) -> Self {
        let device = &gpu.device;
        let mesh = MeshBuffers::from_position_normal(device, positions, normals, indices);

        let instance_stride = std::mem::size_of::<V>() as u64;
        let mut instance_usage = wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
        if config.storage_binding {
            instance_usage |= wgpu::BufferUsages::STORAGE;
        }
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_data_buffer"),
            size: (INITIAL_INSTANCE_CAPACITY as u64) * instance_stride,
            usage: instance_usage,
            mapped_at_creation: false,
        });

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("instancing_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let mut layouts = vec![Some(scene_bind_group_layout)];
        if let Some(layout) = shadow.and_then(|s| s.receiver_layout()) {
            layouts.push(Some(layout));
        }
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("instancing_pipeline_layout"),
            bind_group_layouts: &layouts,
            immediate_size: 0,
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
                format: target.depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            })
        } else {
            None
        };

        let mut descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("instancing_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some(config.vertex_entry),
                compilation_options: Default::default(),
                buffers: &[
                    Some(POSITION_NORMAL_LAYOUT),
                    Some(instance_buffer_layout.clone()),
                ],
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
                count: target.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some(config.fragment_entry),
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
        let pipeline = device.create_render_pipeline(&descriptor);
        let shadows = shadow.and_then(|shadow| {
            let resources = shadow.resources.as_ref()?;
            let entry = match config.fragment_entry {
                "fragment_lit" => Some("fragment_shadowed"),
                "fragment_colored" => Some("fragment_shadowed_colored"),
                _ => None,
            };
            let receiver = entry.map(|entry| {
                descriptor.vertex.entry_point = Some("vertex_shadowed");
                descriptor.fragment.as_mut().unwrap().entry_point = Some(entry);
                device.create_render_pipeline(&descriptor)
            });
            Some(InstanceShadowPipelines {
                caster: shadow.caster_pipeline(
                    device,
                    &shader_module,
                    &[Some(POSITION_NORMAL_LAYOUT), Some(instance_buffer_layout)],
                    None,
                    Some("fragment_shadow_depth"),
                )?,
                receiver,
                bind_group: resources.bind_group.clone(),
            })
        });

        Self {
            pipeline,
            vertex_buffer: mesh.vertex_buffer,
            index_buffer: mesh.index_buffer,
            index_count: mesh.index_count,
            instance_buffer,
            instance_buffer_usage: instance_usage,
            instance_capacity: INITIAL_INSTANCE_CAPACITY,
            instance_count: 0,
            shadows,
            shadow_options: crate::ShadowOptions::default(),
            _phantom: PhantomData,
        }
    }

    /// Updates instance data from CPU. Resizes the GPU buffer if needed.
    pub fn update_instances(&mut self, gpu: &crate::GpuContext, instances: &[V]) {
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
            self.instance_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instance_data_buffer"),
                size: (new_cap as u64) * instance_stride,
                usage: self.instance_buffer_usage,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_cap;
        }

        gpu.queue
            .write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));
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

    /// Prepares the renderer for compute-driven rendering.
    ///
    /// Sets the draw count to `count`, grows the buffer if needed, and returns
    /// a reference to the (possibly new) buffer along with whether it was
    /// reallocated. When `reallocated` is `true`, callers **must** rebuild any
    /// bind groups that reference the instance buffer.
    ///
    /// This is the primary API for compute shader integration. Use it instead
    /// of the lower-level [`set_instance_count`](Self::set_instance_count) +
    /// [`ensure_capacity`](Self::ensure_capacity) combination.
    pub fn prepare_compute(&mut self, device: &wgpu::Device, count: u32) -> ComputeBufferState<'_> {
        self.instance_count = count;
        let reallocated = self.ensure_capacity(device, count);
        ComputeBufferState {
            buffer: &self.instance_buffer,
            reallocated,
        }
    }

    /// Sets the number of instances to draw.
    ///
    /// Low-level method for compute pipelines. Prefer
    /// [`prepare_compute`](Self::prepare_compute) which also handles capacity.
    pub fn set_instance_count(&mut self, count: u32) {
        self.instance_count = count;
    }

    /// Ensures the instance buffer has at least `required` capacity.
    ///
    /// Returns `true` if the buffer was reallocated (callers may need to
    /// rebuild bind groups that reference the buffer).
    ///
    /// Low-level method. Prefer [`prepare_compute`](Self::prepare_compute).
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
            usage: self.instance_buffer_usage,
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
        let pipeline = if self.shadow_options.receive {
            self.shadows
                .as_ref()
                .and_then(|s| s.receiver.as_ref())
                .unwrap_or(&self.pipeline)
        } else {
            &self.pipeline
        };
        if let Some(shadows) = &self.shadows {
            pass.set_bind_group(1, &shadows.bind_group, &[]);
        }
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..self.instance_count);
    }

    /// Batch-wide cast/receive selection, retained across instance updates/growth.
    /// Transparent instances are always excluded from casting on the GPU.
    pub fn set_shadow_options(&mut self, options: crate::ShadowOptions) {
        self.shadow_options = options;
    }

    /// Draw the same current GPU instance buffer into `DirectionalShadow::begin_pass`.
    /// No readback, geometry duplication, or simulation work is performed.
    pub fn render_shadow<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if !self.shadow_options.cast || self.instance_count == 0 {
            return;
        }
        let Some(shadows) = &self.shadows else {
            return;
        };
        pass.set_pipeline(&shadows.caster);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..self.instance_count);
    }
}
