//! Single-directional-light shadow map shared by standard renderers.
use crate::{GpuContext, SceneBinding, SceneUniform};
use wgpu::util::DeviceExt;

/// Independent caster/receiver selection. Transparent solids never cast;
/// transparent instances are discarded on the GPU during the depth pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShadowOptions {
    pub cast: bool,
    pub receive: bool,
}
impl Default for ShadowOptions {
    fn default() -> Self {
        Self {
            cast: true,
            receive: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ShadowFilter {
    /// One nearest depth comparison.
    Hard,
    /// Hardware bilinear depth-comparison filtering (2x2 PCF).
    #[default]
    Pcf,
}
/// Explicit light-space bounds must match your light direction. Resolution,
/// raster bias and filtering are fixed at creation; matrix/enable can change.
#[derive(Debug, Clone, Copy)]
pub struct DirectionalShadowConfig {
    pub resolution: u32,
    /// World to light clip space: X/Y [-1, 1], Z [0, 1].
    pub light_view_projection: glam::Mat4,
    pub depth_bias: i32,
    pub slope_bias: f32,
    pub filter: ShadowFilter,
    pub enabled: bool,
}
impl Default for DirectionalShadowConfig {
    fn default() -> Self {
        Self {
            resolution: 1024,
            light_view_projection: glam::Mat4::IDENTITY,
            depth_bias: 2,
            slope_bias: 2.0,
            filter: ShadowFilter::Pcf,
            enabled: true,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShadowError(pub &'static str);
impl std::fmt::Display for ShadowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for ShadowError {}

pub(crate) struct ShadowResources {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) bind_group: wgpu::BindGroup,
    parameters: wgpu::Buffer,
    light_scene: SceneBinding,
}
/// Single-sample Depth32Float map. Use `begin_pass` and renderer `render_shadow`
/// calls before normal rendering. Preparation and updates never advance ticks.
/// Metal/Vulkan/DX12 and compliant WebGPU need no optional device features.
/// Unsupported sampling (including WebGL) produces an unshadowed fallback:
/// `is_supported` is false and shadow constructors use normal lighting pipelines.
pub struct DirectionalShadow {
    pub(crate) resources: Option<ShadowResources>,
    enabled: bool,
    bias: wgpu::DepthBiasState,
}
impl DirectionalShadow {
    pub fn new(gpu: &GpuContext, config: DirectionalShadowConfig) -> Result<Self, ShadowError> {
        if config.resolution == 0
            || config.resolution > gpu.device.limits().max_texture_dimension_2d
        {
            return Err(ShadowError(
                "shadow resolution must be nonzero and within device texture limits",
            ));
        }
        if !config.light_view_projection.is_finite() || !config.slope_bias.is_finite() {
            return Err(ShadowError("shadow matrix and slope bias must be finite"));
        }
        let bias = wgpu::DepthBiasState {
            constant: config.depth_bias,
            slope_scale: config.slope_bias,
            clamp: 0.0,
        };
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        if !gpu.supports_shadow_sampling()
            || !gpu
                .format_features(crate::DEPTH_FORMAT)
                .allowed_usages
                .contains(usage)
        {
            return Ok(Self {
                resources: None,
                enabled: false,
                bias,
            });
        }
        let device = &gpu.device;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("directional_shadow"),
            size: wgpu::Extent3d {
                width: config.resolution,
                height: config.resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::DEPTH_FORMAT,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let filter = match config.filter {
            ShadowFilter::Hard => wgpu::FilterMode::Nearest,
            ShadowFilter::Pcf => wgpu::FilterMode::Linear,
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_comparison"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            min_filter: filter,
            mag_filter: filter,
            ..Default::default()
        });
        let parameters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shadow_matrix"),
            contents: bytemuck::bytes_of(&ShadowUniform {
                matrix: config.light_view_projection.to_cols_array_2d(),
                flags: [f32::from(config.enabled), 0.0, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_receiver_layout"),
            entries: &[
                crate::uniform_buffer_entry(0, wgpu::ShaderStages::FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_receiver"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: parameters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let light_scene = SceneBinding::new(device);
        light_scene.update(
            &gpu.queue,
            &SceneUniform::new(config.light_view_projection, glam::Vec3::ZERO),
        );
        Ok(Self {
            resources: Some(ShadowResources {
                _texture: texture,
                view,
                layout,
                bind_group,
                parameters,
                light_scene,
            }),
            enabled: config.enabled,
            bias,
        })
    }
    pub fn is_supported(&self) -> bool {
        self.resources.is_some()
    }
    pub fn is_enabled(&self) -> bool {
        self.enabled && self.is_supported()
    }
    /// Enabling requires a fresh caster pass before the next receiver draw.
    pub fn set_enabled(&mut self, queue: &wgpu::Queue, enabled: bool) {
        self.enabled = enabled;
        if let Some(r) = &self.resources {
            queue.write_buffer(
                &r.parameters,
                64,
                bytemuck::bytes_of(&[f32::from(enabled), 0.0, 0.0, 0.0]),
            );
        }
    }
    /// Change only light-space bounds/camera, without a simulation step.
    pub fn update_matrix(
        &self,
        queue: &wgpu::Queue,
        matrix: glam::Mat4,
    ) -> Result<(), ShadowError> {
        if !matrix.is_finite() {
            return Err(ShadowError("shadow matrix must be finite"));
        }
        if let Some(r) = &self.resources {
            queue.write_buffer(
                &r.parameters,
                0,
                bytemuck::bytes_of(&matrix.to_cols_array_2d()),
            );
            r.light_scene
                .update(queue, &SceneUniform::new(matrix, glam::Vec3::ZERO));
        }
        Ok(())
    }
    /// Clear and begin the depth pass; group 0 is supplied. End this pass before
    /// sampling the map in normal rendering. Disabled/unsupported returns None.
    pub fn begin_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
    ) -> Option<wgpu::RenderPass<'a>> {
        if !self.is_enabled() {
            return None;
        }
        let r = self.resources.as_ref()?;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow_casters"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &r.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_bind_group(0, r.light_scene.bind_group(), &[]);
        Some(pass)
    }
    /// Shared receiver resources: solid pipelines use group 2, instances group 1.
    pub fn receiver_layout(&self) -> Option<&wgpu::BindGroupLayout> {
        self.resources.as_ref().map(|r| &r.layout)
    }
    pub fn receiver_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.resources.as_ref().map(|r| &r.bind_group)
    }
    pub(crate) fn caster_pipeline(
        &self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
        model: Option<&wgpu::BindGroupLayout>,
        depth_fragment: Option<&str>,
    ) -> Option<wgpu::RenderPipeline> {
        self.resources.as_ref().map(|r| {
            let mut layouts = vec![Some(r.light_scene.layout())];
            if let Some(model) = model {
                layouts.push(Some(model));
            }
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_caster"),
                bind_group_layouts: &layouts,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shadow_caster"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vertex"),
                    buffers,
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: crate::DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: self.bias,
                }),
                multisample: Default::default(),
                fragment: depth_fragment.map(|entry| wgpu::FragmentState {
                    module: shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[],
                }),
                multiview_mask: None,
                cache: None,
            })
        })
    }
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ShadowUniform {
    matrix: [[f32; 4]; 4],
    flags: [f32; 4],
}
