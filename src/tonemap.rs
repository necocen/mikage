//! Opt-in exposure and tone mapping for app-owned linear HDR textures.

use crate::{GpuContext, GpuInitError, RenderTarget, RenderTargetConfig};
use wgpu::util::DeviceExt;

/// Portable linear floating-point intermediate, with optional MSAA and depth
/// supplied by [`OffscreenTarget`](crate::OffscreenTarget).
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Operator applied after exposure and before output encoding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ToneMapOperator {
    /// Per-channel `x / (1 + x)`, evaluated as `1 - 1 / (1 + x)` for highlights.
    #[default]
    Reinhard,
    /// Exposure only. Unorm destinations then clamp, like the direct default path.
    None,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ToneMapSettings {
    /// Unitless gain `2^exposure_ev`. Zero is unity; +1 doubles radiance.
    /// Non-finite values become zero; finite EV is clamped to [-32, 32].
    pub exposure_ev: f32,
    pub operator: ToneMapOperator,
}

/// Encoding of the final image, independent of whether the destination view
/// performs hardware sRGB encoding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ToneMapOutput {
    /// Display sRGB. An sRGB view encodes in hardware; other views encode in WGSL.
    #[default]
    Srgb,
    /// Linear output for further processing. Requires a non-sRGB destination.
    Linear,
}

fn exposure(ev: f32) -> f32 {
    if ev.is_finite() {
        ev.clamp(-32.0, 32.0)
    } else {
        0.0
    }
}

/// A reusable, single-sample linear HDR input view bound to one [`ToneMapper`].
/// Recreate this binding when the source texture is resized or replaced.
pub struct ToneMapSource {
    bind_group: wgpu::BindGroup,
}

/// Full-screen exposure/tone-map pass. Default application rendering is unchanged.
///
/// Inputs are **linear, premultiplied/composited RGBA**, as produced by standard
/// renderer alpha blending. Transparent geometry must blend into the HDR target
/// before mapping. RGB is unpremultiplied, exposed/mapped, then premultiplied
/// again; alpha is preserved (clamped to [0, 1]). Opaque scene clears are typical.
/// The source view must be a non-sRGB, single-sample float texture with
/// `TEXTURE_BINDING` usage. No filtering sampler or optional GPU feature is used.
pub struct ToneMapper {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    parameters: wgpu::Buffer,
}

impl ToneMapper {
    /// Creates a display-sRGB mapper for the destination's format and MSAA count.
    pub fn new(gpu: &GpuContext, target: RenderTargetConfig) -> Result<Self, GpuInitError> {
        Self::with_output(gpu, target, ToneMapOutput::Srgb)
    }

    /// Selects display or linear output explicitly. Integer destinations and
    /// linear output through an sRGB view are rejected before pipeline creation.
    pub fn with_output(
        gpu: &GpuContext,
        target: RenderTargetConfig,
        output: ToneMapOutput,
    ) -> Result<Self, GpuInitError> {
        gpu.validate_render_target(target)?;
        if !matches!(
            target
                .color_format
                .sample_type(None, Some(gpu.device.features())),
            Some(wgpu::TextureSampleType::Float { .. })
        ) || (output == ToneMapOutput::Linear && target.color_format.is_srgb())
        {
            return Err(GpuInitError::InvalidTargetFormat(target.color_format));
        }
        let device = &gpu.device;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tone_map_input"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                crate::uniform_buffer_entry(1, wgpu::ShaderStages::FRAGMENT),
            ],
        });
        let encode = f32::from(output == ToneMapOutput::Srgb && !target.color_format.is_srgb());
        let parameters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tone_map_settings"),
            contents: bytemuck::bytes_of(&[0.0_f32, 0.0, encode, 0.0]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tone_map"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../assets/shaders/tonemap.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tone_map"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tone_map"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target.color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: target.multisample_state(),
            multiview_mask: None,
            cache: None,
        });
        Ok(Self {
            pipeline,
            layout,
            parameters,
        })
    }

    /// Binds an app-owned resolved linear input. Updates to this mapper's
    /// settings affect every source created by it without rebuilding bindings.
    pub fn source(&self, gpu: &GpuContext, view: &wgpu::TextureView) -> ToneMapSource {
        ToneMapSource {
            bind_group: gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tone_map_source"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.parameters.as_entire_binding(),
                    },
                ],
            }),
        }
    }

    /// Changes exposure/operator without touching light balance, camera, or time.
    pub fn update(&self, queue: &wgpu::Queue, settings: ToneMapSettings) {
        queue.write_buffer(
            &self.parameters,
            0,
            bytemuck::bytes_of(&[
                exposure(settings.exposure_ev),
                match settings.operator {
                    ToneMapOperator::Reinhard => 0.0_f32,
                    ToneMapOperator::None => 1.0,
                },
            ]),
        );
    }

    /// Changes only EV, retaining the selected operator and output encoding.
    pub fn update_exposure(&self, queue: &wgpu::Queue, ev: f32) {
        queue.write_buffer(&self.parameters, 0, bytemuck::bytes_of(&exposure(ev)));
    }

    /// Draws into an app-owned color-only pass matching the constructor target.
    /// Keep GUI and other display-space overlays after this pass.
    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, source: &'a ToneMapSource) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &source.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Encodes a complete color-only output pass, including MSAA resolve.
    /// `target` must match the constructor's color format and sample count.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &ToneMapSource,
        target: RenderTarget<'_>,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tone_map_output"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target.view,
                resolve_target: target.resolve_target,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        self.render(&mut pass, source);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exposure_is_finite_bounded_and_unity_by_default() {
        assert_eq!(exposure(ToneMapSettings::default().exposure_ev), 0.0);
        for ev in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(exposure(ev), 0.0);
        }
        assert_eq!(exposure(100.0), 32.0);
        assert_eq!(exposure(-100.0), -32.0);
        assert_eq!(exposure(-1.0), -1.0);
    }
}
