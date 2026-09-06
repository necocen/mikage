use dpi::PhysicalSize;
use std::sync::Arc;

/// Format and multisampling contract for a particular render target.
/// A GPU can serve several targets with different formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderTargetConfig {
    pub color_format: wgpu::TextureFormat,
    pub depth_format: wgpu::TextureFormat,
    pub sample_count: u32,
}
impl Default for RenderTargetConfig {
    fn default() -> Self {
        Self {
            color_format: wgpu::TextureFormat::Rgba8UnormSrgb,
            depth_format: crate::DEPTH_FORMAT,
            sample_count: 1,
        }
    }
}
impl RenderTargetConfig {
    pub fn color_target_state(&self, blend: wgpu::BlendState) -> wgpu::ColorTargetState {
        wgpu::ColorTargetState {
            format: self.color_format,
            blend: Some(blend),
            write_mask: wgpu::ColorWrites::ALL,
        }
    }
    pub fn multisample_state(&self) -> wgpu::MultisampleState {
        wgpu::MultisampleState {
            count: self.sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        }
    }
    pub fn depth_stencil_state(&self) -> wgpu::DepthStencilState {
        wgpu::DepthStencilState {
            format: self.depth_format,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }
    }
}

/// Hard requirements and opportunistic improvements for a logical device.
#[derive(Debug, Clone, Default)]
pub struct GpuRequirements {
    pub required_features: wgpu::Features,
    pub optional_features: wgpu::Features,
    /// None selects the backend's portable baseline with adapter-sized texture
    /// dimensions. Explicit limits are never silently weakened.
    pub required_limits: Option<wgpu::Limits>,
    /// Clamped to adapter support without weakening required limits.
    pub preferred_limits: Option<wgpu::Limits>,
}

/// An owned display connection, such as winit's OwnedDisplayHandle.
pub trait GpuDisplayHandle: wgpu::rwh::HasDisplayHandle + std::fmt::Debug + Send + Sync {}
impl<T: wgpu::rwh::HasDisplayHandle + std::fmt::Debug + Send + Sync> GpuDisplayHandle for T {}

/// Shared selection options for windowed, external-surface and headless use.
#[derive(Debug, Clone)]
pub struct GpuDescriptor {
    /// None uses native primary backends, or WebGPU with optional WebGL fallback.
    pub backends: Option<wgpu::Backends>,
    pub power_preference: wgpu::PowerPreference,
    pub force_fallback_adapter: bool,
    /// GLES presentation requires the platform display connection, especially
    /// on Wayland. All attached surfaces must use this same connection.
    pub display: Option<Arc<dyn GpuDisplayHandle>>,
    pub requirements: GpuRequirements,
}
impl Default for GpuDescriptor {
    fn default() -> Self {
        Self {
            backends: None,
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            display: None,
            requirements: GpuRequirements::default(),
        }
    }
}

/// Hardware availability and the enabled device contract, available to factories.
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    pub adapter_info: wgpu::AdapterInfo,
    pub supported_features: wgpu::Features,
    pub supported_limits: wgpu::Limits,
    pub enabled_features: wgpu::Features,
    pub enabled_limits: wgpu::Limits,
    /// Nanoseconds per tick, present only when timestamp queries are enabled.
    pub timestamp_period: Option<f32>,
}

/// Presentation options independent of the host's window framework.
#[derive(Debug, Clone)]
pub struct SurfaceDescriptor {
    pub size: PhysicalSize<u32>,
    pub present_mode: wgpu::PresentMode,
    /// None prefers sRGB, including an sRGB view of a linear surface. An explicit
    /// format is used unchanged after validation.
    pub format: Option<wgpu::TextureFormat>,
    /// None chooses the surface's first supported mode.
    pub alpha_mode: Option<wgpu::CompositeAlphaMode>,
    pub depth_format: wgpu::TextureFormat,
    pub sample_count: u32,
    pub desired_maximum_frame_latency: u32,
}
impl Default for SurfaceDescriptor {
    fn default() -> Self {
        Self {
            size: PhysicalSize::new(0, 0),
            present_mode: wgpu::PresentMode::AutoVsync,
            format: None,
            alpha_mode: None,
            depth_format: crate::DEPTH_FORMAT,
            sample_count: 1,
            desired_maximum_frame_latency: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuLimitFailure {
    pub name: &'static str,
    pub required: u64,
    pub supported: u64,
}

/// Errors detected while selecting a GPU or creating a render target.
#[derive(Debug)]
pub enum GpuInitError {
    NoBackends,
    AdapterUnavailable(wgpu::RequestAdapterError),
    MissingFeatures(wgpu::Features),
    UnsupportedLimits(Vec<GpuLimitFailure>),
    RequestDevice(wgpu::RequestDeviceError),
    CreateSurface(wgpu::CreateSurfaceError),
    IncompatibleSurface,
    NoSurfaceFormats,
    NoSurfaceAlphaModes,
    UnsupportedSurfaceFormat(wgpu::TextureFormat),
    UnsupportedPresentMode(wgpu::PresentMode),
    UnsupportedAlphaMode(wgpu::CompositeAlphaMode),
    InvalidSampleCount(u32),
    InvalidTargetFormat(wgpu::TextureFormat),
    UnsupportedTextureUsage {
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    },
    UnsupportedSampleCount {
        format: wgpu::TextureFormat,
        sample_count: u32,
    },
    InvalidSize(PhysicalSize<u32>),
    SizeExceedsLimit {
        size: PhysicalSize<u32>,
        limit: u32,
    },
    WrongDevice,
}
impl std::fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoBackends => write!(f, "no requested GPU backend is enabled in this build"),
            Self::AdapterUnavailable(e) => write!(f, "no compatible GPU adapter: {e}"),
            Self::MissingFeatures(features) => {
                write!(f, "GPU is missing required features: {features:?}")
            }
            Self::UnsupportedLimits(failures) => {
                write!(f, "GPU does not satisfy required limits")?;
                for e in failures {
                    write!(
                        f,
                        "; {}: requested {}, supported {}",
                        e.name, e.required, e.supported
                    )?;
                }
                Ok(())
            }
            Self::RequestDevice(e) => write!(f, "failed to create device: {e}"),
            Self::CreateSurface(e) => write!(f, "failed to create surface: {e}"),
            Self::IncompatibleSurface => {
                write!(f, "surface is incompatible with the selected adapter")
            }
            Self::NoSurfaceFormats => write!(f, "surface reported no supported formats"),
            Self::NoSurfaceAlphaModes => write!(f, "surface reported no supported alpha modes"),
            Self::UnsupportedSurfaceFormat(v) => write!(f, "surface does not support {v:?}"),
            Self::UnsupportedPresentMode(v) => {
                write!(f, "surface does not support present mode {v:?}")
            }
            Self::UnsupportedAlphaMode(v) => write!(f, "surface does not support alpha mode {v:?}"),
            Self::InvalidSampleCount(v) => write!(f, "sample_count must be 1 or 4, got {v}"),
            Self::InvalidTargetFormat(v) => write!(f, "invalid render target format: {v:?}"),
            Self::UnsupportedTextureUsage { format, usage } => {
                write!(f, "{format:?} does not support usage {usage:?}")
            }
            Self::UnsupportedSampleCount {
                format,
                sample_count,
            } => write!(f, "{format:?} does not support {sample_count} samples"),
            Self::InvalidSize(s) => write!(
                f,
                "offscreen size must be nonzero, got {}x{}",
                s.width, s.height
            ),
            Self::SizeExceedsLimit { size, limit } => write!(
                f,
                "size {}x{} exceeds device dimension limit {limit}",
                size.width, size.height
            ),
            Self::WrongDevice => write!(f, "surface belongs to a different GPU device"),
        }
    }
}
impl std::error::Error for GpuInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AdapterUnavailable(e) => Some(e),
            Self::RequestDevice(e) => Some(e),
            Self::CreateSurface(e) => Some(e),
            _ => None,
        }
    }
}

/// A logical GPU independent of presentation and offscreen targets.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    capabilities: GpuCapabilities,
}
impl GpuContext {
    /// Creates a GPU without a surface or event loop.
    pub async fn headless(descriptor: GpuDescriptor) -> Result<Self, GpuInitError> {
        let instance = create_instance(&descriptor).await?;
        Self::initialize(instance, None, &descriptor).await
    }
    /// Selects a GPU for a safe surface target. Borrowed targets retain their
    /// lifetime; owned targets such as Arc<Window> can produce a static surface.
    ///
    /// ```compile_fail
    /// use mikage::{GpuContext, SurfaceContext, wgpu};
    /// fn needs_static(_: SurfaceContext<'static>) {}
    /// async fn borrowed(target: wgpu::SurfaceTarget<'_>) {
    ///     let (_, surface) = GpuContext::for_surface(
    ///         target, Default::default(), Default::default(),
    ///     ).await.unwrap();
    ///     needs_static(surface); // A borrowed target cannot become static.
    /// }
    /// ```
    pub async fn for_surface<'s>(
        target: impl Into<wgpu::SurfaceTarget<'s>>,
        descriptor: GpuDescriptor,
        surface_descriptor: SurfaceDescriptor,
    ) -> Result<(Self, SurfaceContext<'s>), GpuInitError> {
        let instance = create_instance(&descriptor).await?;
        let surface = instance
            .create_surface(target)
            .map_err(GpuInitError::CreateSurface)?;
        let gpu = Self::initialize(instance, Some(&surface), &descriptor).await?;
        let surface = SurfaceContext::new(surface, &gpu, surface_descriptor)?;
        Ok((gpu, surface))
    }
    /// Creates a GPU for a host-owned raw target such as a CAMetalLayer.
    ///
    /// # Safety
    /// The caller must uphold the supplied SurfaceTargetUnsafe variant's
    /// requirements. All native objects must remain valid throughout async
    /// initialization and for the entire lifetime of the returned surface and
    /// acquired textures. The host coordinates rendering, resize and teardown,
    /// releasing textures and the surface before destroying native objects.
    /// Mikage does not take ownership of, or create a window for, the raw target.
    pub async unsafe fn for_surface_unsafe<'s>(
        target: wgpu::SurfaceTargetUnsafe,
        descriptor: GpuDescriptor,
        surface_descriptor: SurfaceDescriptor,
    ) -> Result<(Self, SurfaceContext<'s>), GpuInitError> {
        let instance = create_instance(&descriptor).await?;
        // SAFETY: The caller guarantees native target validity and lifetime.
        let surface = unsafe { instance.create_surface_unsafe(target) }
            .map_err(GpuInitError::CreateSurface)?;
        let gpu = Self::initialize(instance, Some(&surface), &descriptor).await?;
        let surface = SurfaceContext::new(surface, &gpu, surface_descriptor)?;
        Ok((gpu, surface))
    }
    /// Attaches a new surface, including after surface loss. Existing resources
    /// and the selected adapter are retained; incompatible targets return errors.
    pub fn attach_surface<'s>(
        &self,
        target: impl Into<wgpu::SurfaceTarget<'s>>,
        descriptor: SurfaceDescriptor,
    ) -> Result<SurfaceContext<'s>, GpuInitError> {
        let surface = self
            .instance
            .create_surface(target)
            .map_err(GpuInitError::CreateSurface)?;
        SurfaceContext::new(surface, self, descriptor)
    }
    /// Attaches a host-owned raw target to the existing device.
    ///
    /// # Safety
    /// Uphold the target variant's requirements and keep all native objects
    /// valid until the returned surface and acquired textures are released.
    /// The host retains ownership and serializes rendering, resize and teardown.
    pub unsafe fn attach_surface_unsafe<'s>(
        &self,
        target: wgpu::SurfaceTargetUnsafe,
        descriptor: SurfaceDescriptor,
    ) -> Result<SurfaceContext<'s>, GpuInitError> {
        // SAFETY: The caller guarantees native target validity and lifetime.
        let surface = unsafe { self.instance.create_surface_unsafe(target) }
            .map_err(GpuInitError::CreateSurface)?;
        SurfaceContext::new(surface, self, descriptor)
    }
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.capabilities.adapter_info
    }
    pub fn capabilities(&self) -> &GpuCapabilities {
        &self.capabilities
    }
    /// Checks a pipeline/target contract against this device without allocating
    /// textures. Useful before invoking an offscreen application factory.
    pub fn validate_render_target(&self, target: RenderTargetConfig) -> Result<(), GpuInitError> {
        validate_target(self, target)
    }
    /// Clones handles to the same logical device and queue for GPU-aware workers.
    pub fn compute_handles(&self) -> (wgpu::Device, wgpu::Queue) {
        (self.device.clone(), self.queue.clone())
    }
    async fn initialize(
        instance: wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
        descriptor: &GpuDescriptor,
    ) -> Result<Self, GpuInitError> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: descriptor.power_preference,
                compatible_surface: surface,
                force_fallback_adapter: descriptor.force_fallback_adapter,
                ..Default::default()
            })
            .await
            .map_err(GpuInitError::AdapterUnavailable)?;
        let adapter_info = adapter.get_info();
        let supported_features = adapter.features();
        let supported_limits = adapter.limits();
        let (features, limits) = resolve_requirements(
            &descriptor.requirements,
            supported_features,
            &supported_limits,
            adapter_info.backend,
        )?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("mikage_device"),
                required_features: features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                ..Default::default()
            })
            .await
            .map_err(GpuInitError::RequestDevice)?;
        let enabled_features = device.features();
        let capabilities = GpuCapabilities {
            adapter_info,
            supported_features,
            supported_limits,
            enabled_features,
            enabled_limits: device.limits(),
            timestamp_period: enabled_features
                .contains(wgpu::Features::TIMESTAMP_QUERY)
                .then(|| queue.get_timestamp_period()),
        };
        tracing::info!("GPU adapter: {:?}", capabilities.adapter_info);
        Ok(Self {
            device,
            queue,
            instance,
            adapter,
            capabilities,
        })
    }
    fn format_features(&self, format: wgpu::TextureFormat) -> wgpu::TextureFormatFeatures {
        if self
            .device
            .features()
            .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        {
            self.adapter.get_texture_format_features(format)
        } else {
            format.guaranteed_format_features(self.device.features())
        }
    }
}

/// Presentation state preserving the lifetime of the native target.
pub struct SurfaceContext<'surface> {
    surface: wgpu::Surface<'surface>,
    device: wgpu::Device,
    descriptor: SurfaceDescriptor,
    configuration: wgpu::SurfaceConfiguration,
    render_target: RenderTargetConfig,
    msaa_view: Option<wgpu::TextureView>,
}
impl<'s> SurfaceContext<'s> {
    fn new(
        surface: wgpu::Surface<'s>,
        gpu: &GpuContext,
        descriptor: SurfaceDescriptor,
    ) -> Result<Self, GpuInitError> {
        if !gpu.adapter.is_surface_supported(&surface) {
            return Err(GpuInitError::IncompatibleSurface);
        }
        let caps = surface.get_capabilities(&gpu.adapter);
        let (configuration, render_target) = surface_configuration(&caps, &descriptor)?;
        validate_target(gpu, render_target)?;
        validate_size(
            descriptor.size,
            gpu.device.limits().max_texture_dimension_2d,
            true,
        )?;
        let mut result = Self {
            surface,
            device: gpu.device.clone(),
            descriptor,
            configuration,
            render_target,
            msaa_view: None,
        };
        result.reconfigure(gpu)?;
        Ok(result)
    }
    /// Zero in either dimension suspends presentation. All acquired textures
    /// must be presented or dropped first. Serialize configuration with submits.
    pub fn resize(
        &mut self,
        gpu: &GpuContext,
        size: PhysicalSize<u32>,
    ) -> Result<(), GpuInitError> {
        self.check_device(gpu)?;
        validate_size(size, gpu.device.limits().max_texture_dimension_2d, true)?;
        self.descriptor.size = size;
        self.configuration.width = size.width;
        self.configuration.height = size.height;
        self.reconfigure(gpu)
    }
    /// Reconfigures after Outdated or Suboptimal, once the frame is released.
    /// Lost requires recreating the surface through GpuContext::attach_surface.
    pub fn reconfigure(&mut self, gpu: &GpuContext) -> Result<(), GpuInitError> {
        self.check_device(gpu)?;
        self.msaa_view = None;
        if !self.is_suspended() {
            self.surface.configure(&gpu.device, &self.configuration);
            self.msaa_view = create_msaa_view(gpu, self.size(), self.render_target);
        }
        Ok(())
    }
    /// None means zero-sized. Match all CurrentSurfaceTexture variants; submit
    /// work before calling Queue::present on a successfully acquired texture.
    pub fn acquire_surface_texture(&self) -> Option<wgpu::CurrentSurfaceTexture> {
        (!self.is_suspended()).then(|| self.surface.get_current_texture())
    }
    /// Creates a view using the negotiated format, including WebGPU's sRGB view.
    pub fn create_view(&self, texture: &wgpu::SurfaceTexture) -> wgpu::TextureView {
        texture.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.render_target.color_format),
            ..Default::default()
        })
    }
    pub fn descriptor(&self) -> &SurfaceDescriptor {
        &self.descriptor
    }
    pub fn configuration(&self) -> &wgpu::SurfaceConfiguration {
        &self.configuration
    }
    pub fn size(&self) -> PhysicalSize<u32> {
        self.descriptor.size
    }
    pub fn is_suspended(&self) -> bool {
        self.size().width == 0 || self.size().height == 0
    }
    pub fn render_target_config(&self) -> RenderTargetConfig {
        self.render_target
    }
    pub fn render_format(&self) -> wgpu::TextureFormat {
        self.render_target.color_format
    }
    pub fn sample_count(&self) -> u32 {
        self.render_target.sample_count
    }
    pub fn msaa_view(&self) -> Option<&wgpu::TextureView> {
        self.msaa_view.as_ref()
    }
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.configuration.format
    }
    pub fn surface_copy_supported(&self) -> bool {
        self.configuration
            .usage
            .contains(wgpu::TextureUsages::COPY_SRC)
    }
    fn check_device(&self, gpu: &GpuContext) -> Result<(), GpuInitError> {
        if self.device == gpu.device {
            Ok(())
        } else {
            Err(GpuInitError::WrongDevice)
        }
    }
}

/// Owned, copyable color target with matching depth and optional MSAA.
pub struct OffscreenTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    msaa_view: Option<wgpu::TextureView>,
    size: PhysicalSize<u32>,
    render_target: RenderTargetConfig,
}
impl OffscreenTarget {
    pub fn new(
        gpu: &GpuContext,
        size: PhysicalSize<u32>,
        render_target: RenderTargetConfig,
    ) -> Result<Self, GpuInitError> {
        validate_size(size, gpu.device.limits().max_texture_dimension_2d, false)?;
        validate_target(gpu, render_target)?;
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING;
        validate_texture_usage(gpu, render_target.color_format, usage)?;
        let texture = create_texture(
            &gpu.device,
            "mikage_offscreen_color",
            size,
            render_target.color_format,
            1,
            usage,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = create_texture(
            &gpu.device,
            "mikage_offscreen_depth",
            size,
            render_target.depth_format,
            render_target.sample_count,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let msaa_view = create_msaa_view(gpu, size, render_target);
        Ok(Self {
            texture,
            view,
            depth_view,
            msaa_view,
            size,
            render_target,
        })
    }
    /// Recreates size-dependent textures, preserving this target on failure.
    pub fn resize(
        &mut self,
        gpu: &GpuContext,
        size: PhysicalSize<u32>,
    ) -> Result<(), GpuInitError> {
        *self = Self::new(gpu, size, self.render_target)?;
        Ok(())
    }
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
    /// Resolved single-sample view.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }
    pub fn msaa_view(&self) -> Option<&wgpu::TextureView> {
        self.msaa_view.as_ref()
    }
    /// Color attachment view, multisampled when MSAA is enabled.
    pub fn color_view(&self) -> &wgpu::TextureView {
        self.msaa_view.as_ref().unwrap_or(&self.view)
    }
    pub fn resolve_target(&self) -> Option<&wgpu::TextureView> {
        self.msaa_view.as_ref().map(|_| &self.view)
    }
    pub fn size(&self) -> PhysicalSize<u32> {
        self.size
    }
    pub fn render_target_config(&self) -> RenderTargetConfig {
        self.render_target
    }
}

fn resolve_requirements(
    requirements: &GpuRequirements,
    supported_features: wgpu::Features,
    supported_limits: &wgpu::Limits,
    backend: wgpu::Backend,
) -> Result<(wgpu::Features, wgpu::Limits), GpuInitError> {
    let missing = requirements.required_features - supported_features;
    if !missing.is_empty() {
        return Err(GpuInitError::MissingFeatures(missing));
    }
    let required = requirements.required_limits.clone().unwrap_or_else(|| {
        let baseline = if backend == wgpu::Backend::Gl {
            wgpu::Limits::downlevel_webgl2_defaults()
        } else {
            wgpu::Limits::downlevel_defaults()
        };
        baseline.using_resolution(supported_limits.clone())
    });
    check_limits(&required, supported_limits)?;
    let resolved = if let Some(preferred) = &requirements.preferred_limits {
        preferred
            .clone()
            .or_worse_values_from(supported_limits)
            .or_better_values_from(&required)
    } else {
        required
    };
    check_limits(&resolved, supported_limits)?;
    Ok((
        requirements.required_features | (requirements.optional_features & supported_features),
        resolved,
    ))
}
fn check_limits(required: &wgpu::Limits, supported: &wgpu::Limits) -> Result<(), GpuInitError> {
    let mut failures = Vec::new();
    required.check_limits_with_fail_fn(supported, false, |name, required, supported| {
        failures.push(GpuLimitFailure {
            name,
            required,
            supported,
        })
    });
    if failures.is_empty() {
        Ok(())
    } else {
        Err(GpuInitError::UnsupportedLimits(failures))
    }
}
fn surface_configuration(
    caps: &wgpu::SurfaceCapabilities,
    descriptor: &SurfaceDescriptor,
) -> Result<(wgpu::SurfaceConfiguration, RenderTargetConfig), GpuInitError> {
    validate_sample_count(descriptor.sample_count)?;
    let first = *caps.formats.first().ok_or(GpuInitError::NoSurfaceFormats)?;
    let (format, view_format) = if let Some(format) = descriptor.format {
        if !caps.formats.contains(&format) {
            return Err(GpuInitError::UnsupportedSurfaceFormat(format));
        }
        (format, format)
    } else if let Some(format) = caps
        .formats
        .iter()
        .copied()
        .find(wgpu::TextureFormat::is_srgb)
    {
        (format, format)
    } else {
        (first, first.add_srgb_suffix())
    };
    let present_mode = descriptor.present_mode;
    if !matches!(
        present_mode,
        wgpu::PresentMode::AutoVsync | wgpu::PresentMode::AutoNoVsync
    ) && !caps.present_modes.contains(&present_mode)
    {
        return Err(GpuInitError::UnsupportedPresentMode(present_mode));
    }
    let first_alpha = *caps
        .alpha_modes
        .first()
        .ok_or(GpuInitError::NoSurfaceAlphaModes)?;
    let alpha_mode = descriptor.alpha_mode.unwrap_or(first_alpha);
    if alpha_mode != wgpu::CompositeAlphaMode::Auto && !caps.alpha_modes.contains(&alpha_mode) {
        return Err(GpuInitError::UnsupportedAlphaMode(alpha_mode));
    }
    let usage =
        wgpu::TextureUsages::RENDER_ATTACHMENT | (caps.usages & wgpu::TextureUsages::COPY_SRC);
    if !caps.usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
        return Err(GpuInitError::UnsupportedTextureUsage { format, usage });
    }
    let configuration = wgpu::SurfaceConfiguration {
        usage,
        format,
        color_space: wgpu::SurfaceColorSpace::Auto,
        width: descriptor.size.width,
        height: descriptor.size.height,
        present_mode,
        alpha_mode,
        view_formats: if format == view_format {
            vec![]
        } else {
            vec![view_format]
        },
        desired_maximum_frame_latency: descriptor.desired_maximum_frame_latency,
    };
    Ok((
        configuration,
        RenderTargetConfig {
            color_format: view_format,
            depth_format: descriptor.depth_format,
            sample_count: descriptor.sample_count,
        },
    ))
}
fn validate_sample_count(sample_count: u32) -> Result<(), GpuInitError> {
    if matches!(sample_count, 1 | 4) {
        Ok(())
    } else {
        Err(GpuInitError::InvalidSampleCount(sample_count))
    }
}
fn validate_size(
    size: PhysicalSize<u32>,
    limit: u32,
    allow_zero: bool,
) -> Result<(), GpuInitError> {
    if !allow_zero && (size.width == 0 || size.height == 0) {
        return Err(GpuInitError::InvalidSize(size));
    }
    if size.width > limit || size.height > limit {
        return Err(GpuInitError::SizeExceedsLimit { size, limit });
    }
    Ok(())
}
fn validate_target(gpu: &GpuContext, target: RenderTargetConfig) -> Result<(), GpuInitError> {
    validate_sample_count(target.sample_count)?;
    if target.color_format.is_depth_stencil_format() {
        return Err(GpuInitError::InvalidTargetFormat(target.color_format));
    }
    if !target.depth_format.has_depth_aspect() {
        return Err(GpuInitError::InvalidTargetFormat(target.depth_format));
    }
    for format in [target.color_format, target.depth_format] {
        validate_texture_usage(gpu, format, wgpu::TextureUsages::RENDER_ATTACHMENT)?;
        let flags = gpu.format_features(format).flags;
        if !flags.sample_count_supported(target.sample_count)
            || (format == target.color_format
                && target.sample_count > 1
                && !flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE))
        {
            return Err(GpuInitError::UnsupportedSampleCount {
                format,
                sample_count: target.sample_count,
            });
        }
    }
    Ok(())
}
fn validate_texture_usage(
    gpu: &GpuContext,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> Result<(), GpuInitError> {
    if gpu.format_features(format).allowed_usages.contains(usage) {
        Ok(())
    } else {
        Err(GpuInitError::UnsupportedTextureUsage { format, usage })
    }
}
fn create_texture(
    device: &wgpu::Device,
    label: &str,
    size: PhysicalSize<u32>,
    format: wgpu::TextureFormat,
    sample_count: u32,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}
fn create_msaa_view(
    gpu: &GpuContext,
    size: PhysicalSize<u32>,
    target: RenderTargetConfig,
) -> Option<wgpu::TextureView> {
    (target.sample_count > 1).then(|| {
        create_texture(
            &gpu.device,
            "mikage_msaa",
            size,
            target.color_format,
            target.sample_count,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
        .create_view(&wgpu::TextureViewDescriptor::default())
    })
}
async fn create_instance(descriptor: &GpuDescriptor) -> Result<wgpu::Instance, GpuInitError> {
    let backends = descriptor.backends.unwrap_or_else(default_backends);
    if (backends & wgpu::Instance::enabled_backend_features()).is_empty() {
        return Err(GpuInitError::NoBackends);
    }
    let mut options = wgpu::InstanceDescriptor::new_without_display_handle();
    options.backends = backends;
    if let Some(display) = &descriptor.display {
        options = options.with_display_handle(Box::new(display.clone()));
    }
    // Probe before creating a canvas: it cannot switch between WebGPU and GL.
    #[cfg(all(target_family = "wasm", feature = "webgl"))]
    if backends.contains(wgpu::Backends::BROWSER_WEBGPU) && backends.contains(wgpu::Backends::GL) {
        return Ok(wgpu::util::new_instance_with_webgpu_detection(options).await);
    }
    Ok(wgpu::Instance::new(options))
}
fn default_backends() -> wgpu::Backends {
    #[cfg(not(target_family = "wasm"))]
    {
        wgpu::Backends::PRIMARY
    }
    #[cfg(all(target_family = "wasm", not(feature = "webgl")))]
    {
        wgpu::Backends::BROWSER_WEBGPU
    }
    #[cfg(all(target_family = "wasm", feature = "webgl"))]
    {
        wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn optional_features_are_opportunistic() {
        let requirements = GpuRequirements {
            optional_features: wgpu::Features::TIMESTAMP_QUERY,
            ..Default::default()
        };
        let limits = wgpu::Limits::default();
        let (features, _) = resolve_requirements(
            &requirements,
            wgpu::Features::empty(),
            &limits,
            wgpu::Backend::Metal,
        )
        .unwrap();
        assert!(features.is_empty());
        let (features, _) = resolve_requirements(
            &requirements,
            wgpu::Features::TIMESTAMP_QUERY,
            &limits,
            wgpu::Backend::Metal,
        )
        .unwrap();
        assert!(features.contains(wgpu::Features::TIMESTAMP_QUERY));
    }
    #[test]
    fn missing_required_features_are_reported() {
        let r = GpuRequirements {
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            ..Default::default()
        };
        assert!(
            matches!(resolve_requirements(&r, wgpu::Features::empty(), &wgpu::Limits::default(), wgpu::Backend::Metal), Err(GpuInitError::MissingFeatures(f)) if f == wgpu::Features::TIMESTAMP_QUERY)
        );
    }
    #[test]
    fn preferred_limits_clamp_in_both_directions() {
        let supported = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 4,
            min_uniform_buffer_offset_alignment: 128,
            ..wgpu::Limits::default()
        };
        let required = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 4,
            min_uniform_buffer_offset_alignment: 256,
            ..wgpu::Limits::downlevel_defaults()
        };
        let preferred = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 7,
            min_uniform_buffer_offset_alignment: 64,
            ..required.clone()
        };
        let r = GpuRequirements {
            required_limits: Some(required),
            preferred_limits: Some(preferred),
            ..Default::default()
        };
        let (_, resolved) = resolve_requirements(
            &r,
            wgpu::Features::empty(),
            &supported,
            wgpu::Backend::Metal,
        )
        .unwrap();
        assert_eq!(resolved.max_storage_buffers_per_shader_stage, 4);
        assert_eq!(resolved.min_uniform_buffer_offset_alignment, 128);
    }
    #[test]
    fn required_limits_report_all_named_failures() {
        let supported = wgpu::Limits::default();
        let required = wgpu::Limits {
            max_storage_buffers_per_shader_stage: supported.max_storage_buffers_per_shader_stage
                + 1,
            max_bind_groups: supported.max_bind_groups + 1,
            ..supported.clone()
        };
        let Err(GpuInitError::UnsupportedLimits(failures)) = check_limits(&required, &supported)
        else {
            panic!("expected failures")
        };
        assert!(failures.iter().any(|f| f.name == "max_bind_groups"));
        assert!(
            failures
                .iter()
                .any(|f| f.name == "max_storage_buffers_per_shader_stage")
        );
    }
    #[test]
    fn preferences_never_weaken_required_limits() {
        let r = GpuRequirements {
            required_limits: Some(wgpu::Limits {
                max_storage_buffers_per_shader_stage: 7,
                ..wgpu::Limits::downlevel_defaults()
            }),
            preferred_limits: Some(wgpu::Limits::downlevel_defaults()),
            ..Default::default()
        };
        let (_, resolved) = resolve_requirements(
            &r,
            wgpu::Features::empty(),
            &wgpu::Limits::default(),
            wgpu::Backend::Metal,
        )
        .unwrap();
        assert_eq!(resolved.max_storage_buffers_per_shader_stage, 7);
    }
    fn capabilities() -> wgpu::SurfaceCapabilities {
        wgpu::SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Opaque],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
            ..Default::default()
        }
    }
    #[test]
    fn linear_surface_uses_srgb_view_without_copy_support() {
        let (config, target) =
            surface_configuration(&capabilities(), &SurfaceDescriptor::default()).unwrap();
        assert_eq!(config.format, wgpu::TextureFormat::Bgra8Unorm);
        assert_eq!(target.color_format, wgpu::TextureFormat::Bgra8UnormSrgb);
        assert_eq!(config.view_formats, vec![target.color_format]);
        assert_eq!(config.usage, wgpu::TextureUsages::RENDER_ATTACHMENT);
        assert_eq!((config.width, config.height), (0, 0));
    }
    #[test]
    fn explicit_format_and_present_mode_are_validated() {
        let d = SurfaceDescriptor {
            format: Some(wgpu::TextureFormat::Bgra8Unorm),
            ..Default::default()
        };
        let (config, target) = surface_configuration(&capabilities(), &d).unwrap();
        assert_eq!(config.format, target.color_format);
        let d = SurfaceDescriptor {
            present_mode: wgpu::PresentMode::Immediate,
            ..Default::default()
        };
        assert!(matches!(
            surface_configuration(&capabilities(), &d),
            Err(GpuInitError::UnsupportedPresentMode(
                wgpu::PresentMode::Immediate
            ))
        ));
    }
    #[test]
    fn empty_surface_capabilities_return_errors() {
        let mut caps = capabilities();
        caps.alpha_modes.clear();
        assert!(matches!(
            surface_configuration(&caps, &SurfaceDescriptor::default()),
            Err(GpuInitError::NoSurfaceAlphaModes)
        ));
        caps.formats.clear();
        assert!(matches!(
            surface_configuration(&caps, &SurfaceDescriptor::default()),
            Err(GpuInitError::NoSurfaceFormats)
        ));
    }
    #[test]
    fn surface_can_suspend_but_offscreen_size_must_be_positive() {
        assert!(validate_size(PhysicalSize::new(0, 32), 8192, true).is_ok());
        assert!(matches!(
            validate_size(PhysicalSize::new(0, 32), 8192, false),
            Err(GpuInitError::InvalidSize(_))
        ));
        assert!(matches!(
            validate_size(PhysicalSize::new(8193, 32), 8192, true),
            Err(GpuInitError::SizeExceedsLimit { .. })
        ));
    }
}
