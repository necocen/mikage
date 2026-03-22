use std::sync::Arc;
use winit::dpi::PhysicalSize;
use winit::window::Window;

/// GPU context holding wgpu Device, Queue, and Surface.
///
/// Created and managed by the framework. Passed to [`App`](crate::App) methods.
/// The `device` and `queue` fields are public so you can create
/// pipelines, buffers, and textures directly.
pub struct GpuContext {
    /// The wgpu device. Use for creating pipelines, buffers, and textures.
    pub device: wgpu::Device,
    /// The wgpu queue. Use for submitting commands and writing buffers.
    pub queue: wgpu::Queue,
    /// The texture format to use for render pipelines (may differ from surface config format on WASM).
    render_format: wgpu::TextureFormat,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
}

impl GpuContext {
    pub(crate) async fn new(
        window: Arc<Window>,
        present_mode: wgpu::PresentMode,
        required_features: wgpu::Features,
        required_limits: Option<wgpu::Limits>,
    ) -> Self {
        let size = window.inner_size();
        tracing::info!("Initial window size: {}x{}", size.width, size.height);

        // WASM では WebGPU バックエンドを明示的に指定
        #[cfg(target_family = "wasm")]
        let backends = wgpu::Backends::BROWSER_WEBGPU;
        #[cfg(not(target_family = "wasm"))]
        let backends = wgpu::Backends::PRIMARY;

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find a suitable GPU adapter");

        tracing::info!("Adapter: {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("mikage_device"),
                required_features,
                required_limits: required_limits
                    .unwrap_or_else(wgpu::Limits::downlevel_defaults)
                    .using_resolution(adapter.limits()),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                ..Default::default()
            })
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        tracing::info!("Surface formats: {:?}", surface_caps.formats);

        // sRGB フォーマットを優先（正しいガンマ補正のため）
        // WebGPU では sRGB フォーマットが直接利用できない場合がある。
        // その場合は非sRGB フォーマットで configure し、sRGB の view を作って使う。
        let preferred_srgb = surface_caps.formats.iter().find(|f| f.is_srgb()).copied();

        let (config_format, view_format) = if let Some(srgb) = preferred_srgb {
            // sRGB が直接使える（native の場合）
            (srgb, srgb)
        } else {
            // sRGB がない（WebGPU の場合）: 非sRGB で configure + sRGB view
            let base = surface_caps.formats[0];
            // Bgra8Unorm → Bgra8UnormSrgb, Rgba8Unorm → Rgba8UnormSrgb
            let srgb_view = match base {
                wgpu::TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8UnormSrgb,
                wgpu::TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8UnormSrgb,
                other => other, // フォールバック: そのまま（sRGB 変換なし）
            };
            (base, srgb_view)
        };

        // view_formats に sRGB バリアントを追加（config_format と異なる場合のみ）
        let view_formats = if config_format != view_format {
            vec![view_format]
        } else {
            vec![]
        };

        tracing::info!(
            "Surface config format: {:?}, view format: {:?}",
            config_format,
            view_format
        );

        // WASM ではキャンバスサイズが 0 の場合がある。フォールバック。
        let width = if size.width > 0 { size.width } else { 800 };
        let height = if size.height > 0 { size.height } else { 600 };

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: config_format,
            width,
            height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats,
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        Self {
            device,
            queue,
            // パイプライン作成に使うフォーマットは sRGB ビュー側
            render_format: view_format,
            surface,
            surface_config,
        }
    }

    pub(crate) fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    pub(crate) fn surface_texture(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }

    /// Returns the texture format to use when creating render pipelines.
    ///
    /// On WASM (WebGPU), this may be an sRGB view format that differs from
    /// the underlying surface configuration format, ensuring correct
    /// gamma correction across platforms.
    pub fn render_format(&self) -> wgpu::TextureFormat {
        self.render_format
    }

    /// Returns the current surface size in pixels.
    pub fn window_size(&self) -> PhysicalSize<u32> {
        PhysicalSize::new(self.surface_config.width, self.surface_config.height)
    }
}
