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
    /// MSAA sample count for render pipelines.
    sample_count: u32,
    /// Multisample render target (None if sample_count == 1).
    msaa_texture_view: Option<wgpu::TextureView>,
}

impl GpuContext {
    pub(crate) async fn new(
        window: Arc<Window>,
        present_mode: wgpu::PresentMode,
        required_features: wgpu::Features,
        required_limits: Option<wgpu::Limits>,
        sample_count: u32,
    ) -> Self {
        let size = window.inner_size();
        tracing::info!("Initial window size: {}x{}", size.width, size.height);

        #[cfg(not(target_family = "wasm"))]
        let backends = wgpu::Backends::PRIMARY;

        // WASM (webgl feature 無効): WebGPU のみ
        #[cfg(all(target_family = "wasm", not(feature = "webgl")))]
        let backends = wgpu::Backends::BROWSER_WEBGPU;

        // WASM (webgl feature 有効): WebGPU を優先し、アダプタが取れなければ WebGL2。
        // surface 作成前にバックエンドを決定する必要がある（canvas は一度コンテキストを
        // 取得すると他の種類のコンテキストを取得できないため）。
        #[cfg(all(target_family = "wasm", feature = "webgl"))]
        let backends = {
            let probe = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::BROWSER_WEBGPU,
                ..Default::default()
            });
            let has_webgpu = probe
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .is_ok();
            if has_webgpu {
                tracing::info!("Using WebGPU backend");
                wgpu::Backends::BROWSER_WEBGPU
            } else {
                tracing::info!("WebGPU not available, using WebGL2 backend");
                wgpu::Backends::GL
            }
        };

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

        let default_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
            wgpu::Limits::downlevel_webgl2_defaults()
        } else {
            wgpu::Limits::downlevel_defaults()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("mikage_device"),
                required_features,
                required_limits: required_limits
                    .unwrap_or(default_limits)
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

        // WASM では ResizeObserver が非同期のため初期サイズが 0x0 になることがある。
        // 1x1 でフォールバックし、後続の Resized イベントで正しいサイズに更新される。
        let width = size.width.max(1);
        let height = size.height.max(1);

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

        let msaa_texture_view = if sample_count > 1 {
            Some(Self::create_msaa_texture(
                &device,
                view_format,
                width,
                height,
                sample_count,
            ))
        } else {
            None
        };

        Self {
            device,
            queue,
            // パイプライン作成に使うフォーマットは sRGB ビュー側
            render_format: view_format,
            surface,
            surface_config,
            msaa_texture_view,
            sample_count,
        }
    }

    pub(crate) fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
            if self.sample_count > 1 {
                self.msaa_texture_view = Some(Self::create_msaa_texture(
                    &self.device,
                    self.render_format,
                    new_size.width,
                    new_size.height,
                    self.sample_count,
                ));
            }
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

    /// Returns the MSAA sample count configured for this context.
    ///
    /// Use this value in `MultisampleState::count` when creating render pipelines.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Returns the current surface size in pixels.
    pub fn window_size(&self) -> PhysicalSize<u32> {
        PhysicalSize::new(self.surface_config.width, self.surface_config.height)
    }

    /// Returns the MSAA texture view if `sample_count > 1`, or `None`.
    ///
    /// When MSAA is enabled, use this as the render pass color attachment `view`
    /// and the surface texture view as `resolve_target`.
    pub fn msaa_view(&self) -> Option<&wgpu::TextureView> {
        self.msaa_texture_view.as_ref()
    }

    fn create_msaa_texture(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("msaa_texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }
}
