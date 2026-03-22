use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::app::{App, ComputeContext, RenderContext, UpdateContext};
use crate::camera::{CameraController, OrbitCamera};
use crate::context::GpuContext;
use crate::egui_integration::EguiIntegration;
use crate::input::InputState;
use crate::time::FrameTime;

/// Application launch configuration.
///
/// Pass to [`run`] to start the application.
///
/// # Example
/// ```no_run
/// use mikage::RunConfig;
///
/// let config = RunConfig {
///     title: "My App".to_string(),
///     width: 1920,
///     height: 1080,
///     ..Default::default()
/// };
/// ```
pub struct RunConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub camera: Box<dyn CameraController>,
    /// Presentation mode. `AutoVsync` (default) enables vsync.
    /// Use `Immediate` or `Mailbox` to uncap the frame rate.
    pub present_mode: wgpu::PresentMode,
    /// Required wgpu features. Default: empty (no extra features).
    pub wgpu_features: wgpu::Features,
    /// Required wgpu limits. `None` (default) uses `downlevel_defaults()`
    /// resolved against the adapter's actual limits.
    pub wgpu_limits: Option<wgpu::Limits>,
    /// Whether to initialize the tracing logger. Default: `true`.
    /// Set to `false` if you initialize your own tracing subscriber.
    pub init_logging: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            title: "mikage".to_string(),
            width: 1280,
            height: 720,
            camera: Box::new(OrbitCamera::default()),
            present_mode: wgpu::PresentMode::AutoVsync,
            wgpu_features: wgpu::Features::empty(),
            wgpu_limits: None,
            init_logging: true,
        }
    }
}

/// Starts the application.
///
/// Creates a window, initializes the GPU, and enters the event loop.
/// Blocks on native; non-blocking on WASM.
#[cfg(not(target_family = "wasm"))]
pub fn run<A: App>(app: A, config: RunConfig) {
    if config.init_logging {
        crate::logging::init_logging();
    }

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut handler = AppHandler::new(app, config);
    event_loop.run_app(&mut handler).expect("Event loop error");
}

/// Starts the application (WASM).
///
/// Uses `EventLoop::spawn_app` for non-blocking execution.
/// GPU initialization runs asynchronously; rendering starts when ready.
#[cfg(target_family = "wasm")]
pub fn run<A: App>(app: A, config: RunConfig) {
    if config.init_logging {
        crate::logging::init_logging();
    }

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let handler = AppHandler::new(app, config);
    use winit::platform::web::EventLoopExtWebSys;
    event_loop.spawn_app(handler);
}

// --- 共通の RunState ---

struct RunState {
    window: Arc<Window>,
    gpu: GpuContext,
    egui: EguiIntegration,
    input: InputState,
    camera: Box<dyn CameraController>,
    frame_time: FrameTime,
}

// --- AppHandler ---

struct AppHandler<A: App> {
    app: A,
    config: Option<RunConfig>,
    state: Option<RunState>,
    /// WASM: async GPU 初期化の完了を受け取るための共有スロット
    #[cfg(target_family = "wasm")]
    pending_gpu: Option<PendingGpuInit>,
}

/// WASM 用: async GPU 初期化の完了待ち
#[cfg(target_family = "wasm")]
struct PendingGpuInit {
    window: Arc<Window>,
    camera: Box<dyn CameraController>,
    slot: std::rc::Rc<std::cell::RefCell<Option<GpuContext>>>,
}

impl<A: App> AppHandler<A> {
    fn new(app: A, config: RunConfig) -> Self {
        Self {
            app,
            config: Some(config),
            state: None,
            #[cfg(target_family = "wasm")]
            pending_gpu: None,
        }
    }

    /// GPU 初期化完了後の共通セットアップ
    fn complete_init(
        &mut self,
        window: Arc<Window>,
        gpu: GpuContext,
        camera: Box<dyn CameraController>,
    ) {
        let egui = EguiIntegration::new(&window, &gpu);
        // サーフェスの実サイズを使う（WASM では window.inner_size() と異なる場合がある）
        let size = gpu.window_size();

        tracing::info!("App init with size: {}x{}", size.width, size.height);
        self.app.init(&gpu, size);

        self.state = Some(RunState {
            window,
            gpu,
            egui,
            input: InputState::default(),
            camera,
            frame_time: FrameTime::new(),
        });
    }
}

impl<A: App> ApplicationHandler for AppHandler<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let config = match self.config.take() {
            Some(c) => c,
            None => return,
        };

        #[allow(unused_mut)]
        let mut window_attrs = WindowAttributes::default()
            .with_title(&config.title)
            .with_inner_size(PhysicalSize::new(config.width, config.height));

        #[cfg(target_family = "wasm")]
        {
            use winit::platform::web::WindowAttributesExtWebSys;
            window_attrs = window_attrs.with_append(true);
        }

        let window = Arc::new(
            event_loop
                .create_window(window_attrs)
                .expect("Failed to create window"),
        );

        // Native: 同期的に GPU 初期化
        #[cfg(not(target_family = "wasm"))]
        {
            let gpu = pollster::block_on(GpuContext::new(
                window.clone(),
                config.present_mode,
                config.wgpu_features,
                config.wgpu_limits.clone(),
            ));
            self.complete_init(window, gpu, config.camera);
        }

        // WASM: 非同期に GPU 初期化
        #[cfg(target_family = "wasm")]
        {
            let slot = std::rc::Rc::new(std::cell::RefCell::new(None));
            let slot_clone = slot.clone();
            let window_clone = window.clone();

            let present_mode = config.present_mode;
            let wgpu_features = config.wgpu_features;
            let wgpu_limits = config.wgpu_limits.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let gpu =
                    GpuContext::new(window_clone, present_mode, wgpu_features, wgpu_limits).await;
                *slot_clone.borrow_mut() = Some(gpu);
            });

            self.pending_gpu = Some(PendingGpuInit {
                window: window.clone(),
                camera: config.camera,
                slot,
            });

            // redraw をリクエストして初期化完了チェックを駆動
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // WASM: async GPU 初期化の完了チェック
        #[cfg(target_family = "wasm")]
        if self.state.is_none() {
            if let Some(pending) = self.pending_gpu.take() {
                if let Some(gpu) = pending.slot.borrow_mut().take() {
                    // 初期化完了
                    let window = pending.window.clone();
                    self.complete_init(pending.window, gpu, pending.camera);
                    tracing::info!("GPU initialization complete (WASM)");
                    // 初期化直後に redraw をリクエスト
                    window.request_redraw();
                } else {
                    // まだ完了していない、戻す
                    let window = pending.window.clone();
                    self.pending_gpu = Some(pending);
                    window.request_redraw();
                    return;
                }
            } else {
                return;
            }
        }

        let Some(state) = self.state.as_mut() else {
            return;
        };

        // egui にイベントを先に転送
        let _egui_consumed = state.egui.handle_window_event(&state.window, &event);

        // 入力状態を更新
        state.input.handle_event(&event);

        // カメラへのマウスドラッグ転送（egui が入力を要求していなければ）
        if !state.egui.wants_any_input() {
            match &event {
                WindowEvent::CursorMoved { .. } => {
                    let (dx, dy) = state.input.mouse_delta;
                    if dx != 0.0 || dy != 0.0 {
                        let buttons = &state.input.mouse_buttons_down;
                        state.camera.on_mouse_drag(
                            dx,
                            dy,
                            buttons.left,
                            buttons.right,
                            buttons.middle,
                        );
                    }
                }
                WindowEvent::MouseWheel { .. } => {
                    state.camera.on_scroll(state.input.scroll_delta);
                }
                _ => {}
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    state.gpu.resize(new_size);
                    state.egui.resize(
                        new_size.width,
                        new_size.height,
                        crate::egui_integration::EguiIntegration::compute_pixels_per_point(
                            &state.window,
                        ),
                    );
                    self.app.resize(&state.gpu, new_size);
                }
            }
            WindowEvent::RedrawRequested => {
                render_frame(&mut self.app, state);
            }
            ref other => {
                self.app.on_window_event(other);
            }
        }
    }
}

// --- タッチ入力 ---
// winit on WASM は 1本指タッチを CursorMoved + MouseInput(Left) として転送するため、
// 1本指ドラッグ = orbit は追加コード不要で動作する。
// 2本指のピンチ/パンは必要に応じて追加実装する。

// --- レンダリング ---

fn render_frame<A: App>(app: &mut A, state: &mut RunState) {
    state.frame_time.tick();

    let size = state.gpu.window_size();

    // カメラ更新
    state.camera.update(state.frame_time.dt);

    // App 更新
    {
        let mut update_ctx = UpdateContext {
            dt: state.frame_time.dt,
            elapsed: state.frame_time.elapsed,
            window_size: size,
            gpu: &state.gpu,
            input: &state.input,
            camera: &mut *state.camera,
        };
        app.update(&mut update_ctx);
    }

    let surface_texture = match state.gpu.surface_texture() {
        Ok(tex) => tex,
        Err(wgpu::SurfaceError::Lost) => {
            state.gpu.resize(size);
            return;
        }
        Err(wgpu::SurfaceError::OutOfMemory) => {
            tracing::error!("Out of GPU memory");
            return;
        }
        Err(e) => {
            tracing::warn!("Surface error: {e:?}");
            return;
        }
    };

    let surface_view = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor {
            // render_format は sRGB view format（WebGPU で view_formats 経由）
            format: Some(state.gpu.render_format()),
            ..Default::default()
        });

    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mikage_frame_encoder"),
        });

    // Compute pass
    {
        let mut compute_ctx = ComputeContext {
            device: &state.gpu.device,
            queue: &state.gpu.queue,
            encoder: &mut encoder,
        };
        app.compute(&mut compute_ctx);
    }

    // Render pass
    {
        let mut render_ctx = RenderContext {
            device: &state.gpu.device,
            queue: &state.gpu.queue,
            encoder: &mut encoder,
            surface_view: &surface_view,
            render_format: state.gpu.render_format(),
            window_size: size,
            camera: state.camera.as_ref() as &dyn crate::camera::Camera,
        };
        app.render(&mut render_ctx);
    }

    // egui: UI 構築 + 描画
    state.egui.render(
        &state.window,
        &state.gpu,
        &mut encoder,
        &surface_view,
        |egui_ctx| {
            app.gui(egui_ctx);
        },
    );

    state.gpu.queue.submit(std::iter::once(encoder.finish()));
    surface_texture.present();

    state.window.request_redraw();

    // フレーム末尾で per-frame 入力状態をリセット。
    // 次フレームの window_event() で蓄積されたイベントが
    // その次の render_frame() の app.update() で見えるようにする。
    state.input.begin_frame();
}
