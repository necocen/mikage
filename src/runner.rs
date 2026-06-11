use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::app::{App, FrameContext, UpdateContext};
use crate::camera::{InteractiveCamera, OrbitCamera};
use crate::context::{GpuContext, SurfaceContext};
use crate::egui_integration::EguiIntegration;
use crate::input::InputState;
use crate::time::FrameTime;

/// Controls when the runner requests the next frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedrawPolicy {
    /// Request a new frame after every rendered frame.
    Continuous,
    /// Request frames after input/window events or immediate egui repaint requests.
    Reactive,
}

/// Application launch configuration.
///
/// Pass to [`run`] to start the application.
///
/// The type parameter `C` is the [`InteractiveCamera`] type. It defaults to
/// [`OrbitCamera`], so `RunConfig` without an explicit type parameter uses
/// `OrbitCamera` and supports [`Default`].
///
/// # Example
/// ```no_run
/// use mikage::RunConfig;
///
/// // OrbitCamera (default) — supports Default
/// let config = RunConfig {
///     title: "My App".to_string(),
///     width: 1920,
///     height: 1080,
///     ..Default::default()
/// };
///
/// // Other camera — use with_defaults or builder
/// use mikage::Camera2d;
/// let config = RunConfig::new("My 2D App").with_camera(Camera2d::default());
/// ```
pub struct RunConfig<C: InteractiveCamera = OrbitCamera> {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub camera: C,
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
    /// MSAA sample count for render pipelines. Default: 1 (no MSAA).
    /// Must be `1` (no MSAA) or `4` (4× MSAA); other values will panic
    /// at startup. The framework stores this value and exposes it via
    /// [`GpuContext::sample_count`](crate::GpuContext::sample_count);
    /// applications use it when creating render pipelines.
    pub sample_count: u32,
    /// CSS selector for an existing `<canvas>` element (e.g., `"#my-canvas"`).
    ///
    /// If `Some`, the specified canvas is used and its size is controlled by the user's CSS.
    /// If `None` (default), a canvas is created automatically and sized to fill the viewport.
    /// This option is only used on WASM; it is ignored on native platforms.
    pub canvas: Option<String>,
    /// Redraw scheduling policy. Default: [`RedrawPolicy::Continuous`].
    pub redraw_policy: RedrawPolicy,
    /// Number of physical scroll pixels treated as one line unit.
    /// Default: `50.0`.
    pub pixel_scroll_per_line: f32,
    /// Multiplier applied to touch pinch distance deltas. Default: `5.0`.
    pub touch_pinch_sensitivity: f32,
}

impl Default for RunConfig<OrbitCamera> {
    fn default() -> Self {
        Self::with_defaults(OrbitCamera::default())
    }
}

impl Default for RunConfig<()> {
    fn default() -> Self {
        Self::with_defaults(())
    }
}

impl RunConfig<OrbitCamera> {
    /// Creates the default config with [`OrbitCamera`].
    pub fn default() -> Self {
        <Self as Default>::default()
    }

    /// Creates a new config with the given title and default [`OrbitCamera`].
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Default::default()
        }
    }
}

impl<C: InteractiveCamera> RunConfig<C> {
    /// Creates a new config with the given camera and default values for everything else.
    ///
    /// Use this instead of [`Default`] when the camera type is not [`OrbitCamera`].
    pub fn with_defaults(camera: C) -> Self {
        Self {
            title: "mikage".to_string(),
            width: 1280,
            height: 720,
            camera,
            present_mode: wgpu::PresentMode::AutoVsync,
            wgpu_features: wgpu::Features::empty(),
            wgpu_limits: None,
            init_logging: true,
            sample_count: 1,
            canvas: None,
            redraw_policy: RedrawPolicy::Continuous,
            pixel_scroll_per_line: InputState::DEFAULT_PIXEL_SCROLL_PER_LINE,
            touch_pinch_sensitivity: TouchTracker::DEFAULT_PINCH_SENSITIVITY,
        }
    }

    /// Replaces the camera, changing the type parameter.
    pub fn with_camera<C2: InteractiveCamera>(self, camera: C2) -> RunConfig<C2> {
        // Keep this exhaustive: changing the type parameter requires rebuilding
        // the struct, so newly added fields must be carried over here.
        RunConfig {
            title: self.title,
            width: self.width,
            height: self.height,
            camera,
            present_mode: self.present_mode,
            wgpu_features: self.wgpu_features,
            wgpu_limits: self.wgpu_limits,
            init_logging: self.init_logging,
            sample_count: self.sample_count,
            canvas: self.canvas,
            redraw_policy: self.redraw_policy,
            pixel_scroll_per_line: self.pixel_scroll_per_line,
            touch_pinch_sensitivity: self.touch_pinch_sensitivity,
        }
    }

    /// Sets the window title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Sets the window size (width and height).
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Sets the presentation mode.
    pub fn with_present_mode(mut self, mode: wgpu::PresentMode) -> Self {
        self.present_mode = mode;
        self
    }

    /// Sets the canvas CSS selector (WASM only).
    pub fn with_canvas(mut self, selector: impl Into<String>) -> Self {
        self.canvas = Some(selector.into());
        self
    }

    /// Sets the redraw policy.
    pub fn with_redraw_policy(mut self, policy: RedrawPolicy) -> Self {
        self.redraw_policy = policy;
        self
    }

    /// Sets how many physical scroll pixels correspond to one line unit.
    pub fn with_pixel_scroll_per_line(mut self, pixels: f32) -> Self {
        self.pixel_scroll_per_line = pixels.max(1.0);
        self
    }

    /// Sets the touch pinch zoom sensitivity multiplier.
    pub fn with_touch_pinch_sensitivity(mut self, sensitivity: f32) -> Self {
        self.touch_pinch_sensitivity = sensitivity.max(0.0);
        self
    }
}

/// Starts the application.
///
/// Creates a window, initializes the GPU, calls the factory closure to create the app,
/// and enters the event loop. Blocks on native; non-blocking on WASM.
#[cfg(not(target_family = "wasm"))]
pub fn run<A: App>(
    init: impl FnOnce(&GpuContext, PhysicalSize<u32>) -> A + 'static,
    config: RunConfig<A::Camera>,
) {
    if config.init_logging {
        crate::logging::init_logging();
    }

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut handler = AppHandler::new(Box::new(init), config);
    event_loop.run_app(&mut handler).expect("Event loop error");
}

/// Starts the application with the native HTTP agent API enabled.
///
/// This is available on native builds with the `agent` feature. The regular
/// frame loop is unchanged, but local HTTP requests can request screenshots,
/// camera movement, redraws, and app-specific JSON commands.
#[cfg(all(feature = "agent", not(target_family = "wasm")))]
pub fn run_with_agent<A: App>(
    init: impl FnOnce(&GpuContext, PhysicalSize<u32>) -> A + 'static,
    config: RunConfig<A::Camera>,
    agent_config: crate::agent::AgentConfig,
) {
    if config.init_logging {
        crate::logging::init_logging();
    }

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let agent = crate::agent::AgentBridge::start(agent_config, event_loop.create_proxy())
        .expect("Failed to start mikage agent HTTP API");
    let mut handler = AppHandler::new_with_agent(Box::new(init), config, agent);
    event_loop.run_app(&mut handler).expect("Event loop error");
}

/// Starts the application (WASM).
///
/// Uses `EventLoop::spawn_app` for non-blocking execution.
/// GPU initialization runs asynchronously; rendering starts when ready.
#[cfg(target_family = "wasm")]
pub fn run<A: App>(
    init: impl FnOnce(&GpuContext, PhysicalSize<u32>) -> A + 'static,
    config: RunConfig<A::Camera>,
) {
    if config.init_logging {
        crate::logging::init_logging();
    }

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let handler = AppHandler::new(Box::new(init), config);
    use winit::platform::web::EventLoopExtWebSys;
    event_loop.spawn_app(handler);
}

// --- 共通の RunState ---

struct RunState<C: InteractiveCamera> {
    window: Arc<Window>,
    gpu: GpuContext,
    surface: SurfaceContext,
    egui: EguiIntegration,
    input: InputState,
    camera: C,
    frame_time: FrameTime,
    touch_tracker: TouchTracker,
    redraw_policy: RedrawPolicy,
    /// Whether pointer events were suppressed (egui capturing) in the previous event.
    /// Used to detect suppress transitions and send on_drag_end.
    pointer_suppressed: bool,
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    pending_screenshot: Option<std::sync::mpsc::Sender<crate::agent::AgentResponse>>,
    /// WASM: Resized イベントを render_frame の先頭まで遅延させる。
    /// 連続リサイズ中に毎イベント surface.configure() が走るのを防ぐ。
    #[cfg(target_family = "wasm")]
    pending_resize: Option<PhysicalSize<u32>>,
}

// --- タッチ・ジェスチャー入力 ---

/// Tracks touch gestures: one-finger drag (orbit) and two-finger pinch/pan.
struct TouchTracker {
    /// Active touch points.
    touches: std::collections::HashMap<u64, (f64, f64)>,
    /// Previous distance between two fingers (for pinch detection).
    prev_pinch_distance: Option<f64>,
    /// Previous midpoint of two fingers (for two-finger pan).
    prev_midpoint: Option<(f64, f64)>,
    pinch_sensitivity: f32,
}

impl Default for TouchTracker {
    fn default() -> Self {
        Self {
            touches: std::collections::HashMap::new(),
            prev_pinch_distance: None,
            prev_midpoint: None,
            pinch_sensitivity: Self::DEFAULT_PINCH_SENSITIVITY,
        }
    }
}

enum TouchGestureAction {
    /// One-finger drag (orbit): dx, dy in pixels.
    OneDrag { dx: f64, dy: f64 },
    /// One-finger released.
    OneDragEnd,
    /// Two-finger gesture: pinch zoom + pan.
    TwoFinger {
        scroll_delta: f32,
        pan_dx: f64,
        pan_dy: f64,
        /// Midpoint of the two fingers in physical pixels.
        midpoint: (f64, f64),
    },
}

impl TouchTracker {
    const DEFAULT_PINCH_SENSITIVITY: f32 = 5.0;

    fn new(pinch_sensitivity: f32) -> Self {
        Self {
            pinch_sensitivity,
            ..Default::default()
        }
    }

    fn handle_touch(&mut self, touch: &winit::event::Touch) -> Option<TouchGestureAction> {
        use winit::event::TouchPhase;
        match touch.phase {
            TouchPhase::Started => {
                self.touches
                    .insert(touch.id, (touch.location.x, touch.location.y));
                if self.touches.len() == 2 {
                    let (dist, mid) = self.two_finger_state();
                    self.prev_pinch_distance = Some(dist);
                    self.prev_midpoint = Some(mid);
                }
                None
            }
            TouchPhase::Moved => {
                let new_pos = (touch.location.x, touch.location.y);
                let prev_pos = self.touches.insert(touch.id, new_pos);

                match self.touches.len() {
                    1 => {
                        // One-finger drag
                        if let Some((px, py)) = prev_pos {
                            let dx = new_pos.0 - px;
                            let dy = new_pos.1 - py;
                            Some(TouchGestureAction::OneDrag { dx, dy })
                        } else {
                            None
                        }
                    }
                    2 => {
                        // Two-finger pinch + pan
                        let (dist, mid) = self.two_finger_state();
                        let scroll_delta = self
                            .prev_pinch_distance
                            .map(|prev| ((dist / prev) - 1.0) as f32 * self.pinch_sensitivity)
                            .unwrap_or(0.0);
                        let (pan_dx, pan_dy) = self
                            .prev_midpoint
                            .map(|(px, py)| (mid.0 - px, mid.1 - py))
                            .unwrap_or((0.0, 0.0));

                        self.prev_pinch_distance = Some(dist);
                        self.prev_midpoint = Some(mid);
                        Some(TouchGestureAction::TwoFinger {
                            scroll_delta,
                            pan_dx,
                            pan_dy,
                            midpoint: mid,
                        })
                    }
                    _ => None,
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.touches.remove(&touch.id);
                self.prev_pinch_distance = None;
                self.prev_midpoint = None;
                if self.touches.is_empty() {
                    Some(TouchGestureAction::OneDragEnd)
                } else {
                    None
                }
            }
        }
    }

    fn two_finger_state(&self) -> (f64, (f64, f64)) {
        let mut iter = self.touches.values();
        let &(x0, y0) = iter.next().unwrap();
        let &(x1, y1) = iter.next().unwrap();
        let dx = x1 - x0;
        let dy = y1 - y0;
        let dist = (dx * dx + dy * dy).sqrt().max(1.0);
        let mid = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
        (dist, mid)
    }
}

// --- AppHandler ---

type InitFn<A> = Box<dyn FnOnce(&GpuContext, PhysicalSize<u32>) -> A>;

struct AppHandler<A: App> {
    app: Option<A>,
    init_fn: Option<InitFn<A>>,
    config: Option<RunConfig<A::Camera>>,
    state: Option<RunState<A::Camera>>,
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    agent: Option<crate::agent::AgentBridge>,
    /// WASM: async GPU 初期化の完了を受け取るための共有スロット
    #[cfg(target_family = "wasm")]
    pending_gpu: Option<PendingGpuInit<A::Camera>>,
}

/// WASM 用: async GPU 初期化の完了待ち
#[cfg(target_family = "wasm")]
struct PendingGpuInit<C: InteractiveCamera> {
    window: Arc<Window>,
    camera: C,
    slot: std::rc::Rc<
        std::cell::RefCell<Option<Result<(GpuContext, SurfaceContext), crate::GpuInitError>>>,
    >,
    /// GPU 初期化中に届いた最新の Resized イベントをバッファリングする。
    /// winit の ResizeObserver は非同期に発火するため、初期化完了前に
    /// 正しいサイズの Resized が届くことがある。
    buffered_resize: Option<PhysicalSize<u32>>,
    redraw_policy: RedrawPolicy,
    pixel_scroll_per_line: f32,
    touch_pinch_sensitivity: f32,
}

impl<A: App> AppHandler<A> {
    fn new(init_fn: InitFn<A>, config: RunConfig<A::Camera>) -> Self {
        Self {
            app: None,
            init_fn: Some(init_fn),
            config: Some(config),
            state: None,
            #[cfg(all(feature = "agent", not(target_family = "wasm")))]
            agent: None,
            #[cfg(target_family = "wasm")]
            pending_gpu: None,
        }
    }

    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    fn new_with_agent(
        init_fn: InitFn<A>,
        config: RunConfig<A::Camera>,
        agent: crate::agent::AgentBridge,
    ) -> Self {
        let mut handler = Self::new(init_fn, config);
        handler.agent = Some(agent);
        handler
    }

    /// GPU 初期化完了後の共通セットアップ
    fn complete_init(
        &mut self,
        window: Arc<Window>,
        gpu: GpuContext,
        surface: SurfaceContext,
        camera: A::Camera,
        redraw_policy: RedrawPolicy,
        pixel_scroll_per_line: f32,
        touch_pinch_sensitivity: f32,
    ) {
        let egui = EguiIntegration::new(&window, &gpu, &surface);
        let size = surface.window_size();

        tracing::info!("App init with size: {}x{}", size.width, size.height);
        let init_fn = self.init_fn.take().expect("init_fn already consumed");
        self.app = Some(init_fn(&gpu, size));

        // Notify camera of initial viewport size
        let mut camera = camera;
        camera.set_viewport_size(size.width, size.height);

        self.state = Some(RunState {
            window: window.clone(),
            gpu,
            surface,
            egui,
            input: InputState::new(pixel_scroll_per_line),
            camera,
            frame_time: FrameTime::new(),
            touch_tracker: TouchTracker::new(touch_pinch_sensitivity),
            redraw_policy,
            pointer_suppressed: false,
            #[cfg(all(feature = "agent", not(target_family = "wasm")))]
            pending_screenshot: None,
            #[cfg(target_family = "wasm")]
            pending_resize: None,
        });

        #[cfg(all(feature = "agent", not(target_family = "wasm")))]
        self.publish_agent_snapshot();

        window.request_redraw();
    }

    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    fn publish_agent_snapshot(&self) {
        let Some(agent) = &self.agent else {
            return;
        };
        let Some(state) = &self.state else {
            return;
        };

        let app_status = self
            .app
            .as_ref()
            .map(|app| app.agent_status())
            .unwrap_or(serde_json::Value::Null);
        let size = state.surface.window_size();
        agent.update_snapshot(|snapshot| {
            snapshot.ready = self.app.is_some();
            snapshot.window_size = Some([size.width, size.height]);
            snapshot.frame_count = state.frame_time.frame_count;
            snapshot.elapsed = state.frame_time.elapsed;
            snapshot.screenshot_supported = state.surface.screenshot_supported();
            snapshot.camera = Some(state.camera.agent_snapshot());
            snapshot.app = app_status;
        });
    }

    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    fn handle_agent_requests(&mut self, event_loop: &ActiveEventLoop) {
        let requests = match self.agent.as_mut() {
            Some(agent) => agent.drain_requests(),
            None => return,
        };

        for request in requests {
            self.handle_agent_request(event_loop, request);
        }
        self.publish_agent_snapshot();
    }

    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    fn handle_agent_request(
        &mut self,
        event_loop: &ActiveEventLoop,
        request: crate::agent::AgentRequest,
    ) {
        match request.kind {
            crate::agent::AgentRequestKind::Command(command) => {
                self.handle_agent_command(event_loop, command, request.respond_to);
            }
            crate::agent::AgentRequestKind::Screenshot => {
                let Some(state) = self.state.as_mut() else {
                    let _ = request
                        .respond_to
                        .send(crate::agent::AgentResponse::unavailable("app is not ready"));
                    return;
                };

                if state.pending_screenshot.is_some() {
                    let _ = request
                        .respond_to
                        .send(crate::agent::AgentResponse::unavailable(
                            "another screenshot is already pending",
                        ));
                    return;
                }

                state.pending_screenshot = Some(request.respond_to);
                state.window.request_redraw();
            }
        }
    }

    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    fn handle_agent_command(
        &mut self,
        event_loop: &ActiveEventLoop,
        command: crate::agent::AgentCommand,
        respond_to: std::sync::mpsc::Sender<crate::agent::AgentResponse>,
    ) {
        match command {
            crate::agent::AgentCommand::Redraw => {
                if let Some(state) = &self.state {
                    state.window.request_redraw();
                }
                let _ = respond_to.send(crate::agent::AgentResponse::ok());
            }
            crate::agent::AgentCommand::Shutdown => {
                let _ = respond_to.send(crate::agent::AgentResponse::ok());
                event_loop.exit();
            }
            crate::agent::AgentCommand::AppCommand { payload } => {
                let Some(app) = self.app.as_mut() else {
                    let _ = respond_to
                        .send(crate::agent::AgentResponse::unavailable("app is not ready"));
                    return;
                };
                match app.on_agent_command(payload) {
                    Ok(value) => {
                        let _ = respond_to.send(crate::agent::AgentResponse::json(value));
                    }
                    Err(message) => {
                        let _ = respond_to.send(crate::agent::AgentResponse::bad_request(message));
                    }
                }
            }
            command if command.is_camera_command() => {
                let Some(state) = self.state.as_mut() else {
                    let _ = respond_to
                        .send(crate::agent::AgentResponse::unavailable("app is not ready"));
                    return;
                };
                match state.camera.apply_agent_command(&command) {
                    Ok(()) => {
                        state.window.request_redraw();
                        let _ = respond_to.send(crate::agent::AgentResponse::ok());
                    }
                    Err(message) => {
                        let _ = respond_to.send(crate::agent::AgentResponse::bad_request(message));
                    }
                }
            }
            _ => {
                let _ = respond_to.send(crate::agent::AgentResponse::bad_request(
                    "unsupported agent command",
                ));
            }
        }
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

        let mut window_attrs = WindowAttributes::default().with_title(&config.title);

        // Native: ウィンドウサイズを指定
        #[cfg(not(target_family = "wasm"))]
        {
            window_attrs =
                window_attrs.with_inner_size(PhysicalSize::new(config.width, config.height));
        }

        // WASM: ユーザー指定のキャンバスを使うか、自動作成する
        #[cfg(target_family = "wasm")]
        let auto_created_canvas = config.canvas.is_none();
        #[cfg(target_family = "wasm")]
        {
            use winit::platform::web::WindowAttributesExtWebSys;
            if let Some(selector) = &config.canvas {
                // ユーザー提供: with_inner_size を使わず、ユーザーの CSS に任せる
                use wasm_bindgen::JsCast;
                let document = web_sys::window()
                    .expect("no window")
                    .document()
                    .expect("no document");
                let el = document
                    .query_selector(selector)
                    .expect("invalid selector")
                    .unwrap_or_else(|| panic!("canvas not found: {selector}"));
                let canvas: web_sys::HtmlCanvasElement =
                    el.dyn_into().expect("element is not a canvas");
                window_attrs = window_attrs.with_canvas(Some(canvas));
            } else {
                // 自動作成: with_inner_size で初期サイズを設定し、後で 100% に上書き
                window_attrs = window_attrs
                    .with_inner_size(PhysicalSize::new(config.width, config.height))
                    .with_append(true);
            }
        }

        let window = Arc::new(
            event_loop
                .create_window(window_attrs)
                .expect("Failed to create window"),
        );

        // WASM: 自動作成の場合、キャンバスをビューポートに合わせる
        // （winit のインラインスタイルを上書き）
        #[cfg(target_family = "wasm")]
        if auto_created_canvas {
            use winit::platform::web::WindowExtWebSys;
            if let Some(canvas) = window.canvas() {
                let _ = canvas.set_attribute("style", "width: 100%; height: 100%;");
            }
        }

        // Native: 同期的に GPU 初期化
        #[cfg(not(target_family = "wasm"))]
        {
            let (gpu, surface) = pollster::block_on(GpuContext::new_for_window(
                window.clone(),
                config.present_mode,
                config.wgpu_features,
                config.wgpu_limits.clone(),
                config.sample_count,
            ))
            .expect("Failed to initialize GPU");
            self.complete_init(
                window,
                gpu,
                surface,
                config.camera,
                config.redraw_policy,
                config.pixel_scroll_per_line,
                config.touch_pinch_sensitivity,
            );
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
            let sample_count = config.sample_count;
            wasm_bindgen_futures::spawn_local(async move {
                let result = GpuContext::new_for_window(
                    window_clone,
                    present_mode,
                    wgpu_features,
                    wgpu_limits,
                    sample_count,
                )
                .await;
                *slot_clone.borrow_mut() = Some(result);
            });

            self.pending_gpu = Some(PendingGpuInit {
                window: window.clone(),
                camera: config.camera,
                slot,
                buffered_resize: None,
                redraw_policy: config.redraw_policy,
                pixel_scroll_per_line: config.pixel_scroll_per_line,
                touch_pinch_sensitivity: config.touch_pinch_sensitivity,
            });

            // redraw をリクエストして初期化完了チェックを駆動
            window.request_redraw();
        }
    }

    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    fn user_event(&mut self, event_loop: &ActiveEventLoop, _event: ()) {
        self.handle_agent_requests(event_loop);
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
            // GPU 初期化中に届いた Resized イベントをバッファリング
            if let WindowEvent::Resized(size) = &event {
                if let Some(pending) = &mut self.pending_gpu {
                    if size.width > 0 && size.height > 0 {
                        pending.buffered_resize = Some(*size);
                    }
                }
            }

            if let Some(pending) = self.pending_gpu.take() {
                let result = { pending.slot.borrow_mut().take() };
                if let Some(result) = result {
                    let (gpu, mut surface) = match result {
                        Ok(pair) => pair,
                        Err(err) => {
                            tracing::error!("GPU initialization failed: {err}");
                            show_wasm_gpu_error(&pending.window, &err.to_string());
                            return;
                        }
                    };
                    // 初期化完了: バッファされた Resized を適用
                    if let Some(size) = pending.buffered_resize {
                        surface.resize(&gpu.device, size);
                    }
                    let window = pending.window.clone();
                    self.complete_init(
                        pending.window,
                        gpu,
                        surface,
                        pending.camera,
                        pending.redraw_policy,
                        pending.pixel_scroll_per_line,
                        pending.touch_pinch_sensitivity,
                    );
                    tracing::info!("GPU initialization complete (WASM)");
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
        #[cfg(all(feature = "agent", not(target_family = "wasm")))]
        let mut publish_agent_snapshot = false;
        let redraw_after_event = state.redraw_policy == RedrawPolicy::Reactive
            && !matches!(
                event,
                WindowEvent::RedrawRequested | WindowEvent::CloseRequested
            );

        // ---------------------------------------------------------------
        // Event routing: egui → filter → InputState / camera / app
        //
        // 1. Forward to egui first and get per-event `consumed` flag.
        // 2. Classify the event:
        //    - keyboard: filtered by `consumed` (egui handles Tab specially)
        //    - pointer:  filtered by `wants_pointer_input()` (covers hover)
        //    - system:   always passed through
        // 3. Only filtered-through events reach InputState, camera, and
        //    on_window_event, so update() and on_window_event() see the
        //    same egui-filtered view.
        // ---------------------------------------------------------------

        let egui_consumed = state.egui.handle_window_event(&state.window, &event);

        // Determine whether this event should be suppressed from the app.
        let is_keyboard_event = matches!(
            event,
            WindowEvent::KeyboardInput { .. } | WindowEvent::Ime(..)
        );
        let is_pointer_event = matches!(
            event,
            WindowEvent::CursorMoved { .. }
                | WindowEvent::CursorEntered { .. }
                | WindowEvent::CursorLeft { .. }
                | WindowEvent::MouseInput { .. }
                | WindowEvent::MouseWheel { .. }
                | WindowEvent::Touch(..)
                | WindowEvent::PinchGesture { .. }
                | WindowEvent::PanGesture { .. }
        );

        // Always update cursor position for the camera (needed for zoom-to-cursor),
        // even when egui captures pointer events.
        if let WindowEvent::CursorMoved { position, .. } = &event {
            state.camera.set_cursor_position(position.x, position.y);
        }

        let pointer_suppress = is_pointer_event && state.egui.wants_pointer_input();
        let suppress = (is_keyboard_event && egui_consumed) || pointer_suppress;

        // When egui captures a category, clear stuck state in InputState.
        if is_keyboard_event && egui_consumed {
            state.input.clear_keyboard();
        }
        if pointer_suppress {
            // End any active drag when egui starts capturing pointer.
            if !state.pointer_suppressed {
                state.camera.on_drag_end();
                state.pointer_suppressed = true;
            }
            state.input.clear_pointer();
        } else if is_pointer_event {
            state.pointer_suppressed = false;
        }

        // Update InputState and dispatch to camera only for non-suppressed events.
        if !suppress {
            state.input.handle_event(&event);

            // Camera input
            match &event {
                WindowEvent::CursorMoved { .. } => {
                    let (dx, dy) = state.input.event_mouse_delta;
                    let buttons = &state.input.mouse_buttons_down;
                    // Only dispatch drag when a button is pressed and there's actual movement.
                    if (buttons.left || buttons.right || buttons.middle) && (dx != 0.0 || dy != 0.0)
                    {
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
                    state.camera.on_scroll(state.input.event_scroll_delta);
                }
                WindowEvent::MouseInput {
                    state: winit::event::ElementState::Released,
                    ..
                } => {
                    // Only end drag when all buttons are released.
                    let buttons = &state.input.mouse_buttons_down;
                    if !buttons.left && !buttons.right && !buttons.middle {
                        state.camera.on_drag_end();
                    }
                }
                // Touch gestures: runner tracks state, camera interprets
                WindowEvent::Touch(touch) => {
                    if let Some(action) = state.touch_tracker.handle_touch(touch) {
                        match action {
                            TouchGestureAction::OneDrag { dx, dy } => {
                                state.camera.on_touch_drag(dx, dy);
                            }
                            TouchGestureAction::OneDragEnd => {
                                state.camera.on_touch_drag_end();
                            }
                            TouchGestureAction::TwoFinger {
                                scroll_delta,
                                pan_dx,
                                pan_dy,
                                midpoint,
                            } => {
                                state.camera.on_pinch_pan(
                                    scroll_delta,
                                    pan_dx,
                                    pan_dy,
                                    Some(midpoint),
                                );
                            }
                        }
                    }
                }
                // Trackpad pinch gesture (native)
                WindowEvent::PinchGesture { delta, .. } => {
                    state.camera.on_pinch_pan(*delta as f32, 0.0, 0.0, None);
                }
                // Trackpad pan gesture (native)
                WindowEvent::PanGesture { delta, .. } => {
                    state
                        .camera
                        .on_pinch_pan(0.0, delta.x as f64, delta.y as f64, None);
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
                    #[cfg(target_family = "wasm")]
                    {
                        state.pending_resize = Some(new_size);
                    }
                    #[cfg(not(target_family = "wasm"))]
                    {
                        state.surface.resize(&state.gpu.device, new_size);
                        state.egui.resize(
                            new_size.width,
                            new_size.height,
                            crate::egui_integration::EguiIntegration::compute_pixels_per_point(
                                &state.window,
                            ),
                        );
                        state
                            .camera
                            .set_viewport_size(new_size.width, new_size.height);
                        if let Some(app) = self.app.as_mut() {
                            app.resize(&state.gpu, new_size);
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(app) = self.app.as_mut() {
                    let outcome = render_frame(app, state);
                    if outcome.exit {
                        event_loop.exit();
                    }
                    if outcome.request_redraw {
                        state.window.request_redraw();
                    }
                    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
                    {
                        publish_agent_snapshot = true;
                    }
                }
            }
            ref other => {
                if !suppress && let Some(app) = self.app.as_mut() {
                    app.on_window_event(other);
                }
            }
        }

        if redraw_after_event {
            state.window.request_redraw();
        }

        #[cfg(all(feature = "agent", not(target_family = "wasm")))]
        if publish_agent_snapshot {
            self.publish_agent_snapshot();
        }
    }
}

// --- レンダリング ---

struct RenderFrameOutcome {
    request_redraw: bool,
    exit: bool,
}

fn render_frame<A: App>(app: &mut A, state: &mut RunState<A::Camera>) -> RenderFrameOutcome {
    state.frame_time.tick();

    // WASM: Resized イベントで蓄積されたリサイズをここで一括適用
    #[cfg(target_family = "wasm")]
    if let Some(new_size) = state.pending_resize.take() {
        state.surface.resize(&state.gpu.device, new_size);
        state.egui.resize(
            new_size.width,
            new_size.height,
            crate::egui_integration::EguiIntegration::compute_pixels_per_point(&state.window),
        );
        state
            .camera
            .set_viewport_size(new_size.width, new_size.height);
        app.resize(&state.gpu, new_size);
    }

    let size = state.surface.window_size();

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
            camera: &mut state.camera,
            window: &state.window,
        };
        app.update(&mut update_ctx);
    }

    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    let render_screenshot_offscreen =
        state.pending_screenshot.is_some() && !state.surface.surface_copy_supported();
    #[cfg(not(all(feature = "agent", not(target_family = "wasm"))))]
    let render_screenshot_offscreen = false;

    let offscreen_texture = if render_screenshot_offscreen {
        Some(create_offscreen_frame_texture(&state.gpu, size))
    } else {
        None
    };

    let surface_texture = if offscreen_texture.is_some() {
        None
    } else {
        match state.surface.surface_texture() {
            Ok(tex) => Some(tex),
            Err(wgpu::SurfaceError::Lost) => {
                state.surface.resize(&state.gpu.device, size);
                return RenderFrameOutcome {
                    request_redraw: true,
                    exit: false,
                };
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                tracing::error!("Out of GPU memory; exiting render loop");
                return RenderFrameOutcome {
                    request_redraw: false,
                    exit: true,
                };
            }
            Err(e) => {
                tracing::warn!("Surface error: {e:?}");
                return RenderFrameOutcome {
                    request_redraw: true,
                    exit: false,
                };
            }
        }
    };

    let target_view = if let Some(texture) = &offscreen_texture {
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    } else {
        surface_texture
            .as_ref()
            .expect("surface texture must exist for surface rendering")
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                // render_format は sRGB view format（WebGPU で view_formats 経由）
                format: Some(state.surface.render_format()),
                ..Default::default()
            })
    };

    // MSAA: App は msaa_view に描画し、surface_view に resolve する。
    // sample_count == 1 の場合は target_view に直接描画。
    let render_view = state.surface.msaa_view().unwrap_or(&target_view);
    let resolve_target = state.surface.msaa_view().map(|_| &target_view);

    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mikage_frame_encoder"),
        });

    // Encode (compute + render)
    {
        let mut frame_ctx = FrameContext {
            gpu: &state.gpu,
            encoder: &mut encoder,
            surface_view: render_view,
            resolve_target,
            window_size: size,
            camera: &state.camera,
        };
        app.encode(&mut frame_ctx);
    }

    // egui: UI 構築 + 描画
    state.egui.render(
        &state.window,
        &state.gpu,
        &state.surface,
        &mut encoder,
        &target_view,
        |egui_ctx| {
            app.gui(egui_ctx);
        },
    );

    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    let screenshot_readback = if state.pending_screenshot.is_some() {
        let (source, format) = if let Some(texture) = &offscreen_texture {
            (texture, state.gpu.render_format())
        } else {
            let surface_texture = surface_texture
                .as_ref()
                .expect("surface texture must exist for surface screenshot");
            (&surface_texture.texture, state.surface.surface_format())
        };
        match crate::agent::ScreenshotReadback::encode(
            &state.gpu.device,
            &mut encoder,
            source,
            size,
            format,
        ) {
            Ok(readback) => Some(readback),
            Err(message) => {
                if let Some(respond_to) = state.pending_screenshot.take() {
                    let _ = respond_to.send(crate::agent::AgentResponse::internal(message));
                }
                None
            }
        }
    } else {
        None
    };

    state.gpu.queue.submit(std::iter::once(encoder.finish()));
    if let Some(surface_texture) = surface_texture {
        surface_texture.present();
    }

    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    if let Some(readback) = screenshot_readback {
        if let Some(respond_to) = state.pending_screenshot.take() {
            let _ = respond_to.send(readback.read_png(&state.gpu.device));
        }
    }

    let request_redraw = state.redraw_policy == RedrawPolicy::Continuous
        || state.egui.has_immediate_repaint_request();

    // フレーム末尾で per-frame 入力状態をリセット。
    // 次フレームの window_event() で蓄積されたイベントが
    // その次の render_frame() の app.update() で見えるようにする。
    state.input.end_frame();

    RenderFrameOutcome {
        request_redraw,
        exit: false,
    }
}

fn create_offscreen_frame_texture(gpu: &GpuContext, size: PhysicalSize<u32>) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mikage_offscreen_frame"),
        size: wgpu::Extent3d {
            width: size.width.max(1),
            height: size.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.render_format(),
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

#[cfg(target_family = "wasm")]
fn show_wasm_gpu_error(window: &Window, message: &str) {
    use wasm_bindgen::JsCast;
    use winit::platform::web::WindowExtWebSys;

    let Some(canvas) = window.canvas() else {
        return;
    };
    let Some(document) = web_sys::window().and_then(|window| window.document()) else {
        return;
    };
    let Ok(element) = document.create_element("div") else {
        return;
    };
    element.set_text_content(Some(&format!(
        "WebGPU is not available in this browser.\n{message}"
    )));
    let _ = element.set_attribute(
        "style",
        "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;\
         box-sizing:border-box;padding:24px;background:#101014;color:#f5f5f5;\
         font:14px system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;\
         line-height:1.45;text-align:center;white-space:pre-wrap;",
    );

    if let Some(parent) = canvas.parent_node() {
        let _ = parent.append_child(&element);
    } else if let Some(body) = document.body() {
        let _ = body.append_child(&element);
    }
    let _ = canvas.dyn_ref::<web_sys::Element>().map(|canvas| {
        canvas.set_attribute("style", "display:none;").ok();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let config = RunConfig::default();
        assert_eq!(config.title, "mikage");
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert_eq!(config.sample_count, 1);
        assert!(config.init_logging);
        assert!(config.canvas.is_none());
        assert_eq!(config.present_mode, wgpu::PresentMode::AutoVsync);
        assert_eq!(config.redraw_policy, RedrawPolicy::Continuous);
        assert_eq!(
            config.pixel_scroll_per_line,
            InputState::DEFAULT_PIXEL_SCROLL_PER_LINE
        );
        assert_eq!(
            config.touch_pinch_sensitivity,
            TouchTracker::DEFAULT_PINCH_SENSITIVITY
        );
    }

    #[test]
    fn new_sets_title() {
        let config = RunConfig::new("Test App");
        assert_eq!(config.title, "Test App");
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert_eq!(config.sample_count, 1);
        assert!(config.init_logging);
        assert!(config.canvas.is_none());
        assert_eq!(config.present_mode, wgpu::PresentMode::AutoVsync);
        assert_eq!(config.redraw_policy, RedrawPolicy::Continuous);
    }

    #[test]
    fn with_title_builder() {
        let config = RunConfig::new("A").with_title("B");
        assert_eq!(config.title, "B");
    }

    #[test]
    fn with_size_builder() {
        let config = RunConfig::new("X").with_size(1920, 1080);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn with_camera_changes_type() {
        let config = RunConfig::new("X")
            .with_redraw_policy(RedrawPolicy::Reactive)
            .with_pixel_scroll_per_line(100.0)
            .with_touch_pinch_sensitivity(2.0)
            .with_camera(crate::camera::camera2d::Camera2d::default());
        assert_eq!(config.title, "X");
        assert_eq!(config.redraw_policy, RedrawPolicy::Reactive);
        assert_eq!(config.pixel_scroll_per_line, 100.0);
        assert_eq!(config.touch_pinch_sensitivity, 2.0);
        // Verify the camera field exists and is accessible
        let _camera: &crate::camera::camera2d::Camera2d = &config.camera;
    }

    #[test]
    fn unit_camera_config_supports_default() {
        let config: RunConfig<()> = RunConfig {
            title: "No Camera".to_string(),
            ..Default::default()
        };
        assert_eq!(config.title, "No Camera");
        assert_eq!(config.camera, ());
    }

    #[test]
    fn with_defaults_preserves_defaults() {
        let config = RunConfig::with_defaults(crate::camera::camera2d::Camera2d::default());
        assert_eq!(config.title, "mikage");
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
    }
}
