use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::app::{App, FrameContext, UpdateContext};
use crate::camera::{InteractiveCamera, OrbitCamera};
use crate::context::GpuContext;
use crate::egui_integration::EguiIntegration;
use crate::input::InputState;
use crate::time::FrameTime;

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
    /// Typical values: 1, 4. The framework stores this value and exposes it
    /// via [`GpuContext::sample_count`](crate::GpuContext::sample_count);
    /// applications use it when creating render pipelines.
    pub sample_count: u32,
    /// CSS selector for an existing `<canvas>` element (e.g., `"#my-canvas"`).
    ///
    /// If `Some`, the specified canvas is used and its size is controlled by the user's CSS.
    /// If `None` (default), a canvas is created automatically and sized to fill the viewport.
    /// This option is only used on WASM; it is ignored on native platforms.
    pub canvas: Option<String>,
}

impl Default for RunConfig<OrbitCamera> {
    fn default() -> Self {
        Self {
            title: "mikage".to_string(),
            width: 1280,
            height: 720,
            camera: OrbitCamera::default(),
            present_mode: wgpu::PresentMode::AutoVsync,
            wgpu_features: wgpu::Features::empty(),
            wgpu_limits: None,
            init_logging: true,
            sample_count: 1,
            canvas: None,
        }
    }
}

impl RunConfig<OrbitCamera> {
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
        }
    }

    /// Replaces the camera, changing the type parameter.
    pub fn with_camera<C2: InteractiveCamera>(self, camera: C2) -> RunConfig<C2> {
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
    egui: EguiIntegration,
    input: InputState,
    camera: C,
    frame_time: FrameTime,
    touch_tracker: TouchTracker,
    /// WASM: Resized イベントを render_frame の先頭まで遅延させる。
    /// 連続リサイズ中に毎イベント surface.configure() が走るのを防ぐ。
    #[cfg(target_family = "wasm")]
    pending_resize: Option<PhysicalSize<u32>>,
}

// --- タッチ・ジェスチャー入力 ---

/// Tracks touch gestures: one-finger drag (orbit) and two-finger pinch/pan.
#[derive(Default)]
struct TouchTracker {
    /// Active touch points.
    touches: std::collections::HashMap<u64, (f64, f64)>,
    /// Previous distance between two fingers (for pinch detection).
    prev_pinch_distance: Option<f64>,
    /// Previous midpoint of two fingers (for two-finger pan).
    prev_midpoint: Option<(f64, f64)>,
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
    },
}

impl TouchTracker {
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
                            .map(|prev| ((dist / prev) - 1.0) as f32 * 5.0)
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
    /// WASM: async GPU 初期化の完了を受け取るための共有スロット
    #[cfg(target_family = "wasm")]
    pending_gpu: Option<PendingGpuInit<A::Camera>>,
}

/// WASM 用: async GPU 初期化の完了待ち
#[cfg(target_family = "wasm")]
struct PendingGpuInit<C: InteractiveCamera> {
    window: Arc<Window>,
    camera: C,
    slot: std::rc::Rc<std::cell::RefCell<Option<GpuContext>>>,
    /// GPU 初期化中に届いた最新の Resized イベントをバッファリングする。
    /// winit の ResizeObserver は非同期に発火するため、初期化完了前に
    /// 正しいサイズの Resized が届くことがある。
    buffered_resize: Option<PhysicalSize<u32>>,
}

impl<A: App> AppHandler<A> {
    fn new(init_fn: InitFn<A>, config: RunConfig<A::Camera>) -> Self {
        Self {
            app: None,
            init_fn: Some(init_fn),
            config: Some(config),
            state: None,
            #[cfg(target_family = "wasm")]
            pending_gpu: None,
        }
    }

    /// GPU 初期化完了後の共通セットアップ
    fn complete_init(&mut self, window: Arc<Window>, gpu: GpuContext, camera: A::Camera) {
        let egui = EguiIntegration::new(&window, &gpu);
        let size = gpu.window_size();

        tracing::info!("App init with size: {}x{}", size.width, size.height);
        let init_fn = self.init_fn.take().expect("init_fn already consumed");
        self.app = Some(init_fn(&gpu, size));

        // Notify camera of initial viewport size
        let mut camera = camera;
        camera.set_viewport_size(size.width, size.height);

        self.state = Some(RunState {
            window,
            gpu,
            egui,
            input: InputState::default(),
            camera,
            frame_time: FrameTime::new(),
            touch_tracker: TouchTracker::default(),
            #[cfg(target_family = "wasm")]
            pending_resize: None,
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
            let gpu = pollster::block_on(GpuContext::new(
                window.clone(),
                config.present_mode,
                config.wgpu_features,
                config.wgpu_limits.clone(),
                config.sample_count,
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
            let sample_count = config.sample_count;
            wasm_bindgen_futures::spawn_local(async move {
                let gpu = GpuContext::new(
                    window_clone,
                    present_mode,
                    wgpu_features,
                    wgpu_limits,
                    sample_count,
                )
                .await;
                *slot_clone.borrow_mut() = Some(gpu);
            });

            self.pending_gpu = Some(PendingGpuInit {
                window: window.clone(),
                camera: config.camera,
                slot,
                buffered_resize: None,
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
            // GPU 初期化中に届いた Resized イベントをバッファリング
            if let WindowEvent::Resized(size) = &event {
                if let Some(pending) = &mut self.pending_gpu {
                    if size.width > 0 && size.height > 0 {
                        pending.buffered_resize = Some(*size);
                    }
                }
            }

            if let Some(pending) = self.pending_gpu.take() {
                if let Some(mut gpu) = pending.slot.borrow_mut().take() {
                    // 初期化完了: バッファされた Resized を適用
                    if let Some(size) = pending.buffered_resize {
                        gpu.resize(size);
                    }
                    let window = pending.window.clone();
                    self.complete_init(pending.window, gpu, pending.camera);
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

        let suppress = (is_keyboard_event && egui_consumed)
            || (is_pointer_event && state.egui.wants_pointer_input());

        // When egui captures a category, clear stuck state in InputState.
        if is_keyboard_event && egui_consumed {
            state.input.clear_keyboard();
        }
        if is_pointer_event && state.egui.wants_pointer_input() {
            state.input.clear_pointer();
        }

        // Update InputState and dispatch to camera only for non-suppressed events.
        if !suppress {
            state.input.handle_event(&event);

            // Camera input
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
                WindowEvent::MouseInput {
                    state: winit::event::ElementState::Released,
                    ..
                } => {
                    state.camera.on_drag_end();
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
                            } => {
                                state.camera.on_pinch_pan(scroll_delta, pan_dx, pan_dy);
                            }
                        }
                    }
                }
                // Trackpad pinch gesture (native)
                WindowEvent::PinchGesture { delta, .. } => {
                    state.camera.on_pinch_pan(*delta as f32, 0.0, 0.0);
                }
                // Trackpad pan gesture (native)
                WindowEvent::PanGesture { delta, .. } => {
                    state
                        .camera
                        .on_pinch_pan(0.0, delta.x as f64, delta.y as f64);
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
                        state.gpu.resize(new_size);
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
                    render_frame(app, state);
                }
            }
            ref other => {
                if !suppress && let Some(app) = self.app.as_mut() {
                    app.on_window_event(other);
                }
            }
        }
    }
}

// --- レンダリング ---

fn render_frame<A: App>(app: &mut A, state: &mut RunState<A::Camera>) {
    state.frame_time.tick();

    // WASM: Resized イベントで蓄積されたリサイズをここで一括適用
    #[cfg(target_family = "wasm")]
    if let Some(new_size) = state.pending_resize.take() {
        state.gpu.resize(new_size);
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
            camera: &mut state.camera,
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

    // MSAA: App は msaa_view に描画し、surface_view に resolve する。
    // sample_count == 1 の場合は surface_view に直接描画。
    let render_view = state.gpu.msaa_view().unwrap_or(&surface_view);
    let resolve_target = state.gpu.msaa_view().map(|_| &surface_view);

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
    state.input.end_frame();
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
        let config = RunConfig::new("X").with_camera(crate::camera::camera2d::Camera2d::default());
        assert_eq!(config.title, "X");
        // Verify the camera field exists and is accessible
        let _camera: &crate::camera::camera2d::Camera2d = &config.camera;
    }

    #[test]
    fn with_defaults_preserves_defaults() {
        let config = RunConfig::with_defaults(crate::camera::camera2d::Camera2d::default());
        assert_eq!(config.title, "mikage");
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
    }
}
