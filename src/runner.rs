//! winit host for the portable [`AppRuntime`].

use crate::egui_integration::EguiIntegration;
use crate::input::InputState;
use crate::{
    App, AppRuntime, GpuContext, GpuDescriptor, InteractiveCamera, OrbitCamera, RenderTarget,
    RenderTargetConfig, RuntimeError, SurfaceContext, SurfaceDescriptor, WindowInputContext,
};
use dpi::{LogicalSize, PhysicalSize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
#[cfg(not(target_family = "wasm"))]
use std::time::Instant;
#[cfg(target_family = "wasm")]
use web_time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowAttributes, WindowId};

mod presentation;
mod simulation;
use presentation::Presentation;
pub use presentation::PresentationPolicy;
use simulation::AutomaticSimulation;
pub use simulation::SimulationControl;

#[cfg(all(feature = "agent", not(target_family = "wasm")))]
#[path = "runner/agent_driver.rs"]
mod agent_driver;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RedrawPolicy {
    /// Draw continuously, including when the camera and GUI are idle.
    #[default]
    Continuous,
    /// Draw on input or invalidation, and while camera/GUI animations are active.
    Reactive,
}

/// A cloneable, thread-safe request to redraw the window, independent of egui.
///
/// The runner supplies this handle through [`App::set_redraw_request`]. Store it
/// in the app or a background worker and call [`request_redraw`](Self::request_redraw)
/// after changing visible state. Requests are coalesced until the runner handles
/// them, including requests made during rendering. This does not advance a
/// `Manual` simulation or bypass GPU submission limits.
#[derive(Clone)]
pub struct RedrawRequest {
    pending: Arc<AtomicBool>,
    wake: Arc<dyn Fn() + Send + Sync>,
}

impl RedrawRequest {
    fn new(wake: Arc<dyn Fn() + Send + Sync>) -> Self {
        Self {
            pending: Arc::new(AtomicBool::new(false)),
            wake,
        }
    }

    pub fn request_redraw(&self) {
        if !self.pending.swap(true, Ordering::AcqRel) {
            (self.wake)();
        }
    }

    fn take_pending(&self) -> bool {
        self.pending.swap(false, Ordering::AcqRel)
    }
}

/// Camera time excludes reactive idle time, independently of simulation time.
struct CameraClock {
    last_update: Option<Instant>,
}

impl CameraClock {
    fn new(now: Instant) -> Self {
        Self {
            last_update: Some(now),
        }
    }

    fn before_change(&mut self, now: Instant, animating: bool) {
        // In particular, releasing a held drag starts inertia *now*, even if
        // the last drag/render happened a long time ago.
        if !animating {
            self.last_update = Some(now);
        }
    }

    fn advance(&mut self, now: Instant) -> Duration {
        let dt = self.last_update.map_or(Duration::ZERO, |last| {
            now.duration_since(last).min(Duration::from_millis(250))
        });
        self.last_update = Some(now);
        dt
    }

    fn advance_frame(&mut self, now: Instant, occluded: bool) -> Duration {
        if occluded {
            self.last_update = Some(now);
            Duration::ZERO
        } else {
            self.advance(now)
        }
    }

    fn finish_frame(
        &mut self,
        policy: RedrawPolicy,
        camera_animating: bool,
        repaint: Duration,
    ) -> bool {
        let keep_updating = policy == RedrawPolicy::Continuous || camera_animating;
        if !keep_updating {
            self.last_update = None;
        }
        keep_updating || repaint.is_zero()
    }
}

// A capture renders privately; only an ordinary frame consumes the window's
// pending invalidation and GUI deadline.
fn frame_requests(
    capture: bool,
    previous: (bool, Option<Instant>),
    current: (bool, Option<Instant>),
) -> (bool, Option<Instant>) {
    if !capture {
        return current;
    }
    let deadline = match (previous.1, current.1) {
        (Some(old), Some(new)) => Some(old.min(new)),
        (old, new) => old.or(new),
    };
    (previous.0 || current.0, deadline)
}

/// Simulation scheduling is independent of presentation scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimulationPolicy {
    /// One variable-duration tick for an ordinary redraw. Capture-only redraws do not tick.
    #[default]
    PerRedraw,
    /// Timer-driven fixed steps, with bounded catch-up work per event-loop wake.
    Fixed {
        dt: Duration,
        max_ticks_per_wake: usize,
        max_backlog: Duration,
    },
    /// Only explicit runtime/agent requests advance the simulation.
    Manual,
}
impl SimulationPolicy {
    pub fn fixed(dt: Duration) -> Self {
        assert!(!dt.is_zero(), "fixed tick duration must be positive");
        Self::Fixed {
            dt,
            max_ticks_per_wake: 8,
            max_backlog: Duration::from_millis(250),
        }
    }
}

pub struct RunConfig<C: InteractiveCamera = OrbitCamera> {
    pub title: String,
    /// Initial window width in logical pixels, scaled by the display's scale factor.
    pub width: u32,
    /// Initial window height in logical pixels, scaled by the display's scale factor.
    pub height: u32,
    pub camera: C,
    pub gpu: GpuDescriptor,
    pub present_mode: wgpu::PresentMode,
    pub sample_count: u32,
    pub init_logging: bool,
    pub canvas: Option<String>,
    pub redraw_policy: RedrawPolicy,
    /// Frame availability/backpressure, independent of the simulation clock.
    pub presentation_policy: PresentationPolicy,
    pub simulation_policy: SimulationPolicy,
    pub max_in_flight_submissions: usize,
    pub pixel_scroll_per_line: f32,
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
    #[allow(clippy::should_implement_trait)] // Selects OrbitCamera when the generic camera is not otherwise inferred.
    pub fn default() -> Self {
        <Self as Default>::default()
    }
    pub fn new(title: impl Into<String>) -> Self {
        Self::default().with_title(title)
    }
}
impl<C: InteractiveCamera> RunConfig<C> {
    pub fn with_defaults(camera: C) -> Self {
        Self {
            title: "mikage".into(),
            width: 1280,
            height: 720,
            camera,
            gpu: GpuDescriptor::default(),
            present_mode: wgpu::PresentMode::AutoVsync,
            sample_count: 1,
            init_logging: true,
            canvas: None,
            redraw_policy: RedrawPolicy::Continuous,
            presentation_policy: PresentationPolicy::Immediate,
            simulation_policy: SimulationPolicy::PerRedraw,
            max_in_flight_submissions: 8,
            pixel_scroll_per_line: 50.0,
            touch_pinch_sensitivity: 5.0,
        }
    }
    pub fn with_camera<C2: InteractiveCamera>(self, camera: C2) -> RunConfig<C2> {
        RunConfig {
            title: self.title,
            width: self.width,
            height: self.height,
            camera,
            gpu: self.gpu,
            present_mode: self.present_mode,
            sample_count: self.sample_count,
            init_logging: self.init_logging,
            canvas: self.canvas,
            redraw_policy: self.redraw_policy,
            presentation_policy: self.presentation_policy,
            simulation_policy: self.simulation_policy,
            max_in_flight_submissions: self.max_in_flight_submissions,
            pixel_scroll_per_line: self.pixel_scroll_per_line,
            touch_pinch_sensitivity: self.touch_pinch_sensitivity,
        }
    }
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }
    /// Sets the initial window size in logical pixels.
    ///
    /// For example, 800 × 600 uses 1600 × 1200 physical pixels on a display
    /// with a scale factor of 2. The app factory and resize hooks receive the
    /// actual physical size. On WASM, the canvas's CSS layout controls its size.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }
    pub fn with_present_mode(mut self, mode: wgpu::PresentMode) -> Self {
        self.present_mode = mode;
        self
    }
    pub fn with_canvas(mut self, selector: impl Into<String>) -> Self {
        self.canvas = Some(selector.into());
        self
    }
    pub fn with_redraw_policy(mut self, policy: RedrawPolicy) -> Self {
        self.redraw_policy = policy;
        self
    }
    /// Selects display-paced presentation independently of simulation scheduling.
    /// Vsync requires a VSync present mode and leaves fixed ticks on their own clock.
    pub fn with_presentation_policy(mut self, policy: PresentationPolicy) -> Self {
        self.presentation_policy = policy;
        self
    }
    pub fn with_simulation_policy(mut self, policy: SimulationPolicy) -> Self {
        self.simulation_policy = policy;
        self
    }
    pub fn with_gpu(mut self, gpu: GpuDescriptor) -> Self {
        self.gpu = gpu;
        self
    }
    pub fn with_pixel_scroll_per_line(mut self, pixels: f32) -> Self {
        self.pixel_scroll_per_line = pixels.max(1.0);
        self
    }
    pub fn with_touch_pinch_sensitivity(mut self, value: f32) -> Self {
        self.touch_pinch_sensitivity = value.max(0.0);
        self
    }
}

type InitFn<A> = Box<dyn FnOnce(&GpuContext, RenderTargetConfig, PhysicalSize<u32>) -> A>;

/// A startup or terminal failure of the window runner.
#[derive(Debug)]
pub enum RunError {
    EventLoop(winit::error::EventLoopError),
    Window(winit::error::OsError),
    Gpu(crate::GpuInitError),
    Runtime(RuntimeError),
    InvalidConfig(&'static str),
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    Agent(std::io::Error),
    #[cfg(target_family = "wasm")]
    Canvas(String),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EventLoop(error) => write!(f, "event loop failed: {error}"),
            Self::Window(error) => write!(f, "window creation failed: {error}"),
            Self::Gpu(error) => write!(f, "GPU/surface failed: {error}"),
            Self::Runtime(error) => write!(f, "application runtime failed: {error}"),
            Self::InvalidConfig(message) => write!(f, "invalid run configuration: {message}"),
            #[cfg(all(feature = "agent", not(target_family = "wasm")))]
            Self::Agent(error) => write!(f, "agent startup failed: {error}"),
            #[cfg(target_family = "wasm")]
            Self::Canvas(message) => write!(f, "canvas setup failed: {message}"),
        }
    }
}
impl std::error::Error for RunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::EventLoop(error) => Some(error),
            Self::Window(error) => Some(error),
            Self::Gpu(error) => Some(error),
            Self::Runtime(error) => Some(error),
            #[cfg(all(feature = "agent", not(target_family = "wasm")))]
            Self::Agent(error) => Some(error),
            _ => None,
        }
    }
}
impl From<crate::GpuInitError> for RunError {
    fn from(error: crate::GpuInitError) -> Self {
        Self::Gpu(error)
    }
}
impl From<RuntimeError> for RunError {
    fn from(error: RuntimeError) -> Self {
        Self::Runtime(error)
    }
}

fn validate_run_config<C: InteractiveCamera>(config: &RunConfig<C>) -> Result<(), RunError> {
    if config.presentation_policy == PresentationPolicy::Vsync
        && !matches!(
            config.present_mode,
            wgpu::PresentMode::AutoVsync | wgpu::PresentMode::Fifo | wgpu::PresentMode::FifoRelaxed
        )
    {
        return Err(RunError::InvalidConfig(
            "Vsync presentation requires a VSync present mode",
        ));
    }
    if config.max_in_flight_submissions < 2 {
        return Err(RunError::InvalidConfig(
            "window runtime requires at least two submission slots",
        ));
    }
    if let SimulationPolicy::Fixed {
        dt,
        max_ticks_per_wake,
        max_backlog,
    } = config.simulation_policy
        && (dt.is_zero() || max_ticks_per_wake == 0 || max_backlog < dt)
    {
        return Err(RunError::InvalidConfig("invalid fixed-step policy"));
    }
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
fn finish_run(
    first_error: Option<RunError>,
    event_loop_result: Result<(), winit::error::EventLoopError>,
) -> Result<(), RunError> {
    first_error.map_or(event_loop_result.map_err(RunError::EventLoop), Err)
}

/// Starts a windowed application. GPU selection finishes before invoking the factory.
///
/// Native runs return their first terminal GPU/runtime failure after the event
/// loop exits. Normal window close or agent shutdown returns `Ok(())`.
/// On WASM this returns once the event loop is spawned: later asynchronous
/// initialization/runtime errors are logged and displayed in the page instead
/// of being returned through this already-completed call.
pub fn run<A: App>(
    init: impl FnOnce(&GpuContext, RenderTargetConfig, PhysicalSize<u32>) -> A + 'static,
    config: RunConfig<A::Camera>,
) -> Result<(), RunError> {
    validate_run_config(&config)?;
    if config.init_logging {
        crate::logging::init_logging();
    }
    let event_loop = EventLoop::new().map_err(RunError::EventLoop)?;
    let handler = AppHandler::new(Box::new(init), config, event_loop.create_proxy());
    #[cfg(not(target_family = "wasm"))]
    {
        let mut handler = handler;
        let result = event_loop.run_app(&mut handler);
        handler.finish(result)
    }
    #[cfg(target_family = "wasm")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(handler);
        Ok(())
    }
}

#[cfg(all(feature = "agent", not(target_family = "wasm")))]
/// Starts the native window runner with HTTP diagnostics, returning startup and
/// terminal errors through the same result as [`run`].
pub fn run_with_agent<A: App>(
    init: impl FnOnce(&GpuContext, RenderTargetConfig, PhysicalSize<u32>) -> A + 'static,
    config: RunConfig<A::Camera>,
    agent_config: crate::agent::AgentConfig,
) -> Result<(), RunError> {
    validate_run_config(&config)?;
    if config.init_logging {
        crate::logging::init_logging();
    }
    let event_loop = EventLoop::new().map_err(RunError::EventLoop)?;
    let proxy = event_loop.create_proxy();
    let wake = proxy.clone();
    let bridge = crate::agent::AgentBridge::start(agent_config, move || {
        let _ = wake.send_event(());
    })
    .map_err(RunError::Agent)?;
    let mut handler = AppHandler::new(Box::new(init), config, proxy);
    handler.agent = Some(bridge);
    let result = event_loop.run_app(&mut handler);
    handler.finish(result)
}

struct RunState<A: App> {
    window: Arc<Window>,
    runtime: AppRuntime<A>,
    surface: Option<SurfaceContext<'static>>,
    surface_descriptor: SurfaceDescriptor,
    egui: EguiIntegration,
    input: InputState,
    touch_tracker: TouchTracker,
    redraw_policy: RedrawPolicy,
    presentation: Presentation,
    surface_dirty: bool,
    simulation_policy: SimulationPolicy,
    automatic_simulation: AutomaticSimulation,
    pointer_suppressed: bool,
    input_dirty: bool,
    occluded: bool,
    suspended: bool,
    redraw_pending: bool,
    redraw_request: RedrawRequest,
    camera_clock: CameraClock,
    last_render: Instant,
    last_schedule: Instant,
    backlog: Duration,
    discarded_wall_time: Duration,
    gui_deadline: Option<Instant>,
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    diagnostics: agent_driver::Diagnostics,
}

impl<A: App> RunState<A> {
    fn apply_simulation_control(&mut self, now: Instant) -> bool {
        let changed = self.automatic_simulation.apply_pending(
            now,
            &mut self.backlog,
            &mut self.last_schedule,
        );
        self.redraw_pending |= changed;
        changed
    }

    fn permits_automatic_tick(&self) -> bool {
        #[cfg(all(feature = "agent", not(target_family = "wasm")))]
        if !self
            .diagnostics
            .permits_automatic_tick(self.runtime.progress().submitted_ticks)
        {
            return false;
        }
        self.automatic_simulation.permits_tick()
    }

    fn request_frame(&mut self) -> Result<(), RuntimeError> {
        if self.surface_dirty || self.presentation.acquisition_deferred(Instant::now()) {
            return Ok(());
        }
        // Browsers can reserve a display wake before the previous GPU frame
        // completes. render_frame still gates all GUI/render preparation.
        #[cfg(not(target_family = "wasm"))]
        if self
            .presentation
            .frame_in_flight(self.runtime.progress().completed_submissions)
        {
            return Ok(());
        }
        #[cfg(not(target_family = "wasm"))]
        if let Some(worker) = &mut self.presentation.worker {
            if worker.busy() {
                return Ok(());
            }
            if worker.ready.is_none() {
                worker.acquire(
                    self.surface
                        .as_ref()
                        .expect("active surface")
                        .acquisition_handle(),
                )?;
                return Ok(());
            }
        }
        if self.presentation.queue_redraw() {
            self.window.request_redraw();
        }
        Ok(())
    }

    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    fn capture_available(&self) -> bool {
        !self.suspended
            && !self
                .surface
                .as_ref()
                .expect("active surface")
                .is_suspended()
    }

    fn update_surface(&mut self) -> Result<(), RunError> {
        #[cfg(not(target_family = "wasm"))]
        if let Some(worker) = &mut self.presentation.worker {
            if worker.busy() {
                return Ok(());
            }
            if self.surface_dirty || self.suspended || self.occluded {
                worker.ready.take();
            }
        }
        if self.surface_dirty {
            self.surface
                .as_mut()
                .expect("active surface")
                .resize(&self.runtime.gpu, self.surface_descriptor.size)?;
            self.surface_dirty = false;
        }
        Ok(())
    }

    fn before_camera_change(&mut self) {
        if self.redraw_policy == RedrawPolicy::Reactive {
            self.camera_clock
                .before_change(Instant::now(), self.runtime.camera.needs_redraw());
        }
    }

    fn cancel_pointer_input(&mut self) {
        cancel_pointer_input(
            &mut self.input,
            &mut self.touch_tracker,
            &mut self.runtime.camera,
        );
    }
}
/// Tracks touch gestures: one-finger drag (orbit) and two-finger pinch/pan.
struct TouchTracker {
    /// Active camera-owned touch points; GUI/unknown fingers are never added.
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
    /// Camera touch gesture released or cancelled.
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

impl TouchGestureAction {
    fn dispatch(self, camera: &mut impl InteractiveCamera) {
        match self {
            Self::OneDrag { dx, dy } => camera.on_touch_drag(dx, dy),
            Self::OneDragEnd => camera.on_touch_drag_end(),
            Self::TwoFinger {
                scroll_delta,
                pan_dx,
                pan_dy,
                midpoint,
            } => camera.on_pinch_pan(scroll_delta, pan_dx, pan_dy, Some(midpoint)),
        }
    }
}

struct TouchResponse {
    suppressed: bool,
    action: Option<TouchGestureAction>,
}

fn cancel_pointer_input(
    input: &mut InputState,
    touches: &mut TouchTracker,
    camera: &mut impl InteractiveCamera,
) {
    let buttons = input.mouse_buttons_down;
    let touch_end = touches.cancel();
    input.clear_pointer();
    // The default touch-end hook already forwards to on_drag_end. Call the
    // mouse hook separately only for a held mouse button or a mouse-only reset.
    if touch_end.is_none() || buttons.left || buttons.right || buttons.middle {
        camera.on_drag_end();
    }
    if let Some(action) = touch_end {
        action.dispatch(camera);
    }
}

impl TouchTracker {
    const DEFAULT_PINCH_SENSITIVITY: f32 = 5.0;

    fn new(pinch_sensitivity: f32) -> Self {
        Self {
            pinch_sensitivity,
            ..Default::default()
        }
    }

    /// Only a non-GUI Started event may give a finger to the camera. Once
    /// cancelled, its remaining Moved/Ended events cannot restart the gesture.
    fn handle_touch(&mut self, touch: &winit::event::Touch, captured: bool) -> TouchResponse {
        use winit::event::TouchPhase;
        if captured && touch.phase == TouchPhase::Started {
            return TouchResponse {
                suppressed: true,
                action: self.cancel(),
            };
        }
        let suppressed =
            touch.phase != TouchPhase::Started && !self.touches.contains_key(&touch.id);
        TouchResponse {
            suppressed,
            action: self.update_touch(touch),
        }
    }

    fn cancel(&mut self) -> Option<TouchGestureAction> {
        let had_touches = !self.touches.is_empty();
        self.touches.clear();
        self.prev_pinch_distance = None;
        self.prev_midpoint = None;
        had_touches.then_some(TouchGestureAction::OneDragEnd)
    }

    fn update_touch(&mut self, touch: &winit::event::Touch) -> Option<TouchGestureAction> {
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
                let prev_pos = *self.touches.get(&touch.id)?;
                self.touches.insert(touch.id, new_pos);

                match self.touches.len() {
                    1 => {
                        // One-finger drag
                        let dx = new_pos.0 - prev_pos.0;
                        let dy = new_pos.1 - prev_pos.1;
                        Some(TouchGestureAction::OneDrag { dx, dy })
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
                self.touches.remove(&touch.id)?;
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

struct AppHandler<A: App> {
    init_fn: Option<InitFn<A>>,
    config: Option<RunConfig<A::Camera>>,
    state: Option<RunState<A>>,
    proxy: EventLoopProxy<()>,
    first_error: Option<RunError>,
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    agent: Option<crate::agent::AgentBridge>,
    #[cfg(target_family = "wasm")]
    pending_gpu: Option<PendingGpuInit<A::Camera>>,
}
#[cfg(target_family = "wasm")]
struct PendingGpuInit<C: InteractiveCamera> {
    window: Arc<Window>,
    config: RunConfig<C>,
    descriptor: SurfaceDescriptor,
    slot: std::rc::Rc<
        std::cell::RefCell<
            Option<Result<(GpuContext, SurfaceContext<'static>), crate::GpuInitError>>,
        >,
    >,
    buffered_resize: Option<PhysicalSize<u32>>,
}

impl<A: App> AppHandler<A> {
    fn new(init_fn: InitFn<A>, config: RunConfig<A::Camera>, proxy: EventLoopProxy<()>) -> Self {
        Self {
            init_fn: Some(init_fn),
            config: Some(config),
            state: None,
            proxy,
            first_error: None,
            #[cfg(all(feature = "agent", not(target_family = "wasm")))]
            agent: None,
            #[cfg(target_family = "wasm")]
            pending_gpu: None,
        }
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, error: RunError) {
        if self.first_error.is_none() {
            tracing::error!("{error}");
            #[cfg(target_family = "wasm")]
            show_wasm_gpu_error(
                self.state.as_ref().map(|state| &*state.window),
                &error.to_string(),
            );
            #[cfg(all(feature = "agent", not(target_family = "wasm")))]
            if let Some(agent) = &self.agent {
                agent.fail_all(&error.to_string());
            }
            self.first_error = Some(error);
        }
        event_loop.exit();
    }

    #[cfg(not(target_family = "wasm"))]
    fn finish(
        &mut self,
        event_loop_result: Result<(), winit::error::EventLoopError>,
    ) -> Result<(), RunError> {
        finish_run(self.first_error.take(), event_loop_result)
    }

    fn complete_init(
        &mut self,
        window: Arc<Window>,
        gpu: GpuContext,
        surface: SurfaceContext<'static>,
        _descriptor: SurfaceDescriptor,
        config: RunConfig<A::Camera>,
    ) -> Result<(), RunError> {
        let descriptor = surface.descriptor().clone();
        let mut app = self.init_fn.take().expect("factory called once")(
            &gpu,
            surface.render_target_config(),
            surface.size(),
        );
        let redraw_proxy = self.proxy.clone();
        let redraw_request = RedrawRequest::new(Arc::new(move || {
            let _ = redraw_proxy.send_event(());
        }));
        app.set_redraw_request(redraw_request.clone());
        let simulation_proxy = self.proxy.clone();
        let simulation_control = SimulationControl::new(Arc::new(move || {
            let _ = simulation_proxy.send_event(());
        }));
        app.set_simulation_control(simulation_control.clone());
        let gui_proxy = self.proxy.clone();
        let egui = EguiIntegration::new(
            &window,
            &gpu,
            surface.render_target_config(),
            Arc::new(move || {
                let _ = gui_proxy.send_event(());
            }),
        );
        let mut runtime = AppRuntime::new(gpu, app, config.camera);
        runtime.set_config(crate::RuntimeConfig {
            max_in_flight_submissions: config.max_in_flight_submissions,
        })?;
        let proxy = self.proxy.clone();
        let wake: crate::runtime::RuntimeWaker = Arc::new(move || {
            let _ = proxy.send_event(());
        });
        runtime.set_waker(wake.clone());
        runtime
            .camera
            .set_viewport_size(surface.size().width, surface.size().height);
        self.state = Some(RunState {
            window: window.clone(),
            runtime,
            surface: Some(surface),
            surface_descriptor: descriptor,
            egui,
            input: InputState::new(config.pixel_scroll_per_line),
            touch_tracker: TouchTracker::new(config.touch_pinch_sensitivity),
            redraw_policy: config.redraw_policy,
            presentation: Presentation::new(config.presentation_policy, wake),
            surface_dirty: false,
            simulation_policy: config.simulation_policy,
            automatic_simulation: AutomaticSimulation::new(simulation_control, Instant::now()),
            pointer_suppressed: false,
            input_dirty: false,
            occluded: false,
            suspended: false,
            redraw_pending: true,
            redraw_request,
            camera_clock: CameraClock::new(Instant::now()),
            last_render: Instant::now(),
            last_schedule: Instant::now(),
            backlog: Duration::ZERO,
            discarded_wall_time: Duration::ZERO,
            gui_deadline: None,
            #[cfg(all(feature = "agent", not(target_family = "wasm")))]
            diagnostics: agent_driver::Diagnostics::new(),
        });
        window.request_redraw();
        Ok(())
    }

    fn shutdown(&mut self) {
        if let Some(state) = &mut self.state {
            #[cfg(not(target_family = "wasm"))]
            state.presentation.worker.take();
            state.runtime.shutdown();
        }
        #[cfg(all(feature = "agent", not(target_family = "wasm")))]
        if let Some(agent) = &self.agent {
            agent.fail_all("application is shutting down");
        }
    }

    fn pump(&mut self, event_loop: &ActiveEventLoop) {
        if self.first_error.is_some() || event_loop.exiting() {
            return;
        }
        // Input is delivered before any automatic or diagnostic tick, including
        // run_until_completed requests received in the same winit event batch.
        if let Some(state) = &mut self.state {
            if let Err(error) = state.runtime.poll_completions() {
                self.fail(event_loop, error.into());
                return;
            }
            #[cfg(not(target_family = "wasm"))]
            if let Some(worker) = &mut state.presentation.worker {
                match worker.poll() {
                    Ok(Some(token)) => state.runtime.mark_presented(&token),
                    Ok(None) => {}
                    Err(error) => {
                        self.fail(event_loop, error.into());
                        return;
                    }
                }
            }
            if let Err(error) = state.update_surface() {
                self.fail(event_loop, error);
                return;
            }
            deliver_input(state);
        }
        #[cfg(all(feature = "agent", not(target_family = "wasm")))]
        if let (Some(agent), Some(state)) = (&mut self.agent, &mut self.state) {
            agent_driver::pump(agent, state, event_loop);
        }
        let Some(state) = &mut self.state else {
            return;
        };
        if let Err(err) = state.runtime.poll_completions() {
            self.fail(event_loop, err.into());
            return;
        }
        if event_loop.exiting() {
            return;
        }
        #[cfg(all(feature = "agent", not(target_family = "wasm")))]
        if state.diagnostics.requires_frame(&state.runtime)
            && let Err(err) = render_frame(state)
        {
            self.fail(event_loop, err);
            return;
        }
        let now = Instant::now();
        if state.redraw_request.take_pending() {
            state.redraw_pending = true;
        }
        if let Some(deadline) = state.egui.take_repaint_request() {
            state.gui_deadline = Some(
                state
                    .gui_deadline
                    .map_or(deadline, |existing| existing.min(deadline)),
            );
        }
        state.apply_simulation_control(now);
        let delta = now.duration_since(state.last_schedule);
        state.last_schedule = now;
        let active = state.surface_descriptor.size.width > 0
            && state.surface_descriptor.size.height > 0
            && !state.occluded
            && !state.suspended;
        if active && state.presentation.take_retry(now) {
            state.redraw_pending = true;
        }
        let automatic = state.permits_automatic_tick();
        if active && automatic {
            if let SimulationPolicy::Fixed {
                dt,
                max_ticks_per_wake,
                max_backlog,
            } = state.simulation_policy
            {
                let accumulated = state.backlog.saturating_add(delta);
                state.discarded_wall_time += accumulated.saturating_sub(max_backlog);
                state.backlog = accumulated.min(max_backlog);
                for _ in 0..max_ticks_per_wake {
                    state.apply_simulation_control(Instant::now());
                    if state.presentation.presenting()
                        || !state.permits_automatic_tick()
                        || state.backlog < dt
                        || state.runtime.available_submission_slots()
                            <= state
                                .presentation
                                .tick_reserve(state.runtime.progress().completed_submissions)
                    {
                        break;
                    }
                    let control = &state.automatic_simulation.control;
                    match state.runtime.try_tick_if(dt, || control.permits_tick()) {
                        Ok(Some(_)) => {
                            state.backlog -= dt;
                            state.redraw_pending = true;
                        }
                        Ok(None) | Err(RuntimeError::WouldBlock) => break,
                        Err(err) => {
                            self.fail(event_loop, err.into());
                            return;
                        }
                    }
                }
            }
        } else {
            state.backlog = Duration::ZERO;
        }
        state.apply_simulation_control(Instant::now());
        let now = now.max(state.last_schedule);
        let automatic = state.permits_automatic_tick();
        if active && state.gui_deadline.is_some_and(|deadline| deadline <= now) {
            state.redraw_pending = true;
            state.gui_deadline = None;
        }
        let can_request_frame = state.runtime.available_submission_slots() > 0;
        #[cfg(target_family = "wasm")]
        let can_request_frame =
            can_request_frame || state.presentation.policy == PresentationPolicy::Vsync;
        if active
            && state.redraw_pending
            && can_request_frame
            && let Err(error) = state.request_frame()
        {
            self.fail(event_loop, error.into());
            return;
        }
        // Once dirty, the GUI is already waiting for the next usable surface.
        // Do not keep waking on a GUI deadline while presentation/capacity blocks it.
        let mut deadline = if active && !state.redraw_pending {
            state.gui_deadline
        } else {
            None
        };
        if active && let Some(retry) = state.presentation.retry_deadline() {
            deadline = Some(deadline.map_or(retry, |existing| existing.min(retry)));
        }
        if active
            && automatic
            && !state.presentation.presenting()
            && let SimulationPolicy::Fixed { dt, .. } = state.simulation_policy
        {
            // Completion notifications resume a full queue; do not busy-poll a past deadline.
            if state.runtime.available_submission_slots()
                > state
                    .presentation
                    .tick_reserve(state.runtime.progress().completed_submissions)
            {
                let tick_deadline = now + dt.saturating_sub(state.backlog);
                deadline = Some(deadline.map_or(tick_deadline, |d| d.min(tick_deadline)));
            }
        }
        event_loop.set_control_flow(deadline.map_or(ControlFlow::Wait, ControlFlow::WaitUntil));
        #[cfg(all(feature = "agent", not(target_family = "wasm")))]
        if let Some(agent) = &self.agent {
            agent_driver::publish(agent, state);
        }
    }
}
impl<A: App> Drop for AppHandler<A> {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl<A: App> ApplicationHandler for AppHandler<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.first_error.is_some() || event_loop.exiting() {
            return;
        }
        if let Some(state) = &mut self.state {
            state.suspended = false;
            state.presentation.resumed();
            state.last_schedule = Instant::now();
            state.last_render = Instant::now();
            state.camera_clock = CameraClock::new(Instant::now());
            state.redraw_pending = true;
            state.window.request_redraw();
            return;
        }
        let Some(mut config) = self.config.take() else {
            return;
        };
        config.gpu.display = Some(Arc::new(event_loop.owned_display_handle()));
        let mut attrs = WindowAttributes::default().with_title(&config.title);
        #[cfg(not(target_family = "wasm"))]
        {
            attrs = attrs.with_inner_size(LogicalSize::new(config.width, config.height));
        }
        #[cfg(target_family = "wasm")]
        let auto_canvas = config.canvas.is_none();
        #[cfg(target_family = "wasm")]
        {
            use winit::platform::web::WindowAttributesExtWebSys;
            if let Some(selector) = &config.canvas {
                use wasm_bindgen::JsCast;
                let canvas = (|| -> Result<_, RunError> {
                    let document = web_sys::window()
                        .and_then(|window| window.document())
                        .ok_or_else(|| {
                            RunError::Canvas("browser document is unavailable".into())
                        })?;
                    document
                        .query_selector(selector)
                        .map_err(|error| {
                            RunError::Canvas(format!("invalid selector {selector:?}: {error:?}"))
                        })?
                        .ok_or_else(|| {
                            RunError::Canvas(format!("canvas {selector:?} was not found"))
                        })?
                        .dyn_into::<web_sys::HtmlCanvasElement>()
                        .map_err(|_| {
                            RunError::Canvas(format!("element {selector:?} is not a canvas"))
                        })
                })();
                let canvas = match canvas {
                    Ok(canvas) => canvas,
                    Err(error) => {
                        self.fail(event_loop, error);
                        return;
                    }
                };
                attrs = attrs.with_canvas(Some(canvas));
            } else {
                attrs = attrs
                    .with_inner_size(LogicalSize::new(config.width, config.height))
                    .with_append(true);
            }
        }
        let window = match event_loop.create_window(attrs) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                self.fail(event_loop, RunError::Window(error));
                return;
            }
        };
        #[cfg(target_family = "wasm")]
        if auto_canvas {
            use winit::platform::web::WindowExtWebSys;
            if let Some(canvas) = window.canvas() {
                let _ = canvas.set_attribute("style", "width:100%;height:100%");
            }
        }
        let descriptor = SurfaceDescriptor {
            size: window.inner_size(),
            present_mode: config.present_mode,
            sample_count: config.sample_count,
            desired_maximum_frame_latency: if config.presentation_policy
                == PresentationPolicy::Vsync
            {
                1
            } else {
                2
            },
            ..Default::default()
        };
        #[cfg(not(target_family = "wasm"))]
        {
            let result = pollster::block_on(GpuContext::for_surface(
                window.clone(),
                config.gpu.clone(),
                descriptor.clone(),
            ));
            match result {
                Ok((gpu, surface)) => {
                    if let Err(error) = self.complete_init(window, gpu, surface, descriptor, config)
                    {
                        self.fail(event_loop, error);
                    }
                }
                Err(error) => self.fail(event_loop, error.into()),
            }
        }
        #[cfg(target_family = "wasm")]
        {
            let slot = std::rc::Rc::new(std::cell::RefCell::new(None));
            let done = slot.clone();
            let target = window.clone();
            let gpu = config.gpu.clone();
            let surface = descriptor.clone();
            let proxy = self.proxy.clone();
            wasm_bindgen_futures::spawn_local(async move {
                *done.borrow_mut() = Some(GpuContext::for_surface(target, gpu, surface).await);
                let _ = proxy.send_event(());
            });
            self.pending_gpu = Some(PendingGpuInit {
                window,
                config,
                descriptor,
                slot,
                buffered_resize: None,
            });
        }
    }
    fn suspended(&mut self, _: &ActiveEventLoop) {
        if let Some(state) = &mut self.state {
            state.before_camera_change();
            state.input.clear_keyboard();
            state.cancel_pointer_input();
            state.pointer_suppressed = false;
            state.input_dirty = true;
            state.suspended = true;
            state.presentation.redraw_received();
            state.backlog = Duration::ZERO;
        }
    }
    fn exiting(&mut self, _: &ActiveEventLoop) {
        self.shutdown();
    }
    // Native wakes use ApplicationHandler's no-op user_event. macOS drains user
    // events before redraws and AboutToWait: ticking here can perpetually add
    // completion events to that drain. AboutToWait schedules one bounded pass
    // after winit has delivered the batch's window events.
    #[cfg(target_family = "wasm")]
    fn user_event(&mut self, event_loop: &ActiveEventLoop, _: ()) {
        if self.first_error.is_some() || event_loop.exiting() {
            return;
        }
        if let Some(pending) = self.pending_gpu.take() {
            let result = pending.slot.borrow_mut().take();
            if let Some(result) = result {
                match result {
                    Ok((gpu, mut surface)) => {
                        if let Some(size) = pending.buffered_resize
                            && let Err(error) = surface.resize(&gpu, size)
                        {
                            self.fail(event_loop, error.into());
                            return;
                        }
                        if let Err(error) = self.complete_init(
                            pending.window,
                            gpu,
                            surface,
                            pending.descriptor,
                            pending.config,
                        ) {
                            self.fail(event_loop, error);
                            return;
                        }
                    }
                    Err(err) => {
                        self.fail(event_loop, err.into());
                        return;
                    }
                }
            } else {
                self.pending_gpu = Some(pending);
            }
        }
        self.pump(event_loop);
    }
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.pump(event_loop);
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if self.first_error.is_some() || event_loop.exiting() {
            return;
        }
        #[cfg(target_family = "wasm")]
        if let Some(pending) = &mut self.pending_gpu {
            if let WindowEvent::Resized(size) = &event {
                pending.buffered_resize = Some(*size);
            }
        }
        let Some(state) = &mut self.state else {
            return;
        };
        if !matches!(event, WindowEvent::RedrawRequested) {
            state.before_camera_change();
        }
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
            state
                .runtime
                .camera
                .set_cursor_position(position.x, position.y);
        }

        // Touch lifecycle processing must run even while GUI input is captured.
        // The GUI context has not consumed this event yet, so also hit-test a
        // new finger against the last GUI layout at its actual position.
        let gui_pointer_suppress = match &event {
            WindowEvent::Touch(touch) if touch.phase == winit::event::TouchPhase::Started => {
                state.egui.touch_over_gui(&state.window, touch)
            }
            // Ongoing touches retain the owner chosen at Started. Egui's
            // pointer query may still describe a previous mouse/touch event
            // while this gesture is queued for its next GUI frame.
            WindowEvent::Touch(_) => false,
            _ => is_pointer_event && state.egui.wants_pointer_input(),
        };
        // When egui captures a category, clear stuck state in InputState.
        if is_keyboard_event && egui_consumed {
            state.input.clear_keyboard();
        }
        if gui_pointer_suppress {
            // End any active drag when egui starts capturing pointer.
            if !state.pointer_suppressed {
                state.cancel_pointer_input();
                state.pointer_suppressed = true;
            }
            state.input.clear_pointer();
        } else if is_pointer_event {
            state.pointer_suppressed = false;
        }

        let (pointer_suppress, touch_action) = if let WindowEvent::Touch(touch) = &event {
            let response = state
                .touch_tracker
                .handle_touch(touch, gui_pointer_suppress);
            (response.suppressed, response.action)
        } else {
            (gui_pointer_suppress, None)
        };
        let suppress = (is_keyboard_event && egui_consumed) || pointer_suppress;

        if let Some(action) = touch_action {
            action.dispatch(&mut state.runtime.camera);
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
                        state.runtime.camera.on_mouse_drag(
                            dx,
                            dy,
                            buttons.left,
                            buttons.right,
                            buttons.middle,
                        );
                    }
                }
                WindowEvent::MouseWheel { .. } => {
                    state
                        .runtime
                        .camera
                        .on_scroll(state.input.event_scroll_delta);
                }
                WindowEvent::MouseInput {
                    state: winit::event::ElementState::Released,
                    ..
                } => {
                    // Only end drag when all buttons are released.
                    let buttons = &state.input.mouse_buttons_down;
                    if !buttons.left && !buttons.right && !buttons.middle {
                        state.runtime.camera.on_drag_end();
                    }
                }
                // Trackpad pinch gesture (native)
                WindowEvent::PinchGesture { delta, .. } => {
                    state
                        .runtime
                        .camera
                        .on_pinch_pan(*delta as f32, 0.0, 0.0, None);
                }
                // Trackpad pan gesture (native)
                WindowEvent::PanGesture { delta, .. } => {
                    state
                        .runtime
                        .camera
                        .on_pinch_pan(0.0, delta.x as f64, delta.y as f64, None);
                }
                _ => {}
            }
        }

        if is_keyboard_event || is_pointer_event {
            state.input_dirty = true;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Focused(false) => {
                state.input.clear_keyboard();
                state.cancel_pointer_input();
                state.pointer_suppressed = false;
                state.input_dirty = true;
            }
            WindowEvent::Focused(true) => {
                state.presentation.resumed();
                state.occluded = false;
                state.redraw_pending = true;
                state.last_schedule = Instant::now();
                state.last_render = Instant::now();
                state.camera_clock = CameraClock::new(Instant::now());
            }
            WindowEvent::Occluded(value) => {
                state.presentation.resumed();
                state.occluded = value;
                state.backlog = Duration::ZERO;
                state.last_render = Instant::now();
                state.camera_clock = CameraClock::new(Instant::now());
                state.last_schedule = Instant::now();
                if !value {
                    state.redraw_pending = true;
                }
            }
            WindowEvent::Resized(size) => {
                state.presentation.resumed();
                if size.width > 0 && size.height > 0 {
                    state.occluded = false;
                }
                state.surface_descriptor.size = size;
                state.surface_dirty = true;
                if let Err(err) = state.update_surface() {
                    self.fail(event_loop, err);
                    return;
                }
                state
                    .runtime
                    .camera
                    .set_viewport_size(size.width, size.height);
                state.runtime.app.resize(&state.runtime.gpu, size);
                state.backlog = Duration::ZERO;
                state.last_render = Instant::now();
                state.camera_clock = CameraClock::new(Instant::now());
                state.last_schedule = Instant::now();
                state.redraw_pending = true;
            }
            WindowEvent::RedrawRequested => {
                state.presentation.redraw_received();
                state.redraw_pending = true;
                deliver_input(state);
                if let Err(err) = render_frame(state) {
                    self.fail(event_loop, err);
                    return;
                }
            }
            ref other => {
                if !suppress {
                    state.runtime.app.on_window_event(other);
                }
            }
        }
        if !matches!(
            event,
            WindowEvent::RedrawRequested | WindowEvent::CloseRequested
        ) {
            state.redraw_pending = true;
        }
    }
}

fn deliver_input<A: App>(state: &mut RunState<A>) {
    if state.input_dirty {
        state.before_camera_change();
        state.runtime.app.on_input(&mut WindowInputContext {
            window: &state.window,
            input: &state.input,
            camera: &mut state.runtime.camera,
        });
        state.input.end_frame();
        state.input_dirty = false;
    }
}
#[cfg(target_family = "wasm")]
fn show_wasm_gpu_error(window: Option<&Window>, message: &str) {
    use winit::platform::web::WindowExtWebSys;

    let canvas = window.and_then(|window| window.canvas());
    let Some(document) = web_sys::window().and_then(|window| window.document()) else {
        return;
    };
    let Ok(element) = document.create_element("div") else {
        return;
    };
    element.set_text_content(Some(&format!(
        "The mikage application could not continue.\n{message}"
    )));
    let _ = element.set_attribute(
        "style",
        "position:fixed;inset:0;z-index:2147483647;display:flex;align-items:center;justify-content:center;\
         box-sizing:border-box;padding:24px;background:#101014;color:#f5f5f5;\
         font:14px system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;\
         line-height:1.45;text-align:center;white-space:pre-wrap;",
    );

    if let Some(parent) = canvas.as_ref().and_then(|canvas| canvas.parent_node()) {
        let _ = parent.append_child(&element);
    } else if let Some(body) = document.body() {
        let _ = body.append_child(&element);
    }
    if let Some(canvas) = canvas {
        let _ = canvas.set_attribute("style", "display:none;");
    }
}

fn render_frame<A: App>(state: &mut RunState<A>) -> Result<(), RunError> {
    state.runtime.poll_completions()?;
    state.apply_simulation_control(Instant::now());
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    let capture_due = state.diagnostics.requires_frame(&state.runtime);
    #[cfg(not(all(feature = "agent", not(target_family = "wasm"))))]
    let capture_due = false;
    if state
        .surface
        .as_ref()
        .expect("active surface")
        .is_suspended()
        || state.suspended
        || (state.occluded && !capture_due)
    {
        return Ok(());
    }
    if state.runtime.available_submission_slots() == 0
        || state.surface_dirty
        || state.presentation.presenting()
        || (!capture_due && state.presentation.acquisition_deferred(Instant::now()))
        || state
            .presentation
            .frame_in_flight(state.runtime.progress().completed_submissions)
    {
        if !capture_due {
            state.redraw_pending = true;
        }
        return Ok(());
    }
    // Diagnostic frames have their own render target and do not need a visible
    // swapchain. This also avoids waiting for an acquisition already in progress.
    let (frame, suboptimal) = if capture_due {
        tracing::debug!(occluded = state.occluded, "rendering capture offscreen");
        (None, false)
    } else {
        #[cfg(not(target_family = "wasm"))]
        let acquired = if let Some(worker) = &mut state.presentation.worker {
            let Some(frame) = worker.ready.take() else {
                return Ok(());
            };
            Some(frame)
        } else {
            state
                .surface
                .as_ref()
                .expect("active surface")
                .acquire_surface_texture()
        };
        #[cfg(target_family = "wasm")]
        let acquired = state
            .surface
            .as_ref()
            .expect("active surface")
            .acquire_surface_texture();
        match acquired {
            Some(wgpu::CurrentSurfaceTexture::Success(frame)) => (Some(frame), false),
            Some(wgpu::CurrentSurfaceTexture::Suboptimal(frame)) => (Some(frame), true),
            Some(wgpu::CurrentSurfaceTexture::Outdated) => {
                state
                    .surface
                    .as_mut()
                    .expect("active surface")
                    .reconfigure(&state.runtime.gpu)?;
                state.redraw_pending = true;
                return Ok(());
            }
            Some(wgpu::CurrentSurfaceTexture::Lost) => {
                // DX12 allows only one swapchain for an HWND. Release the old
                // surface before constructing/configuring its replacement.
                drop(state.surface.take());
                state.surface = Some(
                    state
                        .runtime
                        .gpu
                        .attach_surface(state.window.clone(), state.surface_descriptor.clone())?,
                );
                state.redraw_pending = true;
                return Ok(());
            }
            Some(wgpu::CurrentSurfaceTexture::Validation) => {
                return Err(RuntimeError::InvalidTarget.into());
            }
            Some(wgpu::CurrentSurfaceTexture::Occluded) => {
                // An older worker result must not overwrite a newer OS visibility
                // event. Only WindowEvent::Occluded owns the persistent OS state.
                state.presentation.defer_acquire(Instant::now());
                state.redraw_pending = false;
                return Ok(());
            }
            Some(wgpu::CurrentSurfaceTexture::Timeout) | None => {
                state.redraw_pending = false;
                state.presentation.defer_acquire(Instant::now());
                return Ok(());
            }
        }
    };
    let display_dirty = state.redraw_pending;
    let display_deadline = state.gui_deadline;
    let camera_was_animating = state.runtime.camera.needs_redraw();
    let now = Instant::now();
    let dt = if state.occluded {
        Duration::ZERO
    } else {
        now.duration_since(state.last_render)
            .min(Duration::from_millis(250))
    };
    let tick_dt = if state.occluded {
        Duration::ZERO
    } else {
        state
            .automatic_simulation
            .per_redraw_dt(now, state.last_render)
    };
    state.last_render = now;
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    let capture_only = capture_due
        || !state
            .diagnostics
            .permits_automatic_tick(state.runtime.progress().submitted_ticks);
    #[cfg(not(all(feature = "agent", not(target_family = "wasm"))))]
    let capture_only = false;
    if state.simulation_policy == SimulationPolicy::PerRedraw
        && !capture_only
        && state.permits_automatic_tick()
        && state.runtime.available_submission_slots() > 1
    {
        let control = &state.automatic_simulation.control;
        state
            .runtime
            .try_tick_if(tick_dt, || control.permits_tick())?;
    }
    state.apply_simulation_control(Instant::now());
    let camera_dt = state.camera_clock.advance_frame(now, state.occluded);
    state.runtime.camera.update(camera_dt.as_secs_f32());
    let prepared = state.egui.build(&state.window, &mut state.runtime.app);
    let config = state
        .surface
        .as_ref()
        .expect("active surface")
        .render_target_config();
    let size = state.surface.as_ref().expect("active surface").size();
    let surface_view = frame.as_ref().map(|frame| {
        frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(config.color_format),
            ..Default::default()
        })
    });
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    let capture_texture = if capture_due {
        Some(
            state
                .runtime
                .gpu
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("mikage_capture_frame"),
                    size: wgpu::Extent3d {
                        width: size.width,
                        height: size.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: config.color_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                }),
        )
    } else {
        None
    };
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    let capture_view = capture_texture
        .as_ref()
        .map(|t| t.create_view(&Default::default()));
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    let target_view = capture_view
        .as_ref()
        .or(surface_view.as_ref())
        .expect("frame target");
    #[cfg(not(all(feature = "agent", not(target_family = "wasm"))))]
    let target_view = surface_view.as_ref().expect("surface frame target");
    let target = RenderTarget {
        view: state
            .surface
            .as_ref()
            .expect("active surface")
            .msaa_view()
            .unwrap_or(target_view),
        resolve_target: state
            .surface
            .as_ref()
            .expect("active surface")
            .msaa_view()
            .map(|_| target_view),
        depth_view: None,
        size,
        config,
    };
    let egui = &mut state.egui;
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    let diagnostics = &mut state.diagnostics;
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    let checkpoint = state.runtime.progress().clone();
    let token = state.runtime.render_with(target, dt, |_app, ctx| {
        #[cfg(all(feature = "agent", not(target_family = "wasm")))]
        if let Some(texture) = &capture_texture {
            diagnostics.encode_frame_captures("scene", ctx, texture, &checkpoint);
        }
        let extra = egui.encode(ctx.gpu, ctx.encoder, target_view, size, &prepared);
        ctx.extra_command_buffers.extend(extra);
        #[cfg(all(feature = "agent", not(target_family = "wasm")))]
        if let Some(texture) = &capture_texture {
            diagnostics.encode_frame_captures("window", ctx, texture, &checkpoint);
        }
    })?;
    state.egui.finish(prepared);
    state.presentation.submitted(token.id);
    if frame.is_some() && state.presentation.policy == PresentationPolicy::Vsync {
        state.window.pre_present_notify();
    }
    if let Some(frame) = frame {
        #[cfg(not(target_family = "wasm"))]
        if let Some(worker) = &mut state.presentation.worker {
            worker.present(state.runtime.gpu.queue.clone(), frame, token.clone())?;
        } else {
            state.runtime.gpu.queue.present(frame);
            state.runtime.mark_presented(&token);
        }
        #[cfg(target_family = "wasm")]
        {
            state.runtime.gpu.queue.present(frame);
            state.runtime.mark_presented(&token);
        }
    }
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    state.diagnostics.after_frame_submit(&token);
    if suboptimal {
        state.surface_dirty = true;
        state.update_surface()?;
    }
    let repaint = state.egui.repaint_after();
    let repaint_deadline = if repaint == Duration::MAX {
        None
    } else {
        Instant::now().checked_add(repaint)
    };
    let redraw = state.camera_clock.finish_frame(
        state.redraw_policy,
        state.runtime.camera.needs_redraw(),
        repaint,
    );
    (state.redraw_pending, state.gui_deadline) = frame_requests(
        capture_due,
        (display_dirty || camera_was_animating, display_deadline),
        (redraw, repaint_deadline),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(id: u64, phase: winit::event::TouchPhase, x: f64, y: f64) -> winit::event::Touch {
        winit::event::Touch {
            device_id: winit::event::DeviceId::dummy(),
            phase,
            location: winit::dpi::PhysicalPosition::new(x, y),
            force: None,
            id,
        }
    }

    #[test]
    fn gui_taps_with_delayed_capture_leave_no_fingers() {
        use winit::event::TouchPhase::*;
        let mut tracker = TouchTracker::default();
        // Even if Started slips through before egui consumes its queued input,
        // Ended must finish the gesture and remove the ID despite GUI capture.
        for id in 0..3 {
            tracker.handle_touch(&touch(id, Started, 1000.0, 30.0), false);
            let ended = tracker.handle_touch(&touch(id, Ended, 1000.0, 30.0), true);
            assert!(!ended.suppressed);
            assert!(matches!(ended.action, Some(TouchGestureAction::OneDragEnd)));
            assert!(tracker.touches.is_empty());
        }
        tracker.handle_touch(&touch(10, Started, 100.0, 100.0), false);
        let moved = tracker.handle_touch(&touch(10, Moved, 140.0, 120.0), false);
        assert!(!moved.suppressed);
        assert!(matches!(
            moved.action,
            Some(TouchGestureAction::OneDrag { dx: 40.0, dy: 20.0 })
        ));
        tracker.handle_touch(&touch(10, Ended, 140.0, 120.0), false);

        tracker.handle_touch(&touch(11, Started, 100.0, 100.0), false);
        tracker.handle_touch(&touch(12, Started, 200.0, 100.0), false);
        let moved = tracker.handle_touch(&touch(11, Moved, 120.0, 100.0), false);
        assert!(matches!(
            moved.action,
            Some(TouchGestureAction::TwoFinger {
                pan_dx: 10.0,
                pan_dy: 0.0,
                ..
            })
        ));
    }

    #[test]
    fn gui_fingers_and_orphan_events_cannot_join_camera_gestures() {
        use winit::event::TouchPhase::*;
        let mut tracker = TouchTracker::default();
        assert!(
            tracker
                .handle_touch(&touch(1, Started, 1000.0, 30.0), true)
                .suppressed
        );
        for phase in [Moved, Moved, Ended, Cancelled] {
            let response = tracker.handle_touch(&touch(1, phase, 100.0, 100.0), false);
            assert!(response.suppressed);
            assert!(response.action.is_none());
            assert!(tracker.touches.is_empty());
        }
        tracker.handle_touch(&touch(2, Started, 100.0, 100.0), false);
        for phase in [Moved, Ended, Cancelled] {
            let response = tracker.handle_touch(&touch(1, phase, 300.0, 100.0), false);
            assert!(response.suppressed);
            assert!(response.action.is_none());
            assert_eq!(tracker.touches.len(), 1);
        }
        let response = tracker.handle_touch(&touch(2, Moved, 110.0, 100.0), false);
        assert!(matches!(
            response.action,
            Some(TouchGestureAction::OneDrag { dx: 10.0, .. })
        ));
    }

    #[test]
    fn gui_capture_cancels_camera_fingers_until_a_fresh_start() {
        use winit::event::TouchPhase::*;
        let mut tracker = TouchTracker::default();
        tracker.handle_touch(&touch(1, Started, 100.0, 100.0), false);
        tracker.handle_touch(&touch(1, Moved, 110.0, 100.0), false);
        let captured = tracker.handle_touch(&touch(2, Started, 1000.0, 30.0), true);
        assert!(matches!(
            captured.action,
            Some(TouchGestureAction::OneDragEnd)
        ));
        let moved = tracker.handle_touch(&touch(1, Moved, 120.0, 100.0), false);
        assert!(moved.suppressed);
        assert!(moved.action.is_none());
        tracker.handle_touch(&touch(1, Ended, 120.0, 100.0), false);
        tracker.handle_touch(&touch(2, Ended, 1000.0, 30.0), false);
        tracker.handle_touch(&touch(3, Started, 100.0, 100.0), false);
        let moved = tracker.handle_touch(&touch(3, Moved, 130.0, 100.0), false);
        assert!(!moved.suppressed);
        assert!(matches!(
            moved.action,
            Some(TouchGestureAction::OneDrag { dx: 30.0, .. })
        ));
    }

    #[cfg(feature = "window-gui")]
    #[test]
    fn scene_touch_keeps_ownership_while_egui_still_has_old_gui_hover() {
        use winit::event::TouchPhase::*;
        let ctx = egui::Context::default();
        let output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280.0, 720.0),
                )),
                events: vec![egui::Event::PointerMoved(egui::pos2(1050.0, 28.0))],
                ..Default::default()
            },
            |ui| {
                egui::Panel::right("controls")
                    .exact_size(300.0)
                    .show(ui, |ui| {
                        let _ = ui.button("Control");
                    });
            },
        );
        output.drop_without_applying_deltas();
        assert!(ctx.egui_wants_pointer_input());
        let scene_pos = egui::pos2(520.0, 300.0);
        let mut tracker = TouchTracker::default();
        let start = tracker.handle_touch(
            &touch(1, Started, 520.0, 300.0),
            crate::egui_integration::position_over_gui(&ctx, scene_pos),
        );
        assert!(!start.suppressed);
        // No GUI frame between Started and Moved: the global query still
        // describes the old hover, but cannot steal an accepted scene finger.
        let moved = tracker.handle_touch(
            &touch(1, Moved, 560.0, 300.0),
            ctx.egui_wants_pointer_input(),
        );
        assert!(!moved.suppressed);
        assert!(matches!(
            moved.action,
            Some(TouchGestureAction::OneDrag { dx: 40.0, dy: 0.0 })
        ));
        let ended = tracker.handle_touch(
            &touch(1, Ended, 560.0, 300.0),
            ctx.egui_wants_pointer_input(),
        );
        assert!(matches!(ended.action, Some(TouchGestureAction::OneDragEnd)));
        assert!(tracker.touches.is_empty());
    }

    #[test]
    fn touch_cancellation_resets_pinch_baselines_and_ends_drag_once() {
        use winit::event::TouchPhase::*;
        for captured in [false, true] {
            let mut tracker = TouchTracker::default();
            tracker.handle_touch(&touch(1, Started, 100.0, 100.0), false);
            let ended = tracker.handle_touch(&touch(1, Cancelled, 100.0, 100.0), captured);
            assert!(matches!(ended.action, Some(TouchGestureAction::OneDragEnd)));
            assert!(tracker.cancel().is_none());

            tracker.handle_touch(&touch(2, Started, 100.0, 100.0), false);
            tracker.handle_touch(&touch(3, Started, 200.0, 100.0), false);
            tracker.handle_touch(&touch(3, Moved, 220.0, 100.0), false);
            // The same reset is used on focus loss and application suspension.
            assert!(matches!(
                tracker.cancel(),
                Some(TouchGestureAction::OneDragEnd)
            ));
            assert!(tracker.touches.is_empty());
            assert!(tracker.prev_pinch_distance.is_none());
            assert!(tracker.prev_midpoint.is_none());
            assert!(
                tracker
                    .handle_touch(&touch(2, Moved, 200.0, 200.0), false)
                    .action
                    .is_none()
            );

            tracker.handle_touch(&touch(4, Started, 0.0, 0.0), false);
            tracker.handle_touch(&touch(5, Started, 100.0, 0.0), false);
            let moved = tracker.handle_touch(&touch(5, Moved, 120.0, 0.0), false);
            match moved.action {
                Some(TouchGestureAction::TwoFinger {
                    scroll_delta,
                    pan_dx,
                    pan_dy,
                    ..
                }) => {
                    assert!((scroll_delta - 1.0).abs() < 1e-5);
                    assert_eq!((pan_dx, pan_dy), (10.0, 0.0));
                }
                _ => panic!("fresh two-finger gesture must use its own baseline"),
            }
        }
    }

    #[test]
    fn touch_finger_count_transitions_do_not_jump() {
        use winit::event::TouchPhase::*;
        let mut tracker = TouchTracker::default();
        tracker.handle_touch(&touch(1, Started, 100.0, 100.0), false);
        tracker.handle_touch(&touch(2, Started, 200.0, 100.0), false);
        tracker.handle_touch(&touch(3, Started, 300.0, 100.0), false);
        assert!(
            tracker
                .handle_touch(&touch(3, Moved, 400.0, 100.0), false)
                .action
                .is_none()
        );
        tracker.handle_touch(&touch(3, Ended, 400.0, 100.0), false);
        assert!(matches!(
            tracker
                .handle_touch(&touch(2, Moved, 210.0, 100.0), false)
                .action,
            Some(TouchGestureAction::TwoFinger {
                scroll_delta: 0.0,
                pan_dx: 0.0,
                pan_dy: 0.0,
                ..
            })
        ));
        tracker.handle_touch(&touch(2, Ended, 210.0, 100.0), false);
        assert!(matches!(
            tracker
                .handle_touch(&touch(1, Moved, 110.0, 100.0), false)
                .action,
            Some(TouchGestureAction::OneDrag { dx: 10.0, dy: 0.0 })
        ));
    }

    #[test]
    fn pointer_reset_notifies_custom_touch_end_without_duplicate_mouse_end() {
        #[derive(Default)]
        struct EndCamera {
            touch_ends: usize,
            drag_ends: usize,
        }
        impl crate::Camera for EndCamera {
            fn view_matrix(&self) -> glam::Mat4 {
                glam::Mat4::IDENTITY
            }
            fn projection_matrix(&self, _: f32) -> glam::Mat4 {
                glam::Mat4::IDENTITY
            }
            fn position(&self) -> glam::Vec3 {
                glam::Vec3::ZERO
            }
        }
        impl InteractiveCamera for EndCamera {
            fn on_mouse_drag(&mut self, _: f64, _: f64, _: bool, _: bool, _: bool) {}
            fn on_scroll(&mut self, _: f32) {}
            fn update(&mut self, _: f32) {}
            fn set_enabled(&mut self, _: bool) {}
            fn is_enabled(&self) -> bool {
                true
            }
            fn on_drag_end(&mut self) {
                self.drag_ends += 1;
            }
            fn on_touch_drag_end(&mut self) {
                self.touch_ends += 1;
                self.on_drag_end();
            }
        }
        let mut camera = EndCamera::default();
        let mut input = InputState::default();
        let mut tracker = TouchTracker::default();
        tracker.handle_touch(
            &touch(1, winit::event::TouchPhase::Started, 100.0, 100.0),
            false,
        );
        cancel_pointer_input(&mut input, &mut tracker, &mut camera);
        assert_eq!((camera.touch_ends, camera.drag_ends), (1, 1));
        assert!(tracker.touches.is_empty());
        cancel_pointer_input(&mut input, &mut tracker, &mut camera);
        assert_eq!((camera.touch_ends, camera.drag_ends), (1, 2));
        input.mouse_buttons_down.left = true;
        cancel_pointer_input(&mut input, &mut tracker, &mut camera);
        assert_eq!((camera.touch_ends, camera.drag_ends), (1, 3));
        assert!(!input.mouse_buttons_down.left);
    }

    #[test]
    fn capture_keeps_window_invalidations_until_an_ordinary_frame() {
        let now = Instant::now();
        let original = (true, Some(now + Duration::from_secs(2)));
        let captured = frame_requests(true, original, (false, None));
        assert_eq!(captured, original);
        let earlier = now + Duration::from_secs(1);
        let captured_again = frame_requests(true, captured, (false, Some(earlier)));
        assert_eq!(captured_again, (true, Some(earlier)));
        assert_eq!(
            frame_requests(false, captured_again, (false, None)),
            (false, None)
        );
        assert_eq!(
            frame_requests(true, (false, None), (false, None)),
            (false, None)
        );
    }

    #[test]
    fn hidden_captures_freeze_camera_animation_and_exclude_hidden_time() {
        let mut camera = crate::Camera2d::default();
        camera.zoom_smoothing = 0.2;
        camera.on_scroll(1.0);
        let start = Instant::now();
        let mut clock = CameraClock::new(start);
        let initial = camera.zoom;
        for seconds in [1, 10, 60] {
            camera.update(
                clock
                    .advance_frame(start + Duration::from_secs(seconds), true)
                    .as_secs_f32(),
            );
            assert_eq!(camera.zoom, initial);
        }
        camera.update(
            clock
                .advance_frame(
                    start + Duration::from_secs(60) + Duration::from_millis(16),
                    false,
                )
                .as_secs_f32(),
        );
        assert!(camera.zoom > initial && camera.zoom < initial + 0.03);
    }

    #[test]
    fn reactive_scroll_after_idle_animates_then_sleeps() {
        let mut camera = crate::Camera2d::default();
        camera.zoom_smoothing = 0.2;
        let start = Instant::now();
        let mut clock = CameraClock::new(start);
        assert!(!clock.finish_frame(RedrawPolicy::Reactive, camera.needs_redraw(), Duration::MAX));

        let input_time = start + Duration::from_secs(60);
        clock.before_change(input_time, camera.needs_redraw());
        camera.on_scroll(1.0);
        let frame_dt = Duration::from_secs_f32(1.0 / 60.0);
        let first_frame = input_time + frame_dt;
        camera.update(clock.advance(first_frame).as_secs_f32());
        assert!(
            (camera.zoom - 1.02).abs() < 1e-5,
            "idle time must not advance the new zoom"
        );
        assert!(clock.finish_frame(RedrawPolicy::Reactive, camera.needs_redraw(), Duration::MAX));

        let mut settled = false;
        for frame in 2..300 {
            camera.update(clock.advance(input_time + frame_dt * frame).as_secs_f32());
            if !clock.finish_frame(RedrawPolicy::Reactive, camera.needs_redraw(), Duration::MAX) {
                settled = true;
                break;
            }
        }
        assert!(
            settled,
            "one-shot input must settle without further input or GUI repaint"
        );
        assert!((camera.zoom - 1.1).abs() < 1e-6);
        assert_eq!(
            clock.advance(input_time + Duration::from_secs(120)),
            Duration::ZERO
        );
    }

    #[test]
    fn held_drag_does_not_add_idle_time_to_release_inertia() {
        let start = Instant::now();
        let mut clock = CameraClock::new(start);
        let mut camera = crate::Camera2d::default();
        camera.damping = 0.95;
        clock.before_change(start, camera.needs_redraw());
        camera.on_mouse_drag(20.0, 0.0, true, false, false);
        camera.update(clock.advance(start).as_secs_f32());
        assert!(!clock.finish_frame(RedrawPolicy::Reactive, camera.needs_redraw(), Duration::MAX));
        let position = camera.position;

        let release = start + Duration::from_secs(60);
        clock.before_change(release, camera.needs_redraw());
        camera.on_drag_end();
        camera.update(clock.advance(release).as_secs_f32());
        assert_eq!(camera.position, position);
        assert!(clock.finish_frame(RedrawPolicy::Reactive, camera.needs_redraw(), Duration::MAX));
        camera.update(
            clock
                .advance(release + Duration::from_millis(16))
                .as_secs_f32(),
        );
        assert_ne!(camera.position, position);
    }

    #[test]
    fn active_camera_time_survives_unrelated_input_and_is_bounded() {
        let start = Instant::now();
        let mut clock = CameraClock::new(start);
        clock.before_change(start + Duration::from_millis(10), true);
        assert_eq!(
            clock.advance(start + Duration::from_millis(16)),
            Duration::from_millis(16)
        );
        assert_eq!(
            clock.advance(start + Duration::from_secs(60)),
            Duration::from_millis(250)
        );
    }

    #[test]
    fn continuous_and_gui_redraws_preserve_camera_idle_policy() {
        let start = Instant::now();
        let mut clock = CameraClock::new(start);
        assert!(clock.finish_frame(RedrawPolicy::Continuous, false, Duration::MAX));
        assert_eq!(
            clock.advance(start + Duration::from_millis(16)),
            Duration::from_millis(16)
        );
        assert!(clock.finish_frame(RedrawPolicy::Reactive, false, Duration::ZERO));
        assert_eq!(
            clock.advance(start + Duration::from_secs(60)),
            Duration::ZERO
        );
        assert!(!clock.finish_frame(RedrawPolicy::Reactive, false, Duration::from_secs(1)));
    }

    #[test]
    fn app_redraw_requests_coalesce_and_wake_again_after_draining() {
        use std::sync::atomic::AtomicUsize;
        let wakes = Arc::new(AtomicUsize::new(0));
        let counter = wakes.clone();
        let request = RedrawRequest::new(Arc::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
        }));
        let worker = request.clone();
        std::thread::spawn(move || {
            worker.request_redraw();
            worker.request_redraw();
        })
        .join()
        .unwrap();
        assert_eq!(wakes.load(Ordering::SeqCst), 1);
        assert!(request.take_pending());
        assert!(!request.take_pending());
        // A request from a render hook after the pump drained pending work
        // must remain available to schedule a subsequent frame.
        request.request_redraw();
        assert_eq!(wakes.load(Ordering::SeqCst), 2);
        assert!(request.take_pending());
    }

    #[test]
    fn default_values() {
        let c = RunConfig::default();
        assert_eq!(c.width, 1280);
        assert_eq!(c.height, 720);
        assert_eq!(c.simulation_policy, SimulationPolicy::PerRedraw);
        assert_eq!(c.presentation_policy, PresentationPolicy::Immediate);
        assert_eq!(c.max_in_flight_submissions, 8);
    }
    #[test]
    fn camera_builder_preserves_policies() {
        let policy = SimulationPolicy::fixed(Duration::from_millis(10));
        let c = RunConfig::new("test")
            .with_size(10, 20)
            .with_simulation_policy(policy)
            .with_presentation_policy(PresentationPolicy::Vsync)
            .with_camera(());
        assert_eq!(c.title, "test");
        assert_eq!(c.width, 10);
        assert_eq!(c.simulation_policy, policy);
        assert_eq!(c.presentation_policy, PresentationPolicy::Vsync);
    }

    #[test]
    fn vsync_rejects_tearing_or_mailbox_modes() {
        for mode in [
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::AutoNoVsync,
            wgpu::PresentMode::Mailbox,
        ] {
            let config = RunConfig::default()
                .with_presentation_policy(PresentationPolicy::Vsync)
                .with_present_mode(mode);
            assert!(matches!(
                validate_run_config(&config),
                Err(RunError::InvalidConfig(_))
            ));
        }
        for mode in [
            wgpu::PresentMode::AutoVsync,
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::FifoRelaxed,
        ] {
            assert!(
                validate_run_config(
                    &RunConfig::default()
                        .with_presentation_policy(PresentationPolicy::Vsync)
                        .with_present_mode(mode)
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn invalid_config_returns_before_creating_an_event_loop() {
        struct UnusedApp;
        impl App for UnusedApp {
            type Camera = ();
            fn render(&mut self, _: &mut crate::RenderContext<Self::Camera>) {}
        }

        let mut config = RunConfig::new("invalid").with_camera(());
        config.max_in_flight_submissions = 1;
        let result = run(
            |_, _, _| -> UnusedApp { panic!("factory must not run") },
            config,
        );
        assert!(matches!(result, Err(RunError::InvalidConfig(_))));

        let config = RunConfig::new("invalid")
            .with_camera(())
            .with_simulation_policy(SimulationPolicy::Fixed {
                dt: Duration::ZERO,
                max_ticks_per_wake: 8,
                max_backlog: Duration::from_millis(250),
            });
        let result = run(
            |_, _, _| -> UnusedApp { panic!("factory must not run") },
            config,
        );
        assert!(matches!(result, Err(RunError::InvalidConfig(_))));
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn terminal_failure_survives_event_loop_exit() {
        for loop_result in [Ok(()), Err(winit::error::EventLoopError::ExitFailure(9))] {
            let result = finish_run(
                Some(RunError::Runtime(RuntimeError::DeviceLost("lost".into()))),
                loop_result,
            );
            assert!(matches!(
                result,
                Err(RunError::Runtime(RuntimeError::DeviceLost(_)))
            ));
        }
        assert!(finish_run(None, Ok(())).is_ok());
        assert!(matches!(
            finish_run(None, Err(winit::error::EventLoopError::ExitFailure(9))),
            Err(RunError::EventLoop(
                winit::error::EventLoopError::ExitFailure(9)
            ))
        ));
    }
}
