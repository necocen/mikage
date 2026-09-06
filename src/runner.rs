//! winit host for the portable [`AppRuntime`].

use crate::egui_integration::EguiIntegration;
use crate::input::InputState;
use crate::{
    App, AppRuntime, GpuContext, GpuDescriptor, InteractiveCamera, OrbitCamera, RenderTarget,
    RenderTargetConfig, RuntimeError, SurfaceContext, SurfaceDescriptor, WindowInputContext,
};
use dpi::PhysicalSize;
use std::sync::Arc;
use std::time::Duration;
#[cfg(not(target_family = "wasm"))]
use std::time::Instant;
#[cfg(target_family = "wasm")]
use web_time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowAttributes, WindowId};

#[cfg(all(feature = "agent", not(target_family = "wasm")))]
#[path = "runner/agent_driver.rs"]
mod agent_driver;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RedrawPolicy {
    #[default]
    Continuous,
    Reactive,
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
    pub width: u32,
    pub height: u32,
    pub camera: C,
    pub gpu: GpuDescriptor,
    pub present_mode: wgpu::PresentMode,
    pub sample_count: u32,
    pub init_logging: bool,
    pub canvas: Option<String>,
    pub redraw_policy: RedrawPolicy,
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
    simulation_policy: SimulationPolicy,
    pointer_suppressed: bool,
    input_dirty: bool,
    occluded: bool,
    suspended: bool,
    redraw_pending: bool,
    last_render: Instant,
    last_schedule: Instant,
    backlog: Duration,
    discarded_wall_time: Duration,
    gui_deadline: Option<Instant>,
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    diagnostics: agent_driver::Diagnostics,
}
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
        let app = self.init_fn.take().expect("factory called once")(
            &gpu,
            surface.render_target_config(),
            surface.size(),
        );
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
        runtime.set_waker(Arc::new(move || {
            let _ = proxy.send_event(());
        }));
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
            simulation_policy: config.simulation_policy,
            pointer_suppressed: false,
            input_dirty: false,
            occluded: false,
            suspended: false,
            redraw_pending: true,
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
        let now = Instant::now();
        if let Some(deadline) = state.egui.take_repaint_request() {
            state.gui_deadline = Some(
                state
                    .gui_deadline
                    .map_or(deadline, |existing| existing.min(deadline)),
            );
        }
        let delta = now.duration_since(state.last_schedule);
        state.last_schedule = now;
        let active = !state
            .surface
            .as_ref()
            .expect("active surface")
            .is_suspended()
            && !state.occluded
            && !state.suspended;
        #[cfg(all(feature = "agent", not(target_family = "wasm")))]
        let automatic = state
            .diagnostics
            .permits_automatic_tick(state.runtime.progress().submitted_ticks);
        #[cfg(not(all(feature = "agent", not(target_family = "wasm"))))]
        let automatic = true;
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
                    if state.backlog < dt || state.runtime.available_submission_slots() <= 1 {
                        break;
                    }
                    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
                    if !state
                        .diagnostics
                        .permits_automatic_tick(state.runtime.progress().submitted_ticks)
                    {
                        break;
                    }
                    match state.runtime.try_tick(dt) {
                        Ok(_) => {
                            state.backlog -= dt;
                            state.redraw_pending = true;
                        }
                        Err(RuntimeError::WouldBlock) => break,
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
        if active && state.gui_deadline.is_some_and(|deadline| deadline <= now) {
            state.redraw_pending = true;
            state.gui_deadline = None;
        }
        if active && state.redraw_pending && state.runtime.available_submission_slots() > 0 {
            state.window.request_redraw();
        }
        let mut deadline = if active { state.gui_deadline } else { None };
        if active
            && automatic
            && let SimulationPolicy::Fixed { dt, .. } = state.simulation_policy
        {
            // Completion notifications resume a full queue; do not busy-poll a past deadline.
            if state.runtime.available_submission_slots() > 1 {
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
            state.last_schedule = Instant::now();
            state.last_render = Instant::now();
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
            attrs = attrs.with_inner_size(PhysicalSize::new(config.width, config.height));
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
                    .with_inner_size(PhysicalSize::new(config.width, config.height))
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
            state.suspended = true;
            state.backlog = Duration::ZERO;
        }
    }
    fn exiting(&mut self, _: &ActiveEventLoop) {
        self.shutdown();
    }
    fn user_event(&mut self, event_loop: &ActiveEventLoop, _: ()) {
        if self.first_error.is_some() || event_loop.exiting() {
            return;
        }
        #[cfg(target_family = "wasm")]
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

        let pointer_suppress = is_pointer_event && state.egui.wants_pointer_input();
        let suppress = (is_keyboard_event && egui_consumed) || pointer_suppress;

        // When egui captures a category, clear stuck state in InputState.
        if is_keyboard_event && egui_consumed {
            state.input.clear_keyboard();
        }
        if pointer_suppress {
            // End any active drag when egui starts capturing pointer.
            if !state.pointer_suppressed {
                state.runtime.camera.on_drag_end();
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
                // Touch gestures: runner tracks state, camera interprets
                WindowEvent::Touch(touch) => {
                    if let Some(action) = state.touch_tracker.handle_touch(touch) {
                        match action {
                            TouchGestureAction::OneDrag { dx, dy } => {
                                state.runtime.camera.on_touch_drag(dx, dy);
                            }
                            TouchGestureAction::OneDragEnd => {
                                state.runtime.camera.on_touch_drag_end();
                            }
                            TouchGestureAction::TwoFinger {
                                scroll_delta,
                                pan_dx,
                                pan_dy,
                                midpoint,
                            } => {
                                state.runtime.camera.on_pinch_pan(
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
                state.input.clear_pointer();
                state.input_dirty = true;
                state.runtime.camera.on_drag_end();
            }
            WindowEvent::Focused(true) => {
                state.occluded = false;
                state.redraw_pending = true;
                state.last_schedule = Instant::now();
                state.last_render = Instant::now();
            }
            WindowEvent::Occluded(value) => {
                state.occluded = value;
                state.backlog = Duration::ZERO;
                state.last_render = Instant::now();
                state.last_schedule = Instant::now();
                if !value {
                    state.redraw_pending = true;
                }
            }
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    state.occluded = false;
                }
                state.surface_descriptor.size = size;
                if let Err(err) = state
                    .surface
                    .as_mut()
                    .expect("active surface")
                    .resize(&state.runtime.gpu, size)
                {
                    self.fail(event_loop, err.into());
                    return;
                }
                state
                    .runtime
                    .camera
                    .set_viewport_size(size.width, size.height);
                state.runtime.app.resize(&state.runtime.gpu, size);
                state.backlog = Duration::ZERO;
                state.last_render = Instant::now();
                state.last_schedule = Instant::now();
                state.redraw_pending = true;
            }
            WindowEvent::RedrawRequested => {
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
    if state
        .surface
        .as_ref()
        .expect("active surface")
        .is_suspended()
        || state.suspended
        || state.occluded
    {
        return Ok(());
    }
    if state.runtime.available_submission_slots() == 0 {
        state.redraw_pending = true;
        return Ok(());
    }
    let (frame, suboptimal) = match state
        .surface
        .as_ref()
        .expect("active surface")
        .acquire_surface_texture()
    {
        Some(wgpu::CurrentSurfaceTexture::Success(frame)) => (frame, false),
        Some(wgpu::CurrentSurfaceTexture::Suboptimal(frame)) => (frame, true),
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
            state.occluded = true;
            state.redraw_pending = false;
            return Ok(());
        }
        Some(wgpu::CurrentSurfaceTexture::Timeout) | None => {
            state.redraw_pending = false;
            state.gui_deadline = Some(Instant::now() + Duration::from_millis(16));
            return Ok(());
        }
    };
    let now = Instant::now();
    let dt = now
        .duration_since(state.last_render)
        .min(Duration::from_millis(250));
    state.last_render = now;
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    let capture_only = state.diagnostics.requires_frame(&state.runtime)
        || !state
            .diagnostics
            .permits_automatic_tick(state.runtime.progress().submitted_ticks);
    #[cfg(not(all(feature = "agent", not(target_family = "wasm"))))]
    let capture_only = false;
    if state.simulation_policy == SimulationPolicy::PerRedraw
        && !capture_only
        && state.runtime.available_submission_slots() > 1
    {
        state.runtime.try_tick(dt)?;
    }
    state.runtime.camera.update(dt.as_secs_f32());
    let prepared = state.egui.build(&state.window, &mut state.runtime.app);
    let config = state
        .surface
        .as_ref()
        .expect("active surface")
        .render_target_config();
    let size = state.surface.as_ref().expect("active surface").size();
    let surface_view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
        format: Some(config.color_format),
        ..Default::default()
    });
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    let capture_texture = if state.diagnostics.requires_frame(&state.runtime) {
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
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::TEXTURE_BINDING,
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
    let target_view = capture_view.as_ref().unwrap_or(&surface_view);
    #[cfg(not(all(feature = "agent", not(target_family = "wasm"))))]
    let target_view = &surface_view;
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
            crate::blit::blit(
                ctx.gpu,
                ctx.encoder,
                target_view,
                &surface_view,
                config.color_format,
            );
        }
    })?;
    state.egui.finish(prepared);
    state.runtime.gpu.queue.present(frame);
    state.runtime.mark_presented(&token);
    #[cfg(all(feature = "agent", not(target_family = "wasm")))]
    state.diagnostics.after_frame_submit(&token);
    if suboptimal {
        state
            .surface
            .as_mut()
            .expect("active surface")
            .reconfigure(&state.runtime.gpu)?;
    }
    let repaint = state.egui.repaint_after();
    state.gui_deadline = if repaint == Duration::MAX {
        None
    } else {
        Instant::now().checked_add(repaint)
    };
    state.redraw_pending = state.redraw_policy == RedrawPolicy::Continuous || repaint.is_zero();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_values() {
        let c = RunConfig::default();
        assert_eq!(c.width, 1280);
        assert_eq!(c.height, 720);
        assert_eq!(c.simulation_policy, SimulationPolicy::PerRedraw);
        assert_eq!(c.max_in_flight_submissions, 8);
    }
    #[test]
    fn camera_builder_preserves_policies() {
        let policy = SimulationPolicy::fixed(Duration::from_millis(10));
        let c = RunConfig::new("test")
            .with_size(10, 20)
            .with_simulation_policy(policy)
            .with_camera(());
        assert_eq!(c.title, "test");
        assert_eq!(c.width, 10);
        assert_eq!(c.simulation_policy, policy);
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
