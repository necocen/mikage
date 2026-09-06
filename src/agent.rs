//! Local HTTP control plane for LLM/debugging agents.
//!
//! This module is native-only and enabled with the `agent` feature.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use image::ImageEncoder;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::camera::InteractiveCamera;

/// Configuration for the local HTTP agent API.
#[derive(Clone, Debug)]
pub struct AgentConfig {
    /// Address to bind. Defaults to `127.0.0.1:3939`.
    pub bind_addr: SocketAddr,
    /// Optional bearer token. If set, clients must send either
    /// `Authorization: Bearer <token>` or `X-Mikage-Token: <token>`.
    pub auth_token: Option<String>,
    /// Optional JSON file written after bind with the active address and token hint.
    pub write_connection_file: Option<PathBuf>,
    /// How long HTTP handlers wait for the render thread to answer.
    pub request_timeout: Duration,
    /// Maximum retained pending or completed jobs.
    pub max_jobs: usize,
    /// Maximum total bytes retained as completed results.
    pub max_result_bytes: usize,
    /// Jobs expire after this interval, including pending jobs.
    pub job_ttl: Duration,
    /// Maximum concurrent HTTP connections.
    pub max_connections: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:3939"
                .parse()
                .expect("default agent bind address must parse"),
            auth_token: None,
            write_connection_file: None,
            request_timeout: Duration::from_secs(10),
            max_jobs: 64,
            max_result_bytes: 64 * 1024 * 1024,
            job_ttl: Duration::from_secs(60),
            max_connections: 32,
        }
    }
}

impl AgentConfig {
    /// Uses a different bind address.
    pub fn with_bind_addr(mut self, bind_addr: SocketAddr) -> Self {
        self.bind_addr = bind_addr;
        self
    }

    /// Enables bearer-token authentication.
    pub fn with_auth_token(mut self, token: impl Into<String>) -> Self {
        self.auth_token = Some(token.into());
        self
    }

    /// Writes connection details to a JSON file when the server starts.
    pub fn with_connection_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.write_connection_file = Some(path.into());
        self
    }
}

/// Mouse button used by camera drag commands.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentMouseButton {
    Left,
    Right,
    Middle,
}

impl AgentMouseButton {
    pub fn flags(self) -> (bool, bool, bool) {
        match self {
            Self::Left => (true, false, false),
            Self::Right => (false, true, false),
            Self::Middle => (false, false, true),
        }
    }
}

/// Commands accepted by `POST /command`.
#[derive(Debug, Deserialize)]
#[serde(tag = "op")]
pub enum AgentCommand {
    /// Applies a synthetic camera drag in physical pixels.
    #[serde(rename = "camera.drag")]
    CameraDrag {
        dx: f64,
        dy: f64,
        #[serde(default = "default_drag_button")]
        button: AgentMouseButton,
    },
    /// Applies a camera scroll/zoom delta.
    #[serde(rename = "camera.zoom")]
    CameraZoom { delta: f32 },
    /// Enables or disables framework camera input.
    #[serde(rename = "camera.set_enabled")]
    CameraSetEnabled { enabled: bool },
    /// Sets fields on the built-in orbit camera.
    #[serde(rename = "camera.set_orbit")]
    CameraSetOrbit {
        target: Option<[f32; 3]>,
        distance: Option<f32>,
        yaw: Option<f32>,
        pitch: Option<f32>,
        fov_y: Option<f32>,
    },
    /// Sets fields on the built-in 2D camera.
    #[serde(rename = "camera.set_2d")]
    CameraSet2d {
        position: Option<[f32; 2]>,
        zoom: Option<f32>,
    },
    /// Calls [`App::on_agent_command`](crate::App::on_agent_command).
    #[serde(rename = "app.command")]
    AppCommand { payload: Value },
    /// Records app GPU work and completes after its submission finishes.
    #[serde(rename = "app.gpu_command")]
    AppGpuCommand {
        payload: Value,
        #[serde(default)]
        at_tick: Option<u64>,
    },
    /// Advances fixed simulation steps until this absolute tick is completed.
    #[serde(rename = "run_until_completed")]
    RunUntilCompleted { target_tick: u64, dt: f32 },
    /// Pauses automatic simulation ticks.
    #[serde(rename = "runtime.pause")]
    Pause,
    /// Resumes the configured automatic simulation policy.
    #[serde(rename = "runtime.resume")]
    Resume,
    /// Requests a redraw and returns immediately.
    #[serde(rename = "redraw")]
    Redraw,
    /// Exits the application event loop after returning a response.
    #[serde(rename = "shutdown")]
    Shutdown,
}

impl AgentCommand {
    pub fn is_camera_command(&self) -> bool {
        matches!(
            self,
            Self::CameraDrag { .. }
                | Self::CameraZoom { .. }
                | Self::CameraSetEnabled { .. }
                | Self::CameraSetOrbit { .. }
                | Self::CameraSet2d { .. }
        )
    }
}

fn default_drag_button() -> AgentMouseButton {
    AgentMouseButton::Left
}

/// Camera state reported by `GET /status`.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CameraSnapshot {
    Generic {
        position: [f32; 3],
        enabled: bool,
    },
    Orbit {
        position: [f32; 3],
        target: [f32; 3],
        distance: f32,
        yaw: f32,
        pitch: f32,
        fov_y: f32,
        enabled: bool,
    },
    Camera2d {
        position: [f32; 2],
        zoom: f32,
        enabled: bool,
    },
}

/// Shared state exposed by `GET /status`.
#[derive(Clone, Debug, Serialize)]
pub struct AgentSnapshot {
    pub ready: bool,
    pub bind_addr: String,
    pub window_size: Option<[u32; 2]>,
    pub frame_count: u64,
    pub elapsed: f64,
    pub screenshot_supported: bool,
    pub camera: Option<CameraSnapshot>,
    pub app: Value,
    pub progress: crate::RuntimeProgress,
    pub gpu: Value,
    pub capture_targets: Vec<String>,
    pub discarded_wall_time: f64,
}

impl AgentSnapshot {
    fn new(bind_addr: SocketAddr) -> Self {
        Self {
            ready: false,
            bind_addr: bind_addr.to_string(),
            window_size: None,
            frame_count: 0,
            elapsed: 0.0,
            screenshot_supported: false,
            camera: None,
            app: Value::Null,
            progress: crate::RuntimeProgress::default(),
            gpu: Value::Null,
            capture_targets: Vec::new(),
            discarded_wall_time: 0.0,
        }
    }
}

/// Structured hardware capabilities and the actual enabled device contract.
pub fn gpu_snapshot(gpu: &crate::GpuContext) -> Value {
    let capabilities = gpu.capabilities();
    json!({
        "adapter": capabilities.adapter_info,
        "supported_features": capabilities.supported_features,
        "supported_limits": capabilities.supported_limits,
        "enabled_features": capabilities.enabled_features,
        "enabled_limits": capabilities.enabled_limits,
        "timestamp_period": capabilities.timestamp_period,
    })
}

/// HTTP job identifier, unique for one server instance.
pub type JobId = u64;

/// Capture result encoding. Raw preserves the GPU format and channel order.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CaptureFormat {
    #[default]
    Png,
    Raw,
}

/// A named resource capture at an optional simulation checkpoint.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CaptureRequest {
    pub target: String,
    pub format: CaptureFormat,
    /// Prevent simulation advancement until this checkpoint is captured.
    pub exact: bool,
    pub at_tick: Option<u64>,
}

impl Default for CaptureRequest {
    fn default() -> Self {
        Self {
            target: "window".into(),
            format: CaptureFormat::Png,
            exact: false,
            at_tick: None,
        }
    }
}

pub struct AgentBridge {
    requests: mpsc::Receiver<AgentRequest>,
    snapshot: Arc<Mutex<AgentSnapshot>>,
    jobs: Arc<JobStore>,
    stopped: Arc<AtomicBool>,
    server_thread: Option<thread::JoinHandle<()>>,
}

impl AgentBridge {
    pub fn start(
        config: AgentConfig,
        wake: impl Fn() + Send + Sync + 'static,
    ) -> std::io::Result<Self> {
        if config.max_jobs == 0
            || config.max_connections == 0
            || config.max_result_bytes == 0
            || config.job_ttl.is_zero()
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "agent capacity and job TTL must be positive",
            ));
        }
        let listener = TcpListener::bind(config.bind_addr)?;
        listener.set_nonblocking(true)?;
        let bind_addr = listener.local_addr()?;
        let snapshot = Arc::new(Mutex::new(AgentSnapshot::new(bind_addr)));
        let jobs = Arc::new(JobStore::new(&config));
        let stopped = Arc::new(AtomicBool::new(false));
        let (request_tx, requests) = mpsc::sync_channel(config.max_jobs);
        if let Some(path) = &config.write_connection_file {
            let connection = json!({ "addr": bind_addr.to_string(), "auth": config.auth_token.as_ref().map(|_| "bearer") });
            std::fs::write(path, serde_json::to_vec_pretty(&connection)?)?;
        }
        let server = Server {
            request_tx,
            wake: Arc::new(wake),
            snapshot: snapshot.clone(),
            jobs: jobs.clone(),
            stopped: stopped.clone(),
            config,
        };
        let server_thread = thread::Builder::new()
            .name("mikage-agent-http".into())
            .spawn(move || serve_http(listener, server))?;
        Ok(Self {
            requests,
            snapshot,
            jobs,
            stopped,
            server_thread: Some(server_thread),
        })
    }

    pub fn drain_requests(&mut self) -> Vec<AgentRequest> {
        self.requests
            .try_iter()
            .filter(|request| request.job_id.is_none_or(|id| self.jobs.is_pending(id)))
            .collect()
    }

    pub fn update_snapshot(&self, update: impl FnOnce(&mut AgentSnapshot)) {
        if let Ok(mut snapshot) = self.snapshot.lock() {
            update(&mut snapshot);
        }
    }

    pub fn is_job_pending(&self, id: JobId) -> bool {
        self.jobs.is_pending(id)
    }

    pub fn complete_job(&self, id: JobId, response: AgentResponse) {
        self.jobs.complete(id, response);
    }

    pub fn fail_all(&self, message: &str) {
        self.jobs.fail_all(message);
    }
}

impl Drop for AgentBridge {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Release);
        self.jobs.fail_all("application shut down");
        if let Some(worker) = self.server_thread.take() {
            let _ = worker.join();
        }
    }
}

pub struct AgentRequest {
    pub kind: AgentRequestKind,
    pub respond_to: mpsc::Sender<AgentResponse>,
    pub job_id: Option<JobId>,
}

pub enum AgentRequestKind {
    Command(AgentCommand),
    Capture(CaptureRequest),
    Screenshot,
}

#[derive(Clone)]
pub enum AgentResponse {
    Json(Value),
    Png(Vec<u8>),
    Bytes {
        bytes: Arc<Vec<u8>>,
        content_type: String,
        metadata: Value,
    },
    Error {
        status: u16,
        message: String,
    },
}

impl AgentResponse {
    pub fn ok() -> Self {
        Self::Json(json!({ "ok": true }))
    }
    pub fn json(value: Value) -> Self {
        Self::Json(value)
    }
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::Error {
            status: 400,
            message: message.into(),
        }
    }
    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::Error {
            status: 503,
            message: message.into(),
        }
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Error {
            status: 500,
            message: message.into(),
        }
    }
    pub fn busy(message: impl Into<String>) -> Self {
        Self::Error {
            status: 429,
            message: message.into(),
        }
    }

    fn byte_len(&self) -> usize {
        match self {
            Self::Json(value) => value.to_string().len(),
            Self::Png(bytes) => bytes.len(),
            Self::Bytes { bytes, .. } => bytes.len(),
            Self::Error { .. } => 0,
        }
    }
}

struct Job {
    created: Instant,
    response: Option<AgentResponse>,
}

struct JobState {
    next_id: JobId,
    jobs: HashMap<JobId, Job>,
    result_bytes: usize,
}

struct JobStore {
    state: Mutex<JobState>,
    changed: Condvar,
    max_jobs: usize,
    max_result_bytes: usize,
    ttl: Duration,
}

impl JobStore {
    fn new(config: &AgentConfig) -> Self {
        Self {
            state: Mutex::new(JobState {
                next_id: 1,
                jobs: HashMap::new(),
                result_bytes: 0,
            }),
            changed: Condvar::new(),
            max_jobs: config.max_jobs,
            max_result_bytes: config.max_result_bytes,
            ttl: config.job_ttl,
        }
    }

    fn expire(&self, state: &mut JobState) {
        state.jobs.retain(|_, job| job.created.elapsed() < self.ttl);
        state.result_bytes = state
            .jobs
            .values()
            .filter_map(|job| job.response.as_ref())
            .map(AgentResponse::byte_len)
            .sum();
    }

    fn create(&self) -> Result<JobId, AgentResponse> {
        let mut state = self.state.lock().unwrap();
        self.expire(&mut state);
        if state.jobs.len() >= self.max_jobs {
            return Err(AgentResponse::busy(
                "job capacity exhausted; wait for results to expire",
            ));
        }
        let id = state.next_id;
        state.next_id = id
            .checked_add(1)
            .ok_or_else(|| AgentResponse::unavailable("job identifiers exhausted"))?;
        state.jobs.insert(
            id,
            Job {
                created: Instant::now(),
                response: None,
            },
        );
        Ok(id)
    }

    /// Release a reservation that was never accepted into the application queue.
    /// The HTTP caller does not receive its id, so retaining it would consume
    /// capacity until TTL without leaving a reachable result.
    fn discard(&self, id: JobId) {
        let mut state = self.state.lock().unwrap();
        if let Some(job) = state.jobs.remove(&id) {
            if let Some(response) = job.response {
                state.result_bytes = state.result_bytes.saturating_sub(response.byte_len());
            }
            self.changed.notify_all();
        }
    }

    fn is_pending(&self, id: JobId) -> bool {
        let mut state = self.state.lock().unwrap();
        self.expire(&mut state);
        state
            .jobs
            .get(&id)
            .is_some_and(|job| job.response.is_none())
    }

    fn complete(&self, id: JobId, mut response: AgentResponse) {
        let mut state = self.state.lock().unwrap();
        self.expire(&mut state);
        if state.jobs.get(&id).is_none_or(|job| job.response.is_some()) {
            return;
        }
        let length = response.byte_len();
        if length > self.max_result_bytes.saturating_sub(state.result_bytes) {
            response =
                AgentResponse::busy("completed result exceeds the retained result memory budget");
        }
        state.result_bytes += response.byte_len();
        state.jobs.get_mut(&id).unwrap().response = Some(response);
        self.changed.notify_all();
    }

    fn status(&self, id: JobId) -> Option<Value> {
        let mut state = self.state.lock().unwrap();
        self.expire(&mut state);
        let job = state.jobs.get(&id)?;
        Some(match &job.response {
            None => json!({"id":id,"state":"pending"}),
            Some(AgentResponse::Error { status, message }) => {
                json!({"id":id,"state":"failed","status":status,"error":message})
            }
            Some(response) => {
                let mut value = json!({"id":id,"state":"completed","result_url":format!("/jobs/{id}/result"),"bytes":response.byte_len()});
                if let AgentResponse::Bytes { metadata, .. } = response {
                    value["metadata"] = metadata.clone();
                }
                value
            }
        })
    }

    fn result(&self, id: JobId) -> Result<Option<AgentResponse>, AgentResponse> {
        let mut state = self.state.lock().unwrap();
        self.expire(&mut state);
        state
            .jobs
            .get(&id)
            .map(|job| job.response.clone())
            .ok_or_else(|| AgentResponse::Error {
                status: 404,
                message: "unknown or expired job".into(),
            })
    }

    fn wait(&self, id: JobId, timeout: Duration) -> Result<AgentResponse, AgentResponse> {
        let mut state = self.state.lock().unwrap();
        let started = Instant::now();
        loop {
            self.expire(&mut state);
            let job = state.jobs.get(&id).ok_or_else(|| AgentResponse::Error {
                status: 404,
                message: "unknown or expired job".into(),
            })?;
            if let Some(response) = &job.response {
                return Ok(response.clone());
            }
            let remaining = timeout
                .saturating_sub(started.elapsed())
                .min(self.ttl.saturating_sub(job.created.elapsed()));
            if remaining.is_zero() {
                return Err(AgentResponse::Error {
                    status: 504,
                    message: format!("job {id} is still pending; inspect /jobs/{id}"),
                });
            }
            state = self.changed.wait_timeout(state, remaining).unwrap().0;
        }
    }

    fn fail_all(&self, message: &str) {
        let ids: Vec<_> = self
            .state
            .lock()
            .unwrap()
            .jobs
            .iter()
            .filter(|(_, job)| job.response.is_none())
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            self.complete(id, AgentResponse::unavailable(message));
        }
    }
}

struct PendingCapture {
    request: CaptureRequest,
    respond_to: mpsc::Sender<AgentResponse>,
    device: wgpu::Device,
}

/// Native readback and encoding worker. No mapped bytes or PNG work runs in a GPU callback.
pub struct AgentCaptureWorker {
    ring: crate::readback::ReadbackRing,
    pending: Arc<Mutex<HashMap<crate::readback::ReadbackId, PendingCapture>>>,
    wake: mpsc::SyncSender<()>,
    stopped: Arc<AtomicBool>,
    worker_thread: Option<thread::JoinHandle<()>>,
}

impl AgentCaptureWorker {
    pub fn new() -> Self {
        let ring = crate::readback::ReadbackRing::default();
        let pending = Arc::new(Mutex::new(HashMap::<
            crate::readback::ReadbackId,
            PendingCapture,
        >::new()));
        let stopped = Arc::new(AtomicBool::new(false));
        let (wake, messages) = mpsc::sync_channel(1);
        let worker_ring = ring.clone();
        let worker_pending = pending.clone();
        let worker_stopped = stopped.clone();
        let worker_thread = thread::Builder::new()
            .name("mikage-capture-worker".into())
            .spawn(move || {
                while !worker_stopped.load(Ordering::Acquire) {
                    let _ = messages.recv_timeout(Duration::from_millis(10));
                    let devices: Vec<_> = worker_pending
                        .lock()
                        .unwrap()
                        .values()
                        .map(|request| request.device.clone())
                        .collect();
                    for device in devices {
                        if let Err(error) = device.poll(wgpu::PollType::Poll) {
                            let failed: Vec<_> = worker_pending
                                .lock()
                                .unwrap()
                                .iter()
                                .filter(|(_, request)| request.device == device)
                                .map(|(id, _)| *id)
                                .collect();
                            for id in failed {
                                worker_ring.cancel(id);
                                if let Some(request) = worker_pending.lock().unwrap().remove(&id) {
                                    let _ = request.respond_to.send(AgentResponse::internal(
                                        format!("capture GPU polling failed: {error}"),
                                    ));
                                }
                            }
                        }
                    }
                    for result in worker_ring.take_ready() {
                        let request = worker_pending.lock().unwrap().remove(&result.id);
                        if let Some(request) = request {
                            let _ = request
                                .respond_to
                                .send(encode_capture_result(result, request.request.format));
                        }
                    }
                }
                for (_, request) in worker_pending.lock().unwrap().drain() {
                    let _ = request
                        .respond_to
                        .send(AgentResponse::unavailable("capture worker stopped"));
                }
            })
            .expect("failed to start capture worker");
        Self {
            ring,
            pending,
            wake,
            stopped,
            worker_thread: Some(worker_thread),
        }
    }

    pub fn pending_count(&self) -> usize {
        self.pending.lock().unwrap().len()
    }

    pub fn enqueue(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        target: &crate::capture::CaptureTarget,
        request: CaptureRequest,
        mut metadata: crate::readback::ReadbackMetadata,
        respond_to: mpsc::Sender<AgentResponse>,
    ) -> Result<crate::readback::ReadbackId, AgentResponse> {
        if self.pending_count() >= 3 {
            return Err(AgentResponse::busy("capture staging slots are occupied"));
        }
        metadata.target = request.target.clone();
        if request.format == CaptureFormat::Png {
            match target {
                crate::capture::CaptureTarget::Texture(texture)
                    if png_format_supported(texture.format()) => {}
                _ => {
                    return Err(AgentResponse::bad_request(
                        "PNG requires an RGBA8 or BGRA8 texture; request raw for other targets",
                    ));
                }
            }
        }
        let id = match target {
            crate::capture::CaptureTarget::Texture(texture) => self
                .ring
                .enqueue_texture(device, encoder, texture, metadata),
            crate::capture::CaptureTarget::Buffer {
                buffer,
                offset,
                size,
            } => self
                .ring
                .enqueue_buffer(device, encoder, buffer, *offset, *size, metadata),
        }
        .map_err(|error| match error {
            crate::readback::ReadbackError::Busy => AgentResponse::busy(error.to_string()),
            _ => AgentResponse::bad_request(error.to_string()),
        })?;
        self.pending.lock().unwrap().insert(
            id,
            PendingCapture {
                request,
                respond_to,
                device: device.clone(),
            },
        );
        let _ = self.wake.try_send(());
        Ok(id)
    }

    pub fn cancel_unsubmitted(&self, id: crate::readback::ReadbackId) {
        self.ring.cancel_unsubmitted(id);
        if let Some(request) = self.pending.lock().unwrap().remove(&id) {
            let _ = request.respond_to.send(AgentResponse::unavailable(
                "capture submission was cancelled",
            ));
        }
    }
}

impl Default for AgentCaptureWorker {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AgentCaptureWorker {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Release);
        let _ = self.wake.try_send(());
        if let Some(worker) = self.worker_thread.take() {
            let _ = worker.join();
        }
    }
}

fn png_format_supported(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Bgra8Unorm
            | wgpu::TextureFormat::Bgra8UnormSrgb
    )
}

fn encode_capture_result(
    result: crate::readback::ReadbackResult,
    format: CaptureFormat,
) -> AgentResponse {
    let mut bytes = match result.data {
        Ok(bytes) => bytes,
        Err(error) => return AgentResponse::internal(error.to_string()),
    };
    let meta = result.metadata;
    let metadata = json!({
        "target":meta.target,"submission_id":meta.submission_id,"tick_id":meta.tick_id,"frame_id":meta.frame_id,
        "size":meta.size,"texture_format":meta.texture_format.map(|format|format!("{format:?}")),
        "bytes_per_row":meta.bytes_per_row,"buffer_offset":meta.buffer_offset,"encoding":format,
    });
    let content_type = if format == CaptureFormat::Png {
        let Some([width, height]) = meta.size else {
            return AgentResponse::bad_request("PNG requires a texture");
        };
        if meta.texture_format.is_some_and(|format| {
            matches!(
                format,
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
            )
        }) {
            for pixel in bytes.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }
        let mut png = Vec::new();
        if let Err(error) = image::codecs::png::PngEncoder::new(&mut png).write_image(
            &bytes,
            width,
            height,
            image::ExtendedColorType::Rgba8,
        ) {
            return AgentResponse::internal(format!("PNG encoding failed: {error}"));
        }
        bytes = png;
        "image/png"
    } else {
        "application/octet-stream"
    };
    AgentResponse::Bytes {
        bytes: Arc::new(bytes),
        content_type: content_type.into(),
        metadata,
    }
}
pub fn apply_basic_camera_command<C: InteractiveCamera + ?Sized>(
    camera: &mut C,
    command: &AgentCommand,
) -> Result<(), String> {
    match *command {
        AgentCommand::CameraDrag { dx, dy, button } => {
            let (left, right, middle) = button.flags();
            camera.on_mouse_drag(dx, dy, left, right, middle);
            Ok(())
        }
        AgentCommand::CameraZoom { delta } => {
            camera.on_scroll(delta);
            Ok(())
        }
        AgentCommand::CameraSetEnabled { enabled } => {
            camera.set_enabled(enabled);
            Ok(())
        }
        AgentCommand::CameraSetOrbit { .. } => {
            Err("camera.set_orbit is only supported by OrbitCamera".to_string())
        }
        AgentCommand::CameraSet2d { .. } => {
            Err("camera.set_2d is only supported by Camera2d".to_string())
        }
        AgentCommand::AppCommand { .. }
        | AgentCommand::AppGpuCommand { .. }
        | AgentCommand::RunUntilCompleted { .. }
        | AgentCommand::Pause
        | AgentCommand::Resume
        | AgentCommand::Redraw
        | AgentCommand::Shutdown => Err("not a camera command".to_string()),
    }
}

#[derive(Clone)]
struct Server {
    request_tx: mpsc::SyncSender<AgentRequest>,
    wake: Arc<dyn Fn() + Send + Sync>,
    snapshot: Arc<Mutex<AgentSnapshot>>,
    jobs: Arc<JobStore>,
    stopped: Arc<AtomicBool>,
    config: AgentConfig,
}

fn serve_http(listener: TcpListener, server: Server) {
    tracing::info!(
        "mikage agent HTTP API listening on {}",
        listener.local_addr().unwrap()
    );
    let connections = Arc::new(AtomicUsize::new(0));
    while !server.stopped.load(Ordering::Acquire) {
        match listener.accept() {
            Ok((mut stream, _)) => {
                if connections.fetch_add(1, Ordering::AcqRel) >= server.config.max_connections {
                    connections.fetch_sub(1, Ordering::AcqRel);
                    let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
                    let _ = write_http_response(
                        &mut stream,
                        HttpResponse::json_error(429, "too many connections"),
                    );
                    continue;
                }
                let server = server.clone();
                let connections = connections.clone();
                thread::spawn(move || {
                    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
                    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
                    let response = match read_http_request(&mut stream) {
                        Ok(request) => handle_http_request(request, &server),
                        Err(error) => {
                            HttpResponse::json_error(400, &format!("invalid HTTP request: {error}"))
                        }
                    };
                    let _ = write_http_response(&mut stream, response);
                    connections.fetch_sub(1, Ordering::AcqRel);
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(10))
            }
            Err(error) => {
                tracing::warn!("agent HTTP accept failed: {error}");
                thread::sleep(Duration::from_millis(10));
            }
        }
    }
}

fn handle_http_request(request: HttpRequest, server: &Server) -> HttpResponse {
    if !authorized(&request.headers, server.config.auth_token.as_deref()) {
        return HttpResponse::json_error(401, "unauthorized");
    }
    if request.method == "OPTIONS" {
        return HttpResponse::empty(204);
    }
    if request.method == "GET" && request.path.starts_with("/jobs/") {
        let path = request.path.trim_start_matches("/jobs/");
        let (id, result) = match path.strip_suffix("/result") {
            Some(id) => (id, true),
            None => (path, false),
        };
        let Ok(id) = id.parse::<JobId>() else {
            return HttpResponse::json_error(404, "not found");
        };
        return if result {
            match server.jobs.result(id) {
                Ok(Some(response)) => response.into(),
                Ok(None) => HttpResponse::json_value(202, json!({"id":id,"state":"pending"})),
                Err(error) => error.into(),
            }
        } else {
            match server.jobs.status(id) {
                Some(status) => HttpResponse::json_value(200, status),
                None => HttpResponse::json_error(404, "unknown or expired job"),
            }
        };
    }
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/status") => match server.snapshot.lock() {
            Ok(snapshot) => HttpResponse::json(200, &*snapshot),
            Err(_) => HttpResponse::json_error(500, "snapshot lock poisoned"),
        },
        ("POST", "/command") => match serde_json::from_slice::<AgentCommand>(&request.body) {
            Ok(command) => {
                if let AgentCommand::RunUntilCompleted { dt, .. } = command
                    && (!dt.is_finite() || dt <= 0.0)
                {
                    return HttpResponse::json_error(400, "dt must be positive and finite");
                }
                if matches!(
                    command,
                    AgentCommand::AppGpuCommand { .. } | AgentCommand::RunUntilCompleted { .. }
                ) {
                    match enqueue_job(server, AgentRequestKind::Command(command)) {
                        Ok(id) => accepted_job(id),
                        Err(error) => error.into(),
                    }
                } else {
                    dispatch(server, AgentRequestKind::Command(command))
                }
            }
            Err(error) => HttpResponse::json_error(400, &format!("invalid command JSON: {error}")),
        },
        ("POST", "/captures") => {
            let capture = if request.body.is_empty() {
                Ok(CaptureRequest::default())
            } else {
                serde_json::from_slice(&request.body)
            };
            match capture {
                Ok(capture) if !capture.target.is_empty() => {
                    match enqueue_job(server, AgentRequestKind::Capture(capture)) {
                        Ok(id) => accepted_job(id),
                        Err(error) => error.into(),
                    }
                }
                Ok(_) => HttpResponse::json_error(400, "capture target must not be empty"),
                Err(error) => {
                    HttpResponse::json_error(400, &format!("invalid capture JSON: {error}"))
                }
            }
        }
        ("GET", "/screenshot") | ("POST", "/screenshot") => {
            match enqueue_job(server, AgentRequestKind::Capture(CaptureRequest::default())) {
                Ok(id) => match server.jobs.wait(id, server.config.request_timeout) {
                    Ok(response) | Err(response) => response.into(),
                },
                Err(error) => error.into(),
            }
        }
        _ => HttpResponse::json_error(404, "not found"),
    }
}

fn accepted_job(id: JobId) -> HttpResponse {
    HttpResponse::json_value(
        202,
        json!({"id":id,"state":"pending","status_url":format!("/jobs/{id}"),"result_url":format!("/jobs/{id}/result")}),
    )
}

fn authorized(headers: &HashMap<String, String>, token: Option<&str>) -> bool {
    let Some(token) = token else {
        return true;
    };
    let bearer = format!("Bearer {token}");
    headers
        .get("authorization")
        .is_some_and(|value| value == &bearer)
        || headers
            .get("x-mikage-token")
            .is_some_and(|value| value == token)
}

fn enqueue_job(server: &Server, kind: AgentRequestKind) -> Result<JobId, AgentResponse> {
    if server.stopped.load(Ordering::Acquire) {
        return Err(AgentResponse::unavailable("application has shut down"));
    }
    let id = server.jobs.create()?;
    let (respond_to, responses) = mpsc::channel();
    let request = AgentRequest {
        kind,
        respond_to,
        job_id: Some(id),
    };
    if let Err(error) = server.request_tx.try_send(request) {
        let response = match error {
            mpsc::TrySendError::Full(_) => AgentResponse::busy("application request queue is full"),
            mpsc::TrySendError::Disconnected(_) => {
                AgentResponse::unavailable("application is unavailable")
            }
        };
        server.jobs.discard(id);
        return Err(response);
    }
    // At most max_jobs waiters can exist, and every waiter expires with its job.
    // The render thread uses the same response sender for immediate and deferred work.
    let jobs = server.jobs.clone();
    let ttl = server.config.job_ttl;
    thread::spawn(move || match responses.recv_timeout(ttl) {
        Ok(response) => jobs.complete(id, response),
        Err(mpsc::RecvTimeoutError::Disconnected) => jobs.complete(
            id,
            AgentResponse::unavailable("application dropped the job"),
        ),
        Err(mpsc::RecvTimeoutError::Timeout) => jobs.complete(
            id,
            AgentResponse::Error {
                status: 504,
                message: "job expired".into(),
            },
        ),
    });
    (server.wake)();
    Ok(id)
}

fn dispatch(server: &Server, kind: AgentRequestKind) -> HttpResponse {
    let (respond_to, response_rx) = mpsc::channel();
    if let Err(error) = server.request_tx.try_send(AgentRequest {
        kind,
        respond_to,
        job_id: None,
    }) {
        return match error {
            mpsc::TrySendError::Full(_) => {
                HttpResponse::json_error(429, "application request queue is full")
            }
            mpsc::TrySendError::Disconnected(_) => {
                HttpResponse::json_error(503, "application is unavailable")
            }
        };
    }
    (server.wake)();
    match response_rx.recv_timeout(server.config.request_timeout) {
        Ok(response) => response.into(),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            HttpResponse::json_error(504, "timed out waiting for application response")
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            HttpResponse::json_error(503, "application response channel closed")
        }
    }
}
struct HttpRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

fn read_http_request(stream: &mut TcpStream) -> std::io::Result<HttpRequest> {
    let mut buffer = Vec::new();
    let header_end = loop {
        let mut chunk = [0; 1024];
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "connection closed before headers",
            ));
        }
        buffer.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_header_end(&buffer) {
            break pos;
        }
        if buffer.len() > 64 * 1024 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "headers too large",
            ));
        }
    };

    let header_bytes = &buffer[..header_end];
    let header_text = std::str::from_utf8(header_bytes)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
    let mut lines = header_text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "empty request"))?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing method"))?
        .to_string();
    let raw_path = parts
        .next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing path"))?;
    let path = raw_path.split('?').next().unwrap_or(raw_path).to_string();

    let mut headers = HashMap::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }

    if headers.contains_key("transfer-encoding") {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "transfer encoding is unsupported",
        ));
    }
    let content_length = headers
        .get("content-length")
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid content length")
        })?
        .unwrap_or(0);
    if content_length > 1024 * 1024 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "request body exceeds 1 MiB",
        ));
    }
    let body_start = header_end + 4;
    let mut body = buffer[body_start..].to_vec();
    while body.len() < content_length {
        let mut chunk = vec![0; content_length - body.len()];
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "incomplete request body",
            ));
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(content_length);

    Ok(HttpRequest {
        method,
        path,
        headers,
        body,
    })
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

struct HttpResponse {
    status: u16,
    content_type: String,
    body: Arc<Vec<u8>>,
}

impl HttpResponse {
    fn empty(status: u16) -> Self {
        Self {
            status,
            content_type: "application/octet-stream".to_string(),
            body: Arc::new(Vec::new()),
        }
    }

    fn json<T: Serialize>(status: u16, value: &T) -> Self {
        Self::json_value(status, serde_json::to_value(value).unwrap_or(Value::Null))
    }

    fn json_value(status: u16, value: Value) -> Self {
        Self {
            status,
            content_type: "application/json".to_string(),
            body: Arc::new(
                serde_json::to_vec(&value).unwrap_or_else(|_| b"{\"ok\":false}".to_vec()),
            ),
        }
    }

    fn json_error(status: u16, message: &str) -> Self {
        Self::json_value(status, json!({ "ok": false, "error": message }))
    }
}

impl From<AgentResponse> for HttpResponse {
    fn from(response: AgentResponse) -> Self {
        match response {
            AgentResponse::Json(value) => Self::json_value(200, value),
            AgentResponse::Png(bytes) => Self {
                status: 200,
                content_type: "image/png".into(),
                body: Arc::new(bytes),
            },
            AgentResponse::Bytes {
                bytes,
                content_type,
                ..
            } => Self {
                status: 200,
                content_type,
                body: bytes,
            },
            AgentResponse::Error { status, message } => Self::json_error(status, &message),
        }
    }
}

fn write_http_response(stream: &mut TcpStream, response: HttpResponse) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {} {}\r\n\
         Content-Type: {}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n",
        response.status,
        status_text(response.status),
        response.content_type,
        response.body.len(),
    )?;
    stream.write_all(&response.body)
}

fn status_text(status: u16) -> &'static str {
    match status {
        200 => "OK",
        202 => "Accepted",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_camera_drag_command() {
        let command: AgentCommand =
            serde_json::from_str(r#"{"op":"camera.drag","dx":-80.0,"dy":20.0,"button":"right"}"#)
                .unwrap();

        match command {
            AgentCommand::CameraDrag { dx, dy, button } => {
                assert_eq!(dx, -80.0);
                assert_eq!(dy, 20.0);
                assert!(matches!(button, AgentMouseButton::Right));
            }
            _ => panic!("expected camera drag"),
        }
    }

    #[test]
    fn parses_shutdown_command() {
        let command: AgentCommand = serde_json::from_str(r#"{"op":"shutdown"}"#).unwrap();
        assert!(matches!(command, AgentCommand::Shutdown));
    }

    #[test]
    fn auth_accepts_bearer_or_header() {
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), "Bearer secret".to_string());
        assert!(authorized(&headers, Some("secret")));

        headers.clear();
        headers.insert("x-mikage-token".to_string(), "secret".to_string());
        assert!(authorized(&headers, Some("secret")));
        assert!(!authorized(&headers, Some("other")));
    }

    fn test_server(config: AgentConfig) -> (Server, mpsc::Receiver<AgentRequest>) {
        let (request_tx, requests) = mpsc::sync_channel(config.max_jobs);
        let server = Server {
            request_tx,
            wake: Arc::new(|| {}),
            snapshot: Arc::new(Mutex::new(AgentSnapshot::new(config.bind_addr))),
            jobs: Arc::new(JobStore::new(&config)),
            stopped: Arc::new(AtomicBool::new(false)),
            config,
        };
        (server, requests)
    }

    fn request(method: &str, path: &str, body: Value) -> HttpRequest {
        HttpRequest {
            method: method.into(),
            path: path.into(),
            headers: HashMap::new(),
            body: serde_json::to_vec(&body).unwrap(),
        }
    }

    #[test]
    fn capture_returns_accepted_before_render_response_and_retains_result() {
        let (server, requests) = test_server(AgentConfig::default());
        let response = handle_http_request(
            request(
                "POST",
                "/captures",
                json!({"target":"particles","format":"raw","exact":true,"at_tick":12}),
            ),
            &server,
        );
        assert_eq!(response.status, 202);
        let json: Value = serde_json::from_slice(&response.body).unwrap();
        let id = json["id"].as_u64().unwrap();
        let job = requests.try_recv().unwrap();
        assert_eq!(job.job_id, Some(id));
        assert!(matches!(
            job.kind,
            AgentRequestKind::Capture(CaptureRequest {
                exact: true,
                at_tick: Some(12),
                ..
            })
        ));
        assert_eq!(server.jobs.status(id).unwrap()["state"], "pending");
        job.respond_to
            .send(AgentResponse::Bytes {
                bytes: Arc::new(vec![1, 2, 3, 4]),
                content_type: "application/octet-stream".into(),
                metadata: json!({"tick_id":12}),
            })
            .ok();
        assert!(server.jobs.wait(id, Duration::from_secs(1)).is_ok());
        assert_eq!(server.jobs.status(id).unwrap()["metadata"]["tick_id"], 12);
        let result = handle_http_request(
            request("GET", &format!("/jobs/{id}/result"), Value::Null),
            &server,
        );
        assert_eq!(result.status, 200);
        assert_eq!(&*result.body, &[1, 2, 3, 4]);
    }

    #[test]
    fn rejected_enqueue_releases_job_capacity_immediately() {
        let (server, requests) = test_server(AgentConfig {
            max_jobs: 1,
            ..Default::default()
        });
        // Immediate commands share the bounded request queue but reserve no job.
        let (respond_to, _response) = mpsc::channel();
        assert!(
            server
                .request_tx
                .try_send(AgentRequest {
                    kind: AgentRequestKind::Command(AgentCommand::Redraw),
                    respond_to,
                    job_id: None,
                })
                .is_ok()
        );

        assert!(matches!(
            enqueue_job(
                &server,
                AgentRequestKind::Capture(CaptureRequest::default())
            ),
            Err(AgentResponse::Error { status: 429, .. })
        ));
        assert!(server.jobs.state.lock().unwrap().jobs.is_empty());
        let _ = requests.try_recv().unwrap();

        let accepted = enqueue_job(
            &server,
            AgentRequestKind::Capture(CaptureRequest::default()),
        )
        .unwrap_or_else(|_| panic!("job capacity must recover when the request queue drains"));
        let request = requests.try_recv().unwrap();
        assert_eq!(request.job_id, Some(accepted));
        request.respond_to.send(AgentResponse::ok()).ok();
        assert!(server.jobs.wait(accepted, Duration::from_secs(1)).is_ok());

        let (disconnected, receiver) = test_server(AgentConfig::default());
        drop(receiver);
        assert!(matches!(
            enqueue_job(
                &disconnected,
                AgentRequestKind::Capture(CaptureRequest::default())
            ),
            Err(AgentResponse::Error { status: 503, .. })
        ));
        assert!(disconnected.jobs.state.lock().unwrap().jobs.is_empty());
    }

    #[test]
    fn job_capacity_ttl_and_result_bytes_are_bounded() {
        let config = AgentConfig {
            max_jobs: 1,
            max_result_bytes: 4,
            ..Default::default()
        };
        let jobs = JobStore::new(&config);
        let id = jobs
            .create()
            .unwrap_or_else(|_| panic!("first job should fit"));
        assert!(matches!(
            jobs.create(),
            Err(AgentResponse::Error { status: 429, .. })
        ));
        jobs.complete(id, AgentResponse::Png(vec![0; 5]));
        assert_eq!(jobs.status(id).unwrap()["state"], "failed");
        assert_eq!(jobs.state.lock().unwrap().result_bytes, 0);
        jobs.state
            .lock()
            .unwrap()
            .jobs
            .get_mut(&id)
            .unwrap()
            .created = Instant::now() - config.job_ttl;
        assert!(jobs.status(id).is_none());
        assert!(jobs.create().is_ok());
    }

    #[test]
    fn deferred_commands_validate_dt_and_shutdown_fails_pending_jobs() {
        let (server, requests) = test_server(AgentConfig::default());
        let invalid = handle_http_request(
            request(
                "POST",
                "/command",
                json!({"op":"run_until_completed","target_tick":8,"dt":0.0}),
            ),
            &server,
        );
        assert_eq!(invalid.status, 400);
        let valid = handle_http_request(
            request(
                "POST",
                "/command",
                json!({"op":"run_until_completed","target_tick":8,"dt":0.01}),
            ),
            &server,
        );
        assert_eq!(valid.status, 202);
        let request = requests.try_recv().unwrap();
        let id = request.job_id.unwrap();
        server.jobs.fail_all("GPU device lost");
        assert_eq!(server.jobs.status(id).unwrap()["state"], "failed");
        assert_eq!(server.jobs.status(id).unwrap()["error"], "GPU device lost");
    }

    #[test]
    fn png_conversion_swizzles_bgra_without_changing_raw_bytes() {
        let make_result = || crate::readback::ReadbackResult {
            id: crate::readback::ReadbackId(1),
            metadata: crate::readback::ReadbackMetadata {
                size: Some([1, 1]),
                texture_format: Some(wgpu::TextureFormat::Bgra8UnormSrgb),
                bytes_per_row: Some(4),
                ..Default::default()
            },
            data: Ok(vec![10, 20, 30, 255]),
        };
        let AgentResponse::Bytes { bytes, .. } =
            encode_capture_result(make_result(), CaptureFormat::Raw)
        else {
            panic!("expected bytes")
        };
        assert_eq!(&*bytes, &[10, 20, 30, 255]);
        let AgentResponse::Bytes { bytes, .. } =
            encode_capture_result(make_result(), CaptureFormat::Png)
        else {
            panic!("expected PNG")
        };
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgba8();
        assert_eq!(decoded.as_raw(), &[30, 20, 10, 255]);
    }

    #[test]
    fn gpu_capture_worker_completes_raw_and_shutdown_replies_to_pending() {
        use wgpu::util::DeviceExt;
        let gpu = match pollster::block_on(crate::GpuContext::headless(
            crate::GpuDescriptor::default(),
        )) {
            Ok(gpu) => gpu,
            Err(crate::GpuInitError::AdapterUnavailable(error))
                if std::env::var_os("MIKAGE_REQUIRE_GPU").is_none() =>
            {
                eprintln!("GPU test unavailable: {error}");
                return;
            }
            Err(error) => panic!("GPU initialization failed: {error}"),
        };
        let buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &[1, 2, 3, 4],
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        let target = crate::capture::CaptureTarget::Buffer {
            buffer,
            offset: 0,
            size: 4,
        };
        let worker = AgentCaptureWorker::new();
        let (tx, rx) = mpsc::channel();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        assert!(matches!(
            worker.enqueue(
                &gpu.device,
                &mut encoder,
                &target,
                CaptureRequest::default(),
                crate::ReadbackMetadata::default(),
                tx.clone()
            ),
            Err(AgentResponse::Error { status: 400, .. })
        ));
        worker
            .enqueue(
                &gpu.device,
                &mut encoder,
                &target,
                CaptureRequest {
                    format: CaptureFormat::Raw,
                    ..Default::default()
                },
                crate::ReadbackMetadata {
                    tick_id: Some(9),
                    ..Default::default()
                },
                tx,
            )
            .unwrap_or_else(|_| panic!("raw readback should be accepted"));
        gpu.queue.submit([encoder.finish()]);
        let response = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("worker must poll and complete without a host poll");
        let AgentResponse::Bytes {
            bytes, metadata, ..
        } = response
        else {
            panic!("expected raw bytes")
        };
        assert_eq!(&*bytes, &[1, 2, 3, 4]);
        assert_eq!(metadata["tick_id"], 9);

        let (tx, rx) = mpsc::channel();
        let mut unsubmitted = gpu.device.create_command_encoder(&Default::default());
        worker
            .enqueue(
                &gpu.device,
                &mut unsubmitted,
                &target,
                CaptureRequest {
                    format: CaptureFormat::Raw,
                    ..Default::default()
                },
                crate::ReadbackMetadata::default(),
                tx,
            )
            .unwrap_or_else(|_| panic!("slot should be reusable"));
        drop(unsubmitted);
        drop(worker);
        assert!(matches!(
            rx.recv_timeout(Duration::from_secs(1)),
            Ok(AgentResponse::Error { status: 503, .. })
        ));
    }
}
