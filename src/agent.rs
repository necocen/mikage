//! Local HTTP control plane for LLM/debugging agents.
//!
//! This module is native-only and enabled with the `agent` feature.

use std::collections::HashMap;
use std::net::{SocketAddr, TcpListener};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::watch;

mod capture;
mod http;
mod jobs;

pub use capture::AgentCaptureWorker;
use jobs::JobStore;

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
    shutdown: watch::Sender<bool>,
    server_thread: Option<thread::JoinHandle<()>>,
}

impl AgentBridge {
    pub fn start(
        config: AgentConfig,
        wake: impl Fn() + Send + Sync + 'static,
    ) -> std::io::Result<Self> {
        Self::start_with_limits(config, wake, http::Limits::default())
    }

    fn start_with_limits(
        config: AgentConfig,
        wake: impl Fn() + Send + Sync + 'static,
        limits: http::Limits,
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
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()?;
        let listener = {
            let _entered = runtime.enter();
            tokio::net::TcpListener::from_std(listener)?
        };
        let snapshot = Arc::new(Mutex::new(AgentSnapshot::new(bind_addr)));
        let jobs = Arc::new(JobStore::new(&config));
        let stopped = Arc::new(AtomicBool::new(false));
        let (shutdown, _) = watch::channel(false);
        let replies = Arc::new(http::ResponseRelay::new(
            config.max_jobs.saturating_add(config.max_connections),
        ));
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
            shutdown: shutdown.clone(),
            replies,
            config,
        };
        let dispatch = tracing::dispatcher::get_default(Clone::clone);
        let server_thread = thread::Builder::new()
            .name("mikage-agent-http".into())
            .spawn(move || {
                tracing::dispatcher::with_default(&dispatch, || {
                    runtime.block_on(http::serve(listener, server, limits))
                })
            })?;
        Ok(Self {
            requests,
            snapshot,
            jobs,
            stopped,
            shutdown,
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
        self.shutdown.send_replace(true);
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
    shutdown: watch::Sender<bool>,
    replies: Arc<http::ResponseRelay>,
}

async fn handle_http_request(request: HttpRequest, server: &Server) -> HttpResponse {
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
                    dispatch(server, AgentRequestKind::Command(command)).await
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
                Ok(id) => match server.jobs.wait(id, server.config.request_timeout).await {
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
    let response = match server.replies.register(responses, server.config.job_ttl) {
        Ok(response) => response,
        Err(error) => {
            server.jobs.discard(id);
            return Err(error);
        }
    };
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
    // The compatibility relay keeps the public std::mpsc sender while all
    // waiters run on the one HTTP runtime, without one OS thread per job.
    let jobs = server.jobs.clone();
    tokio::spawn(async move {
        jobs.complete(id, http::receive_reply(response).await);
    });
    (server.wake)();
    Ok(id)
}

async fn dispatch(server: &Server, kind: AgentRequestKind) -> HttpResponse {
    let (respond_to, response_rx) = mpsc::channel();
    let response = match server
        .replies
        .register(response_rx, server.config.request_timeout)
    {
        Ok(response) => response,
        Err(error) => return error.into(),
    };
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
    http::receive_reply(response).await.into()
}
struct HttpRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
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
        let (shutdown, _) = watch::channel(false);
        let replies = Arc::new(http::ResponseRelay::new(
            config.max_jobs.saturating_add(config.max_connections),
        ));
        tokio::spawn(replies.clone().run(shutdown.subscribe()));
        let server = Server {
            request_tx,
            wake: Arc::new(|| {}),
            snapshot: Arc::new(Mutex::new(AgentSnapshot::new(config.bind_addr))),
            jobs: Arc::new(JobStore::new(&config)),
            stopped: Arc::new(AtomicBool::new(false)),
            config,
            shutdown,
            replies,
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

    #[tokio::test(flavor = "current_thread")]
    async fn capture_returns_accepted_before_render_response_and_retains_result() {
        let (server, requests) = test_server(AgentConfig::default());
        let response = handle_http_request(
            request(
                "POST",
                "/captures",
                json!({"target":"particles","format":"raw","exact":true,"at_tick":12}),
            ),
            &server,
        )
        .await;
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
        assert!(server.jobs.wait(id, Duration::from_secs(1)).await.is_ok());
        assert_eq!(server.jobs.status(id).unwrap()["metadata"]["tick_id"], 12);
        let result = handle_http_request(
            request("GET", &format!("/jobs/{id}/result"), Value::Null),
            &server,
        )
        .await;
        assert_eq!(result.status, 200);
        assert_eq!(&*result.body, &[1, 2, 3, 4]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn rejected_enqueue_releases_job_capacity_immediately() {
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
        assert!(server.jobs.is_empty());
        let _ = requests.try_recv().unwrap();

        let accepted = enqueue_job(
            &server,
            AgentRequestKind::Capture(CaptureRequest::default()),
        )
        .unwrap_or_else(|_| panic!("job capacity must recover when the request queue drains"));
        let request = requests.try_recv().unwrap();
        assert_eq!(request.job_id, Some(accepted));
        request.respond_to.send(AgentResponse::ok()).ok();
        assert!(
            server
                .jobs
                .wait(accepted, Duration::from_secs(1))
                .await
                .is_ok()
        );

        let (disconnected, receiver) = test_server(AgentConfig::default());
        drop(receiver);
        assert!(matches!(
            enqueue_job(
                &disconnected,
                AgentRequestKind::Capture(CaptureRequest::default())
            ),
            Err(AgentResponse::Error { status: 503, .. })
        ));
        assert!(disconnected.jobs.is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn deferred_commands_validate_dt_and_shutdown_fails_pending_jobs() {
        let (server, requests) = test_server(AgentConfig::default());
        let invalid = handle_http_request(
            request(
                "POST",
                "/command",
                json!({"op":"run_until_completed","target_tick":8,"dt":0.0}),
            ),
            &server,
        )
        .await;
        assert_eq!(invalid.status, 400);
        let valid = handle_http_request(
            request(
                "POST",
                "/command",
                json!({"op":"run_until_completed","target_tick":8,"dt":0.01}),
            ),
            &server,
        )
        .await;
        assert_eq!(valid.status, 202);
        let request = requests.try_recv().unwrap();
        let id = request.job_id.unwrap();
        server.jobs.fail_all("GPU device lost");
        assert_eq!(server.jobs.status(id).unwrap()["state"], "failed");
        assert_eq!(server.jobs.status(id).unwrap()["error"], "GPU device lost");
    }
}
