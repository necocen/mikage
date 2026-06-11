//! Local HTTP control plane for LLM/debugging agents.
//!
//! This module is native-only and enabled with the `agent` feature.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Duration;

use image::ImageEncoder;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use winit::dpi::PhysicalSize;
use winit::event_loop::EventLoopProxy;

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
    pub(crate) fn flags(self) -> (bool, bool, bool) {
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
    /// Requests a redraw and returns immediately.
    #[serde(rename = "redraw")]
    Redraw,
    /// Exits the application event loop after returning a response.
    #[serde(rename = "shutdown")]
    Shutdown,
}

impl AgentCommand {
    pub(crate) fn is_camera_command(&self) -> bool {
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
        }
    }
}

pub(crate) struct AgentBridge {
    requests: mpsc::Receiver<AgentRequest>,
    snapshot: Arc<Mutex<AgentSnapshot>>,
}

impl AgentBridge {
    pub(crate) fn start(
        config: AgentConfig,
        event_proxy: EventLoopProxy<()>,
    ) -> std::io::Result<Self> {
        let listener = TcpListener::bind(config.bind_addr)?;
        let bind_addr = listener.local_addr()?;
        let snapshot = Arc::new(Mutex::new(AgentSnapshot::new(bind_addr)));
        let (request_tx, requests) = mpsc::channel();

        if let Some(path) = &config.write_connection_file {
            let connection = json!({
                "addr": bind_addr.to_string(),
                "auth": config.auth_token.as_ref().map(|_| "bearer")
            });
            let _ = std::fs::write(path, serde_json::to_vec_pretty(&connection).unwrap());
        }

        let server_snapshot = snapshot.clone();
        thread::Builder::new()
            .name("mikage-agent-http".to_string())
            .spawn(move || {
                serve_http(listener, request_tx, event_proxy, server_snapshot, config);
            })?;

        Ok(Self { requests, snapshot })
    }

    pub(crate) fn drain_requests(&mut self) -> Vec<AgentRequest> {
        self.requests.try_iter().collect()
    }

    pub(crate) fn update_snapshot(&self, update: impl FnOnce(&mut AgentSnapshot)) {
        if let Ok(mut snapshot) = self.snapshot.lock() {
            update(&mut snapshot);
        }
    }
}

pub(crate) struct AgentRequest {
    pub(crate) kind: AgentRequestKind,
    pub(crate) respond_to: mpsc::Sender<AgentResponse>,
}

pub(crate) enum AgentRequestKind {
    Command(AgentCommand),
    Screenshot,
}

pub(crate) enum AgentResponse {
    Json(Value),
    Png(Vec<u8>),
    Error { status: u16, message: String },
}

impl AgentResponse {
    pub(crate) fn ok() -> Self {
        Self::Json(json!({ "ok": true }))
    }

    pub(crate) fn json(value: Value) -> Self {
        Self::Json(value)
    }

    pub(crate) fn bad_request(message: impl Into<String>) -> Self {
        Self::Error {
            status: 400,
            message: message.into(),
        }
    }

    pub(crate) fn unavailable(message: impl Into<String>) -> Self {
        Self::Error {
            status: 503,
            message: message.into(),
        }
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::Error {
            status: 500,
            message: message.into(),
        }
    }
}

pub(crate) fn apply_basic_camera_command<C: InteractiveCamera + ?Sized>(
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
        AgentCommand::AppCommand { .. } | AgentCommand::Redraw | AgentCommand::Shutdown => {
            Err("not a camera command".to_string())
        }
    }
}

pub(crate) struct ScreenshotReadback {
    buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    bytes_per_row: u32,
    format: wgpu::TextureFormat,
}

impl ScreenshotReadback {
    pub(crate) fn encode(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
        size: PhysicalSize<u32>,
        format: wgpu::TextureFormat,
    ) -> Result<Self, String> {
        let width = size.width.max(1);
        let height = size.height.max(1);
        let bytes_per_row = align_to(width * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mikage_agent_screenshot_readback"),
            size: (bytes_per_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        Ok(Self {
            buffer,
            width,
            height,
            bytes_per_row,
            format,
        })
    }

    pub(crate) fn read_png(self, device: &wgpu::Device) -> AgentResponse {
        let slice = self.buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        if let Err(err) = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        }) {
            return AgentResponse::internal(format!("device poll failed: {err:?}"));
        }

        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                return AgentResponse::internal(format!("screenshot map failed: {err:?}"));
            }
            Err(err) => {
                return AgentResponse::internal(format!("screenshot map channel failed: {err}"));
            }
        }

        let data = slice.get_mapped_range();
        let mut rgba = Vec::with_capacity((self.width * self.height * 4) as usize);
        for row in 0..self.height {
            let start = (row * self.bytes_per_row) as usize;
            let row_bytes = &data[start..start + (self.width * 4) as usize];
            append_rgba_row(&mut rgba, row_bytes, self.format);
        }
        drop(data);
        self.buffer.unmap();

        let mut png = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut png);
        match encoder.write_image(
            &rgba,
            self.width,
            self.height,
            image::ExtendedColorType::Rgba8,
        ) {
            Ok(()) => AgentResponse::Png(png),
            Err(err) => AgentResponse::internal(format!("png encode failed: {err}")),
        }
    }
}

fn append_rgba_row(out: &mut Vec<u8>, row: &[u8], format: wgpu::TextureFormat) {
    match format {
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
            for px in row.chunks_exact(4) {
                out.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
            }
        }
        _ => out.extend_from_slice(row),
    }
}

fn align_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

fn serve_http(
    listener: TcpListener,
    request_tx: mpsc::Sender<AgentRequest>,
    event_proxy: EventLoopProxy<()>,
    snapshot: Arc<Mutex<AgentSnapshot>>,
    config: AgentConfig,
) {
    tracing::info!(
        "mikage agent HTTP API listening on {}",
        listener.local_addr().unwrap()
    );
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let request_tx = request_tx.clone();
                let event_proxy = event_proxy.clone();
                let snapshot = snapshot.clone();
                let config = config.clone();
                thread::spawn(move || {
                    handle_connection(stream, request_tx, event_proxy, snapshot, config);
                });
            }
            Err(err) => tracing::warn!("agent HTTP accept failed: {err}"),
        }
    }
}

fn handle_connection(
    mut stream: TcpStream,
    request_tx: mpsc::Sender<AgentRequest>,
    event_proxy: EventLoopProxy<()>,
    snapshot: Arc<Mutex<AgentSnapshot>>,
    config: AgentConfig,
) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let response = match read_http_request(&mut stream) {
        Ok(request) => handle_http_request(request, request_tx, event_proxy, snapshot, &config),
        Err(err) => HttpResponse::json_error(400, &format!("invalid HTTP request: {err}")),
    };
    let _ = write_http_response(&mut stream, response);
}

fn handle_http_request(
    request: HttpRequest,
    request_tx: mpsc::Sender<AgentRequest>,
    event_proxy: EventLoopProxy<()>,
    snapshot: Arc<Mutex<AgentSnapshot>>,
    config: &AgentConfig,
) -> HttpResponse {
    if request.method == "OPTIONS" {
        return HttpResponse::empty(204);
    }

    if !authorized(&request.headers, config.auth_token.as_deref()) {
        return HttpResponse::json_error(401, "unauthorized");
    }

    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/status") => {
            let snapshot = snapshot.lock().map(|s| s.clone()).unwrap_or_else(|_| {
                let mut fallback = AgentSnapshot::new(config.bind_addr);
                fallback.app = json!({ "error": "snapshot lock poisoned" });
                fallback
            });
            HttpResponse::json(200, &snapshot)
        }
        ("POST", "/command") => match serde_json::from_slice::<AgentCommand>(&request.body) {
            Ok(command) => dispatch(
                AgentRequestKind::Command(command),
                request_tx,
                event_proxy,
                config,
            ),
            Err(err) => HttpResponse::json_error(400, &format!("invalid command JSON: {err}")),
        },
        ("GET", "/screenshot") | ("POST", "/screenshot") => dispatch(
            AgentRequestKind::Screenshot,
            request_tx,
            event_proxy,
            config,
        ),
        _ => HttpResponse::json_error(404, "not found"),
    }
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

fn dispatch(
    kind: AgentRequestKind,
    request_tx: mpsc::Sender<AgentRequest>,
    event_proxy: EventLoopProxy<()>,
    config: &AgentConfig,
) -> HttpResponse {
    let (respond_to, response_rx) = mpsc::channel();
    if request_tx.send(AgentRequest { kind, respond_to }).is_err() {
        return HttpResponse::json_error(503, "application event loop is not available");
    }
    if event_proxy.send_event(()).is_err() {
        return HttpResponse::json_error(503, "application event loop is closed");
    }

    match response_rx.recv_timeout(config.request_timeout) {
        Ok(AgentResponse::Json(value)) => HttpResponse::json_value(200, value),
        Ok(AgentResponse::Png(bytes)) => HttpResponse {
            status: 200,
            content_type: "image/png".to_string(),
            body: bytes,
        },
        Ok(AgentResponse::Error { status, message }) => HttpResponse::json_error(status, &message),
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

    let content_length = headers
        .get("content-length")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let body_start = header_end + 4;
    let mut body = buffer[body_start..].to_vec();
    while body.len() < content_length {
        let mut chunk = vec![0; content_length - body.len()];
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
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
    body: Vec<u8>,
}

impl HttpResponse {
    fn empty(status: u16) -> Self {
        Self {
            status,
            content_type: "application/octet-stream".to_string(),
            body: Vec::new(),
        }
    }

    fn json<T: Serialize>(status: u16, value: &T) -> Self {
        Self::json_value(status, serde_json::to_value(value).unwrap_or(Value::Null))
    }

    fn json_value(status: u16, value: Value) -> Self {
        Self {
            status,
            content_type: "application/json".to_string(),
            body: serde_json::to_vec(&value).unwrap_or_else(|_| b"{\"ok\":false}".to_vec()),
        }
    }

    fn json_error(status: u16, message: &str) -> Self {
        Self::json_value(status, json!({ "ok": false, "error": message }))
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
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
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
}
