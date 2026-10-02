//! HTTP framing and flow control belong to axum/Hyper. This module owns the
//! single I/O thread and the compatibility boundary to public std channels.

use super::*;
use axum::body::{Body, to_bytes};
use axum::extract::{Request, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::{Router, middleware};
use hyper::server::conn::http1;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::service::TowerToHyperService;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio::task::JoinSet;
use tokio::time::{sleep, timeout};

const MAX_BODY_BYTES: usize = 1024 * 1024;
const MAX_HEADER_BYTES: usize = 64 * 1024;
const REPLY_POLL_INTERVAL: Duration = Duration::from_millis(5);

#[derive(Clone, Copy)]
pub(super) struct Limits {
    pub io_timeout: Duration,
    pub shutdown_timeout: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            io_timeout: Duration::from_secs(5),
            shutdown_timeout: Duration::from_secs(1),
        }
    }
}

struct PendingReply {
    receiver: mpsc::Receiver<AgentResponse>,
    completion: oneshot::Sender<AgentResponse>,
    started: Instant,
    timeout: Duration,
}

/// std::mpsc::Sender is part of the public API. Poll its receivers only while
/// there is pending work, on one task, with no blocking pool or per-job threads.
pub(super) struct ResponseRelay {
    pending: Mutex<Vec<PendingReply>>,
    registered: Notify,
    capacity: usize,
}

impl ResponseRelay {
    pub fn new(capacity: usize) -> Self {
        Self {
            pending: Mutex::new(Vec::new()),
            registered: Notify::new(),
            capacity,
        }
    }

    pub fn register(
        &self,
        receiver: mpsc::Receiver<AgentResponse>,
        timeout: Duration,
    ) -> Result<oneshot::Receiver<AgentResponse>, AgentResponse> {
        let mut pending = self.pending.lock().unwrap();
        // Failed queue admission and disconnected HTTP callers release their
        // reservations before accepting more work, even between relay scans.
        pending.retain(|reply| !reply.completion.is_closed());
        if pending.len() >= self.capacity {
            return Err(AgentResponse::busy(
                "application response capacity exhausted",
            ));
        }
        let (completion, result) = oneshot::channel();
        pending.push(PendingReply {
            receiver,
            completion,
            started: Instant::now(),
            timeout,
        });
        self.registered.notify_one();
        Ok(result)
    }

    pub async fn run(self: Arc<Self>, mut shutdown: watch::Receiver<bool>) {
        let mut closed = false;
        loop {
            let stopping = closed || *shutdown.borrow();
            let has_pending = {
                let mut pending = self.pending.lock().unwrap();
                let mut index = 0;
                while index < pending.len() {
                    let reply = &pending[index];
                    if reply.completion.is_closed() {
                        pending.swap_remove(index);
                        continue;
                    }
                    // A shutdown command can send its response immediately
                    // before dropping the bridge. Deliver that response first.
                    let response = match reply.receiver.try_recv() {
                        Ok(response) => Some(response),
                        Err(mpsc::TryRecvError::Disconnected) => Some(AgentResponse::unavailable(
                            "application response channel closed",
                        )),
                        Err(mpsc::TryRecvError::Empty) if stopping => {
                            Some(AgentResponse::unavailable("application shut down"))
                        }
                        Err(mpsc::TryRecvError::Empty)
                            if reply.started.elapsed() >= reply.timeout =>
                        {
                            Some(AgentResponse::Error {
                                status: 504,
                                message: "timed out waiting for application response".into(),
                            })
                        }
                        Err(mpsc::TryRecvError::Empty) => None,
                    };
                    if let Some(response) = response {
                        let reply = pending.swap_remove(index);
                        let _ = reply.completion.send(response);
                    } else {
                        index += 1;
                    }
                }
                !pending.is_empty()
            };
            if stopping {
                return;
            }
            tokio::select! {
                _ = self.registered.notified() => {}
                _ = sleep(REPLY_POLL_INTERVAL), if has_pending => {}
                changed = shutdown.changed() => { closed = changed.is_err(); }
            }
        }
    }
}

pub(super) async fn receive_reply(reply: oneshot::Receiver<AgentResponse>) -> AgentResponse {
    reply
        .await
        .unwrap_or_else(|_| AgentResponse::unavailable("HTTP response relay stopped"))
}

pub(super) async fn serve(listener: TcpListener, server: Server, limits: Limits) {
    tracing::info!(addr = ?listener.local_addr(), "mikage agent HTTP API listening");
    let relay = tokio::spawn(server.replies.clone().run(server.shutdown.subscribe()));
    let router = Router::new()
        .fallback(handle_request)
        .with_state((server.clone(), limits));
    let busy = Router::new().fallback(|| async {
        HttpResponse::json_error(429, "too many connections").into_response()
    });
    let mut shutdown = server.shutdown.subscribe();
    let mut connections = JoinSet::new();
    loop {
        if *shutdown.borrow() {
            break;
        }
        tokio::select! {
            biased;
            _ = shutdown.changed() => break,
            result = connections.join_next(), if !connections.is_empty() => {
                if let Some(Err(error)) = result {
                    tracing::warn!("agent HTTP connection task failed: {error}");
                }
            }
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => {
                    if connections.len() >= server.config.max_connections {
                        // One bounded rejection at a time. Do not create an
                        // unbounded number of tasks just to reject connections.
                        let rejection = serve_connection(stream, busy.clone(), shutdown.clone(), limits, Duration::ZERO);
                        tokio::select! {
                            biased;
                            _ = shutdown.changed() => break,
                            result = timeout(limits.io_timeout.min(Duration::from_secs(1)), rejection) => {
                                if result.is_err() {
                                    tracing::debug!("agent HTTP rejection deadline exceeded");
                                }
                            }
                        }
                    } else {
                        // The slot is occupied until Hyper has flushed the body,
                        // including while a slow client applies backpressure.
                        connections.spawn(serve_connection(
                            stream, router.clone(), shutdown.clone(), limits,
                            server.config.request_timeout,
                        ));
                    }
                }
                Err(error) => {
                    tracing::warn!("agent HTTP accept failed: {error}");
                    tokio::select! {
                        _ = sleep(Duration::from_millis(100)) => {}
                        _ = shutdown.changed() => break,
                    }
                }
            }
        }
    }
    drop(listener);
    // A grace period delivers shutdown responses and lets pending handlers
    // observe fail_all. Unresponsive peers cannot hold AgentBridge::drop open.
    if timeout(limits.shutdown_timeout, async {
        while let Some(result) = connections.join_next().await {
            if let Err(error) = result {
                tracing::warn!("agent HTTP connection task failed: {error}");
            }
        }
    })
    .await
    .is_err()
    {
        tracing::warn!(
            connections = connections.len(),
            "agent HTTP shutdown deadline exceeded"
        );
        connections.abort_all();
        while connections.join_next().await.is_some() {}
    }
    if let Err(error) = relay.await {
        tracing::warn!("agent HTTP response relay failed: {error}");
    }
}

async fn serve_connection(
    stream: TcpStream,
    router: Router,
    mut shutdown: watch::Receiver<bool>,
    limits: Limits,
    application_timeout: Duration,
) {
    let ready = Arc::new(Notify::new());
    let signal = ready.clone();
    let service = router.layer(middleware::map_response(move |response: Response| {
        let signal = signal.clone();
        async move {
            signal.notify_one();
            response
        }
    }));
    let mut builder = http1::Builder::new();
    builder
        .keep_alive(false)
        .max_buf_size(MAX_HEADER_BYTES)
        .timer(TokioTimer::new())
        .header_read_timeout(limits.io_timeout);
    let connection =
        builder.serve_connection(TokioIo::new(stream), TowerToHyperService::new(service));
    tokio::pin!(connection);
    // Bound protocol-error paths too, which may finish without reaching axum.
    let lifetime = application_timeout.saturating_add(limits.io_timeout.saturating_mul(3));
    let result = timeout(lifetime, async {
        tokio::select! {
            result = &mut connection => Some(result),
            _ = ready.notified() => match timeout(limits.io_timeout, &mut connection).await {
                Ok(result) => Some(result),
                Err(_) => {
                    tracing::warn!("agent HTTP response write deadline exceeded");
                    None
                }
            },
            _ = shutdown.changed() => {
                connection.as_mut().graceful_shutdown();
                match timeout(limits.shutdown_timeout, &mut connection).await {
                    Ok(result) => Some(result),
                    Err(_) => {
                        tracing::warn!("agent HTTP connection shutdown deadline exceeded");
                        None
                    }
                }
            }
        }
    })
    .await;
    match result {
        Ok(Some(Err(error))) => tracing::warn!("agent HTTP connection failed: {error}"),
        Err(_) => tracing::warn!("agent HTTP connection deadline exceeded"),
        _ => {}
    }
}

async fn handle_request(
    State((server, limits)): State<(Server, Limits)>,
    request: Request,
) -> Response {
    let (parts, body) = request.into_parts();
    let headers = parts
        .headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.to_string(), value.to_string()))
        })
        .collect();
    if !authorized(&headers, server.config.auth_token.as_deref()) {
        return HttpResponse::json_error(401, "unauthorized").into_response();
    }
    if parts.headers.contains_key(header::TRANSFER_ENCODING) {
        return HttpResponse::json_error(400, "transfer encoding is unsupported").into_response();
    }
    if parts
        .headers
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .is_some_and(|length| length > MAX_BODY_BYTES as u64)
    {
        return HttpResponse::json_error(400, "request body exceeds 1 MiB").into_response();
    }
    let body = match timeout(limits.io_timeout, to_bytes(body, MAX_BODY_BYTES)).await {
        Ok(Ok(body)) => body,
        Ok(Err(error)) => {
            tracing::debug!("agent HTTP request body rejected: {error}");
            return HttpResponse::json_error(400, "invalid request body or body exceeds 1 MiB")
                .into_response();
        }
        Err(_) => {
            return HttpResponse::json_error(408, "request body read deadline exceeded")
                .into_response();
        }
    };
    handle_http_request(
        HttpRequest {
            method: parts.method.to_string(),
            path: parts.uri.path().to_string(),
            headers,
            body: body.to_vec(),
        },
        &server,
    )
    .await
    .into_response()
}

struct SharedBytes(Arc<Vec<u8>>);

impl AsRef<[u8]> for SharedBytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl IntoResponse for HttpResponse {
    fn into_response(self) -> Response {
        let body = bytes::Bytes::from_owner(SharedBytes(self.body));
        let mut response = Body::from(body).into_response();
        *response.status_mut() =
            StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        match self.content_type.parse() {
            Ok(content_type) => {
                response
                    .headers_mut()
                    .insert(header::CONTENT_TYPE, content_type);
                response
            }
            Err(_) => {
                HttpResponse::json_error(500, "invalid response content type").into_response()
            }
        }
    }
}

#[cfg(test)]
mod tests;
