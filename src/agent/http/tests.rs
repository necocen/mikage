use super::*;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream as StdStream;

fn start(config: AgentConfig) -> AgentBridge {
    AgentBridge::start(config.with_bind_addr("127.0.0.1:0".parse().unwrap()), || {}).unwrap()
}

fn connect(bridge: &AgentBridge) -> StdStream {
    let addr = bridge.snapshot.lock().unwrap().bind_addr.clone();
    let stream = StdStream::connect(addr).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    stream
}

fn read_headers(reader: &mut BufReader<StdStream>) -> (u16, HashMap<String, String>) {
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let status = line
        .split_whitespace()
        .nth(1)
        .expect("HTTP status")
        .parse()
        .unwrap();
    let mut headers = HashMap::new();
    loop {
        line.clear();
        reader.read_line(&mut line).unwrap();
        if line == "\r\n" {
            break;
        }
        assert!(!line.is_empty(), "connection ended during headers");
        let (name, value) = line.trim_end().split_once(':').unwrap();
        headers.insert(name.to_ascii_lowercase(), value.trim().to_string());
    }
    (status, headers)
}

fn read_response(stream: StdStream) -> (u16, HashMap<String, String>, Vec<u8>) {
    let mut reader = BufReader::new(stream);
    let (status, headers) = read_headers(&mut reader);
    let mut body = Vec::new();
    reader.read_to_end(&mut body).unwrap();
    let length: usize = headers
        .get("content-length")
        .map(|s| s.parse().unwrap())
        .unwrap_or(0);
    assert_eq!(body.len(), length, "short or incorrectly framed HTTP body");
    (status, headers, body)
}

fn get(bridge: &AgentBridge, path: &str) -> (u16, HashMap<String, String>, Vec<u8>) {
    let mut stream = connect(bridge);
    write!(stream, "GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
    read_response(stream)
}

fn completed(bridge: &AgentBridge, bytes: Arc<Vec<u8>>) -> JobId {
    let id = bridge
        .jobs
        .create()
        .unwrap_or_else(|_| panic!("job reservation"));
    bridge.complete_job(
        id,
        AgentResponse::Bytes {
            bytes,
            content_type: "application/octet-stream".into(),
            metadata: json!({"encoding":"raw"}),
        },
    );
    id
}

fn until(mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while !condition() {
        assert!(Instant::now() < deadline, "condition did not become true");
        thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn large_result_is_complete_under_backpressure_and_repeatable() {
    let bridge = start(AgentConfig::default());
    let bytes = Arc::new(
        (0..2 * 1024 * 1024 + 137)
            .map(|i| (i % 251) as u8)
            .collect::<Vec<_>>(),
    );
    let id = completed(&bridge, bytes.clone());
    let path = format!("/jobs/{id}/result");
    let mut stream = connect(&bridge);
    socket2::SockRef::from(&stream)
        .set_recv_buffer_size(16 * 1024)
        .unwrap();
    write!(stream, "GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
    let mut reader = BufReader::new(stream);
    let (status, headers) = read_headers(&mut reader);
    assert_eq!(status, 200);
    assert_eq!(headers["content-length"], bytes.len().to_string());
    assert_eq!(headers["content-type"], "application/octet-stream");
    thread::sleep(Duration::from_millis(100));
    let mut actual = Vec::new();
    let mut chunk = [0; 16 * 1024];
    loop {
        let size = reader.read(&mut chunk).unwrap();
        if size == 0 {
            break;
        }
        actual.extend_from_slice(&chunk[..size]);
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(&actual, &*bytes);
    assert_eq!(get(&bridge, &path).2, *bytes);
    assert_eq!(bridge.jobs.status(id).unwrap()["state"], "completed");
}

#[test]
fn fragmented_authenticated_request_reaches_host_and_returns_job_result() {
    let mut bridge = start(AgentConfig::default().with_auth_token("secret"));
    let mut stream = connect(&bridge);
    let body = br#"{"target":"sample","format":"raw","exact":true}"#;
    let request = format!(
        "POST /captures?trace=1 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        std::str::from_utf8(body).unwrap(),
    );
    // Include boundaries in both the headers and JSON body.
    for chunk in request.as_bytes().chunks(17) {
        stream.write_all(chunk).unwrap();
        thread::sleep(Duration::from_millis(5));
    }
    let (status, _, response) = read_response(stream);
    assert_eq!(status, 202);
    let id = serde_json::from_slice::<Value>(&response).unwrap()["id"]
        .as_u64()
        .unwrap();
    let request = bridge.drain_requests().pop().expect("capture queued");
    assert_eq!(request.job_id, Some(id));
    assert!(matches!(
        request.kind,
        AgentRequestKind::Capture(CaptureRequest { exact: true, .. })
    ));
    request
        .respond_to
        .send(AgentResponse::Png(vec![1, 2, 3, 4]))
        .ok();
    until(|| bridge.jobs.status(id).unwrap()["state"] == "completed");
    let mut stream = connect(&bridge);
    write!(
        stream,
        "GET /jobs/{id}/result HTTP/1.1\r\nHost: localhost\r\nX-Mikage-Token: secret\r\n\r\n"
    )
    .unwrap();
    let (status, headers, body) = read_response(stream);
    assert_eq!(status, 200);
    assert_eq!(headers["content-type"], "image/png");
    assert_eq!(body, [1, 2, 3, 4]);
}

#[test]
fn authentication_and_request_limits_are_enforced_before_dispatch() {
    let bridge = start(AgentConfig::default().with_auth_token("secret"));
    assert_eq!(get(&bridge, "/status").0, 401);
    for headers in [
        "Content-Length: 1048577\r\n",
        "Transfer-Encoding: chunked\r\n",
    ] {
        let mut stream = connect(&bridge);
        write!(
            stream,
            "POST /command HTTP/1.1\r\nHost: localhost\r\nX-Mikage-Token: secret\r\n{headers}\r\n"
        )
        .unwrap();
        assert_eq!(read_response(stream).0, 400);
    }
    assert!(bridge.requests.try_recv().is_err());
}

#[test]
fn connection_limit_covers_body_transfer_and_disconnect_releases_slot() {
    let bridge = start(AgentConfig {
        max_connections: 1,
        ..Default::default()
    });
    let id = completed(&bridge, Arc::new(vec![7; 16 * 1024 * 1024]));
    let mut stream = connect(&bridge);
    socket2::SockRef::from(&stream)
        .set_recv_buffer_size(4096)
        .unwrap();
    write!(
        stream,
        "GET /jobs/{id}/result HTTP/1.1\r\nHost: localhost\r\n\r\n"
    )
    .unwrap();
    let mut reader = BufReader::new(stream);
    assert_eq!(read_headers(&mut reader).0, 200);
    assert_eq!(get(&bridge, "/status").0, 429);
    drop(reader);
    until(|| get(&bridge, "/status").0 == 200);
    // The disconnect affects the download, not the retained capture result.
    assert_eq!(bridge.jobs.status(id).unwrap()["state"], "completed");
}

#[test]
fn pending_commands_do_not_block_status_and_timeout_releases_relay_slot() {
    let mut bridge = start(AgentConfig {
        request_timeout: Duration::from_millis(150),
        max_connections: 2,
        ..Default::default()
    });
    let mut stream = connect(&bridge);
    let body = r#"{"op":"redraw"}"#;
    write!(
        stream,
        "POST /command HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut request = None;
    until(|| {
        request = bridge.drain_requests().pop();
        request.is_some()
    });
    assert_eq!(get(&bridge, "/status").0, 200);
    assert_eq!(read_response(stream).0, 504);
    // The timed-out receiver has been removed; late host responses are harmless.
    assert!(
        request
            .unwrap()
            .respond_to
            .send(AgentResponse::ok())
            .is_err()
    );
}

#[test]
fn dropping_bridge_fails_pending_command_without_waiting_for_host() {
    let mut bridge = start(AgentConfig {
        request_timeout: Duration::from_secs(60),
        ..Default::default()
    });
    let mut stream = connect(&bridge);
    let body = r#"{"op":"redraw"}"#;
    write!(
        stream,
        "POST /command HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut request = None;
    until(|| {
        request = bridge.drain_requests().pop();
        request.is_some()
    });
    let started = Instant::now();
    drop(bridge);
    assert!(started.elapsed() < Duration::from_secs(2));
    assert_eq!(read_response(stream).0, 503);
    drop(request);
}

#[test]
fn shutdown_acknowledgement_is_delivered_when_bridge_drops_immediately() {
    let mut bridge = start(AgentConfig::default());
    let mut stream = connect(&bridge);
    let body = r#"{"op":"shutdown"}"#;
    write!(
        stream,
        "POST /command HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut request = None;
    until(|| {
        request = bridge.drain_requests().pop();
        request.is_some()
    });
    request.unwrap().respond_to.send(AgentResponse::ok()).ok();
    drop(bridge);
    let (status, _, body) = read_response(stream);
    assert_eq!(status, 200);
    assert_eq!(serde_json::from_slice::<Value>(&body).unwrap()["ok"], true);
}

#[derive(Clone, Default)]
struct Logs(Arc<Mutex<Vec<u8>>>);

impl Write for Logs {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn stalled_upload_and_download_have_deadlines_and_transport_errors_are_logged() {
    let logs = Logs::default();
    let writer = logs.clone();
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_max_level(tracing::Level::WARN)
        .with_writer(move || writer.clone())
        .finish();
    let _logging = tracing::subscriber::set_default(subscriber);
    let bridge = AgentBridge::start_with_limits(
        AgentConfig::default().with_bind_addr("127.0.0.1:0".parse().unwrap()),
        || {},
        Limits {
            io_timeout: Duration::from_millis(150),
            shutdown_timeout: Duration::from_millis(150),
        },
    )
    .unwrap();
    let mut upload = connect(&bridge);
    upload
        .write_all(b"POST /command HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\n{")
        .unwrap();
    assert_eq!(read_response(upload).0, 408);
    let id = completed(&bridge, Arc::new(vec![0; 16 * 1024 * 1024]));
    let mut download = connect(&bridge);
    socket2::SockRef::from(&download)
        .set_recv_buffer_size(4096)
        .unwrap();
    write!(
        download,
        "GET /jobs/{id}/result HTTP/1.1\r\nHost: localhost\r\n\r\n"
    )
    .unwrap();
    let mut reader = BufReader::new(download);
    assert_eq!(read_headers(&mut reader).0, 200);
    until(|| {
        String::from_utf8_lossy(&logs.0.lock().unwrap())
            .contains("response write deadline exceeded")
    });
    assert_eq!(bridge.jobs.status(id).unwrap()["state"], "completed");
    assert_eq!(get(&bridge, "/status").0, 200);
}

#[test]
fn header_timeout_and_shutdown_with_incomplete_headers_are_bounded() {
    let limits = Limits {
        io_timeout: Duration::from_millis(100),
        shutdown_timeout: Duration::from_millis(100),
    };
    let bridge = AgentBridge::start_with_limits(
        AgentConfig::default().with_bind_addr("127.0.0.1:0".parse().unwrap()),
        || {},
        limits,
    )
    .unwrap();
    let mut stream = connect(&bridge);
    stream.write_all(b"GET /status HTTP/1.1\r\nHo").unwrap();
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).unwrap();
    assert!(reply.is_empty() || String::from_utf8_lossy(&reply).contains("408"));
    let mut stream = connect(&bridge);
    stream.write_all(b"GET /status").unwrap();
    let started = Instant::now();
    drop(bridge);
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[tokio::test(flavor = "current_thread")]
async fn relay_capacity_cancellation_and_disconnection_are_bounded() {
    let relay = Arc::new(ResponseRelay::new(1));
    let (shutdown, stopped) = watch::channel(false);
    let worker = tokio::spawn(relay.clone().run(stopped));
    let (sender, receiver) = mpsc::channel();
    let reply = relay
        .register(receiver, Duration::from_secs(60))
        .unwrap_or_else(|_| panic!("slot"));
    let (_, receiver) = mpsc::channel();
    assert!(matches!(
        relay.register(receiver, Duration::from_secs(1)),
        Err(AgentResponse::Error { status: 429, .. })
    ));
    drop(reply);
    let (sender2, receiver) = mpsc::channel();
    let reply = relay
        .register(receiver, Duration::from_secs(1))
        .unwrap_or_else(|_| panic!("cancelled slot"));
    assert!(sender.send(AgentResponse::ok()).is_err());
    drop(sender2);
    assert!(matches!(
        receive_reply(reply).await,
        AgentResponse::Error { status: 503, .. }
    ));
    assert!(relay.pending.lock().unwrap().is_empty());
    shutdown.send_replace(true);
    worker.await.unwrap();
}
