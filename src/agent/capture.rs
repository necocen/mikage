//! Native readback and PNG/raw encoding worker for the agent HTTP API.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Duration;

use image::ImageEncoder;
use serde_json::json;

use super::{AgentResponse, CaptureFormat, CaptureRequest};

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

#[cfg(test)]
mod tests {
    use super::*;

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
