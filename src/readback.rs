//! Bounded, reusable GPU readback without blocking a frame submission.
//!
//! Copies are recorded on the caller's encoder. Mapping is scheduled by
//! `map_buffer_on_submit`, so the caller retains control of submission order.
//! The mapping callback only sends a readiness notification. [`ReadbackRing::take_ready`]
//! copies mapped bytes and unmaps on its calling thread; call it on a worker when
//! readbacks are large. Native applications must also poll the device, and browser
//! applications must yield to the browser event loop for mapping to complete.

use std::sync::{Arc, Mutex, mpsc};

/// Identifier allocated by one readback ring.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReadbackId(pub u64);

/// Identifies the frame or simulation checkpoint from which bytes were copied.
#[derive(Clone, Debug, Default)]
pub struct ReadbackMetadata {
    pub target: String,
    pub submission_id: Option<u64>,
    pub tick_id: Option<u64>,
    pub frame_id: Option<u64>,
    /// Populated automatically for texture readback.
    pub size: Option<[u32; 2]>,
    pub texture_format: Option<wgpu::TextureFormat>,
    /// Row size in the returned, tightly packed bytes, without GPU padding.
    pub bytes_per_row: Option<u32>,
    /// Source offset for buffer readback.
    pub buffer_offset: Option<u64>,
}

/// Failure to schedule or complete a readback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadbackError {
    /// All staging slots are occupied. Try again after consuming a result.
    Busy,
    InvalidRequest(String),
    Mapping(String),
    Cancelled,
}

impl std::fmt::Display for ReadbackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy => f.write_str("all readback slots are occupied"),
            Self::InvalidRequest(message) => write!(f, "invalid readback: {message}"),
            Self::Mapping(message) => write!(f, "readback mapping failed: {message}"),
            Self::Cancelled => f.write_str("readback was cancelled"),
        }
    }
}

impl std::error::Error for ReadbackError {}

/// A completed readback. Texture bytes retain their original GPU channel order.
#[derive(Debug)]
pub struct ReadbackResult {
    pub id: ReadbackId,
    pub metadata: ReadbackMetadata,
    pub data: Result<Vec<u8>, ReadbackError>,
}

struct Pending {
    id: ReadbackId,
    metadata: ReadbackMetadata,
    length: u64,
    padded_row: Option<u32>,
    cancelled: bool,
}

#[derive(Default)]
struct Slot {
    buffer: Option<wgpu::Buffer>,
    device: Option<wgpu::Device>,
    capacity: u64,
    pending: Option<Pending>,
    reading: bool,
}

struct RingState {
    slots: Vec<Slot>,
    next_id: u64,
}

type Ready = (usize, ReadbackId, Result<(), wgpu::BufferAsyncError>);

/// A cloneable staging-buffer pool. Clones share the slots and completion queue.
///
/// The default is three slots, with a 64 MiB limit per copy. Slots are retained
/// for reuse. A slot remains busy until its mapping result has been consumed.
/// Dropping an encoder before submission does not complete its readback; call
/// [`Self::cancel_unsubmitted`] for that request before dropping the encoder.
#[derive(Clone)]
pub struct ReadbackRing {
    state: Arc<Mutex<RingState>>,
    ready_tx: mpsc::Sender<Ready>,
    ready_rx: Arc<Mutex<mpsc::Receiver<Ready>>>,
    max_bytes: u64,
}

impl Default for ReadbackRing {
    fn default() -> Self {
        Self::new(3, 64 * 1024 * 1024).expect("default ring limits are valid")
    }
}

impl ReadbackRing {
    /// Creates a ring with explicit slot and per-copy memory limits.
    pub fn new(slot_count: usize, max_bytes: u64) -> Result<Self, ReadbackError> {
        if slot_count == 0 || max_bytes == 0 || max_bytes > usize::MAX as u64 {
            return Err(ReadbackError::InvalidRequest(
                "nonzero, addressable capacity required".into(),
            ));
        }
        let (ready_tx, ready_rx) = mpsc::channel();
        Ok(Self {
            state: Arc::new(Mutex::new(RingState {
                slots: (0..slot_count).map(|_| Slot::default()).collect(),
                next_id: 1,
            })),
            ready_tx,
            ready_rx: Arc::new(Mutex::new(ready_rx)),
            max_bytes,
        })
    }

    pub fn pending_count(&self) -> usize {
        self.state
            .lock()
            .unwrap()
            .slots
            .iter()
            .filter(|slot| slot.pending.is_some() || slot.reading)
            .count()
    }

    /// Records a copy of an aligned buffer region and schedules mapping on submit.
    pub fn enqueue_buffer(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        offset: u64,
        size: u64,
        mut metadata: ReadbackMetadata,
    ) -> Result<ReadbackId, ReadbackError> {
        validate_buffer_copy(source.size(), offset, size, source.usage())?;
        metadata.buffer_offset = Some(offset);
        metadata.size = None;
        metadata.texture_format = None;
        metadata.bytes_per_row = None;
        let (slot, id, buffer) = self.reserve(device, size, metadata, None)?;
        encoder.copy_buffer_to_buffer(source, offset, &buffer, 0, size);
        self.schedule_map(encoder, slot, id, &buffer, size);
        Ok(id)
    }

    /// Copies an entire single-sampled, uncompressed 2D color texture.
    ///
    /// The source must include `COPY_SRC`. Padded GPU rows are stripped when
    /// the result is consumed. Depth, compressed and array textures are rejected.
    pub fn enqueue_texture(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
        mut metadata: ReadbackMetadata,
    ) -> Result<ReadbackId, ReadbackError> {
        if !source.usage().contains(wgpu::TextureUsages::COPY_SRC)
            || source.dimension() != wgpu::TextureDimension::D2
            || source.sample_count() != 1
            || source.depth_or_array_layers() != 1
            || source.format().is_depth_stencil_format()
            || source.format().block_dimensions() != (1, 1)
        {
            return Err(ReadbackError::InvalidRequest(
                "expected a single-sampled 2D color COPY_SRC texture".into(),
            ));
        }
        let bytes_per_pixel = source.format().block_copy_size(None).ok_or_else(|| {
            ReadbackError::InvalidRequest("texture format cannot be copied to a buffer".into())
        })?;
        let (row, padded_row, length) =
            texture_layout(source.width(), source.height(), bytes_per_pixel)?;
        metadata.size = Some([source.width(), source.height()]);
        metadata.texture_format = Some(source.format());
        metadata.bytes_per_row = Some(row);
        metadata.buffer_offset = None;
        let (slot, id, buffer) = self.reserve(device, length, metadata, Some(padded_row))?;
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
                    bytes_per_row: Some(padded_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: source.width(),
                height: source.height(),
                depth_or_array_layers: 1,
            },
        );
        self.schedule_map(encoder, slot, id, &buffer, length);
        Ok(id)
    }

    fn reserve(
        &self,
        device: &wgpu::Device,
        length: u64,
        metadata: ReadbackMetadata,
        padded_row: Option<u32>,
    ) -> Result<(usize, ReadbackId, wgpu::Buffer), ReadbackError> {
        if length > self.max_bytes || length > device.limits().max_buffer_size {
            return Err(ReadbackError::InvalidRequest(
                "copy exceeds the staging memory limit".into(),
            ));
        }
        let mut state = self.state.lock().unwrap();
        let index = state
            .slots
            .iter()
            .position(|slot| slot.pending.is_none() && !slot.reading)
            .ok_or(ReadbackError::Busy)?;
        let id = ReadbackId(state.next_id);
        state.next_id = state
            .next_id
            .checked_add(1)
            .ok_or_else(|| ReadbackError::InvalidRequest("request identifier exhausted".into()))?;
        let slot = &mut state.slots[index];
        if slot.capacity < length || slot.device.as_ref() != Some(device) {
            slot.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mikage_readback_staging"),
                size: length,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
            slot.capacity = length;
            slot.device = Some(device.clone());
        }
        slot.pending = Some(Pending {
            id,
            metadata,
            length,
            padded_row,
            cancelled: false,
        });
        Ok((index, id, slot.buffer.as_ref().unwrap().clone()))
    }

    fn schedule_map(
        &self,
        encoder: &wgpu::CommandEncoder,
        slot: usize,
        id: ReadbackId,
        buffer: &wgpu::Buffer,
        length: u64,
    ) {
        let tx = self.ready_tx.clone();
        encoder.map_buffer_on_submit(buffer, wgpu::MapMode::Read, 0..length, move |result| {
            // This callback may execute on the rendering thread: only signal.
            let _ = tx.send((slot, id, result));
        });
    }

    /// Marks a submitted request cancelled. Its slot is released after mapping
    /// completes; it is never recycled while the GPU could still be using it.
    pub fn cancel(&self, id: ReadbackId) -> bool {
        let mut state = self.state.lock().unwrap();
        for slot in &mut state.slots {
            if let Some(pending) = &mut slot.pending
                && pending.id == id
            {
                pending.cancelled = true;
                return true;
            }
        }
        false
    }

    /// Releases a copy whose encoder will **never** be submitted. The caller
    /// must drop that encoder rather than subsequently submitting it.
    pub fn cancel_unsubmitted(&self, id: ReadbackId) -> bool {
        let mut state = self.state.lock().unwrap();
        for slot in &mut state.slots {
            if slot
                .pending
                .as_ref()
                .is_some_and(|pending| pending.id == id)
            {
                // Discard the staging buffer too, so an accidentally retained
                // mapping closure cannot map a buffer reused by another copy.
                *slot = Slot::default();
                return true;
            }
        }
        false
    }

    /// Consumes ready results without waiting for the GPU.
    ///
    /// CPU byte copying and row unpacking happen here. Native integrations
    /// should call this on their worker thread, not inside a GPU callback.
    pub fn take_ready(&self) -> Vec<ReadbackResult> {
        let ready: Vec<_> = self.ready_rx.lock().unwrap().try_iter().collect();
        let mut results = Vec::with_capacity(ready.len());
        for (index, id, map_result) in ready {
            let (pending, buffer) = {
                let mut state = self.state.lock().unwrap();
                let slot = &mut state.slots[index];
                if slot.pending.as_ref().is_none_or(|pending| pending.id != id) {
                    continue;
                }
                slot.reading = true;
                (
                    slot.pending.take().unwrap(),
                    slot.buffer.as_ref().unwrap().clone(),
                )
            };
            let data = if pending.cancelled {
                Err(ReadbackError::Cancelled)
            } else {
                map_result
                    .map_err(|error| ReadbackError::Mapping(error.to_string()))
                    .and_then(|()| {
                        let view = buffer
                            .slice(0..pending.length)
                            .get_mapped_range()
                            .map_err(|error| ReadbackError::Mapping(error.to_string()))?;
                        Ok(if let Some(padded_row) = pending.padded_row {
                            unpack_rows(
                                &view,
                                pending.metadata.bytes_per_row.unwrap(),
                                padded_row,
                                pending.metadata.size.unwrap()[1],
                            )
                        } else {
                            view.to_vec()
                        })
                    })
            };
            buffer.unmap();
            self.state.lock().unwrap().slots[index].reading = false;
            results.push(ReadbackResult {
                id,
                metadata: pending.metadata,
                data,
            });
        }
        results
    }
}

fn validate_buffer_copy(
    source_size: u64,
    offset: u64,
    size: u64,
    usage: wgpu::BufferUsages,
) -> Result<(), ReadbackError> {
    if size == 0
        || !offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
        || !size.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
    {
        return Err(ReadbackError::InvalidRequest(
            "buffer range must be nonempty and four-byte aligned".into(),
        ));
    }
    if offset.checked_add(size).is_none_or(|end| end > source_size)
        || !usage.contains(wgpu::BufferUsages::COPY_SRC)
    {
        return Err(ReadbackError::InvalidRequest(
            "buffer range exceeds COPY_SRC source".into(),
        ));
    }
    Ok(())
}

fn texture_layout(
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
) -> Result<(u32, u32, u64), ReadbackError> {
    if width == 0 || height == 0 || bytes_per_pixel == 0 {
        return Err(ReadbackError::InvalidRequest(
            "texture extent must be nonzero".into(),
        ));
    }
    let row = width
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| ReadbackError::InvalidRequest("row size overflow".into()))?;
    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = row
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| ReadbackError::InvalidRequest("padded row size overflow".into()))?;
    Ok((row, padded, u64::from(padded) * u64::from(height)))
}

fn unpack_rows(data: &[u8], row: u32, padded_row: u32, height: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(row as usize * height as usize);
    for source in data.chunks_exact(padded_row as usize).take(height as usize) {
        bytes.extend_from_slice(&source[..row as usize]);
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_padding_from_each_texture_row() {
        let (row, padded, length) = texture_layout(3, 2, 4).unwrap();
        assert_eq!((row, padded, length), (12, 256, 512));
        let mut data = vec![0xff; length as usize];
        data[..12].fill(1);
        data[256..268].fill(2);
        assert_eq!(
            unpack_rows(&data, row, padded, 2),
            [vec![1; 12], vec![2; 12]].concat()
        );
    }

    #[test]
    fn rejects_invalid_buffer_regions_before_gpu_validation() {
        let usage = wgpu::BufferUsages::COPY_SRC;
        assert!(validate_buffer_copy(16, 4, 12, usage).is_ok());
        for (offset, size) in [(0, 0), (2, 4), (0, 3), (8, 12), (u64::MAX - 3, 8)] {
            assert!(validate_buffer_copy(16, offset, size, usage).is_err());
        }
        assert!(validate_buffer_copy(16, 0, 4, wgpu::BufferUsages::STORAGE).is_err());
    }

    #[test]
    fn rejects_overflow_and_zero_capacity() {
        assert!(texture_layout(u32::MAX, 1, 4).is_err());
        assert!(ReadbackRing::new(0, 4).is_err());
        assert!(ReadbackRing::new(1, 0).is_err());
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn gpu_ring_is_bounded_and_reuses_completed_slots() {
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
        let source = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("readback_test_source"),
                contents: &[1, 2, 3, 4, 5, 6, 7, 8],
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        let ring = ReadbackRing::new(2, 1024).unwrap();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let first = ring
            .enqueue_buffer(
                &gpu.device,
                &mut encoder,
                &source,
                0,
                4,
                ReadbackMetadata {
                    tick_id: Some(7),
                    ..Default::default()
                },
            )
            .unwrap();
        let second = ring
            .enqueue_buffer(
                &gpu.device,
                &mut encoder,
                &source,
                4,
                4,
                ReadbackMetadata::default(),
            )
            .unwrap();
        assert_eq!(
            ring.enqueue_buffer(
                &gpu.device,
                &mut encoder,
                &source,
                0,
                4,
                ReadbackMetadata::default()
            ),
            Err(ReadbackError::Busy)
        );
        gpu.queue.submit([encoder.finish()]);
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .unwrap();
        // A ready notification alone must not recycle the still-mapped slot.
        assert_eq!(ring.pending_count(), 2);
        let results = ring.take_ready();
        assert_eq!(results.len(), 2);
        let first_result = results.iter().find(|result| result.id == first).unwrap();
        assert_eq!(first_result.data.as_ref().unwrap(), &[1, 2, 3, 4]);
        assert_eq!(first_result.metadata.tick_id, Some(7));
        assert_eq!(
            results
                .iter()
                .find(|result| result.id == second)
                .unwrap()
                .data
                .as_ref()
                .unwrap(),
            &[5, 6, 7, 8]
        );
        assert_eq!(ring.pending_count(), 0);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let next = ring
            .enqueue_buffer(
                &gpu.device,
                &mut encoder,
                &source,
                0,
                8,
                ReadbackMetadata::default(),
            )
            .unwrap();
        assert!(next.0 > second.0);
        assert!(ring.cancel_unsubmitted(next));
        drop(encoder);
        assert_eq!(ring.pending_count(), 0);
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn gpu_texture_readback_preserves_rows_and_cancel_is_terminal() {
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
        let ring = ReadbackRing::default();
        for format in [
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Bgra8Unorm,
        ] {
            let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("readback_odd_width"),
                size: wgpu::Extent3d {
                    width: 3,
                    height: 2,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let expected: Vec<u8> = (0..24).collect();
            gpu.queue.write_texture(
                texture.as_image_copy(),
                &expected,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(12),
                    rows_per_image: None,
                },
                texture.size(),
            );
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            ring.enqueue_texture(
                &gpu.device,
                &mut encoder,
                &texture,
                ReadbackMetadata::default(),
            )
            .unwrap();
            let cancelled = ring
                .enqueue_texture(
                    &gpu.device,
                    &mut encoder,
                    &texture,
                    ReadbackMetadata::default(),
                )
                .unwrap();
            assert!(ring.cancel(cancelled));
            gpu.queue.submit([encoder.finish()]);
            gpu.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(10)),
                })
                .unwrap();
            let results = ring.take_ready();
            assert_eq!(
                results
                    .iter()
                    .find(|result| result.id != cancelled)
                    .unwrap()
                    .data
                    .as_ref()
                    .unwrap(),
                &expected
            );
            assert_eq!(
                results
                    .iter()
                    .find(|result| result.id == cancelled)
                    .unwrap()
                    .data,
                Err(ReadbackError::Cancelled)
            );
            assert_eq!(ring.pending_count(), 0);
        }
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for (format, usage) in [
            (
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureUsages::COPY_DST,
            ),
            (
                wgpu::TextureFormat::Depth32Float,
                wgpu::TextureUsages::COPY_SRC,
            ),
        ] {
            let source = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("invalid_capture_source"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            });
            assert!(matches!(
                ring.enqueue_texture(
                    &gpu.device,
                    &mut encoder,
                    &source,
                    ReadbackMetadata::default()
                ),
                Err(ReadbackError::InvalidRequest(_))
            ));
        }
        assert_eq!(ring.pending_count(), 0);
    }
}
