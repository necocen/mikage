//! Window-independent simulation, rendering, and GPU submission ownership.

use std::collections::VecDeque;
#[cfg(not(target_family = "wasm"))]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::task::{Poll, Waker};
use std::time::Duration;

use crate::app::{
    App, CommandContext, RenderContext, RenderTarget, RenderUpdateContext, TickContext,
};
use crate::context::GpuContext;

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

/// An identity for work submitted by one runtime. Raw external queue submissions
/// are not included in runtime counters, although a fence waits for earlier work
/// on the same queue as usual.
#[derive(Debug, Clone)]
pub struct SubmissionToken {
    pub id: u64,
    pub submission_index: wgpu::SubmissionIndex,
    /// Latest simulation tick covered by this queue endpoint, including for a
    /// render or diagnostic submission that does not execute another tick.
    pub tick_id: Option<u64>,
    pub frame_id: Option<u64>,
    runtime_id: u64,
}

/// Monotonic progress. Ticks, rendered frames, and submissions are distinct units.
#[derive(Debug, Clone, Default)]
#[cfg_attr(
    all(feature = "agent", not(target_family = "wasm")),
    derive(serde::Serialize)
)]
pub struct RuntimeProgress {
    pub encoded_ticks: u64,
    pub submitted_ticks: u64,
    pub completed_ticks: u64,
    pub encoded_frames: u64,
    pub submitted_frames: u64,
    pub completed_frames: u64,
    pub submitted_submissions: u64,
    pub completed_submissions: u64,
    /// Number of calls to present, not verified scanout or GPU completion.
    pub presented_frames: u64,
    pub last_presented_frame: Option<u64>,
    pub elapsed: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct RuntimeConfig {
    pub max_in_flight_submissions: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_in_flight_submissions: 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    WouldBlock,
    Stopped,
    InvalidConfig,
    InvalidTarget,
    InvalidToken,
    DeviceLost(String),
    Poll(String),
    WorkerStopped,
    Timeout,
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WouldBlock => f.write_str("GPU submission capacity is exhausted"),
            Self::Stopped => f.write_str("the application runtime has stopped"),
            Self::InvalidConfig => f.write_str("max_in_flight_submissions must be positive"),
            Self::InvalidTarget => {
                f.write_str("render target size or MSAA resolve configuration is invalid")
            }
            Self::InvalidToken => f.write_str("submission token belongs to another runtime"),
            Self::DeviceLost(message) => write!(f, "GPU device lost: {message}"),
            Self::Poll(message) => write!(f, "GPU completion polling failed: {message}"),
            Self::WorkerStopped => f.write_str("GPU completion worker stopped"),
            Self::Timeout => f.write_str("GPU completion wait timed out"),
        }
    }
}
impl std::error::Error for RuntimeError {}

pub type RuntimeWaker = Arc<dyn Fn() + Send + Sync + 'static>;

enum Completion {
    Done(SubmissionToken),
    Failed(RuntimeError),
}

struct CompletionSignal {
    sender: mpsc::Sender<Completion>,
    host_waker: Mutex<Option<RuntimeWaker>>,
    task_waker: Mutex<Option<Waker>>,
}

impl CompletionSignal {
    fn send(&self, completion: Completion) {
        let _ = self.sender.send(completion);
        let host = self.host_waker.lock().unwrap().clone();
        let task = self.task_waker.lock().unwrap().take();
        if let Some(waker) = host {
            waker();
        }
        if let Some(waker) = task {
            waker.wake();
        }
    }
}

#[cfg(not(target_family = "wasm"))]
struct CompletionWorker {
    sender: mpsc::Sender<SubmissionToken>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

#[cfg(not(target_family = "wasm"))]
impl CompletionWorker {
    fn new(device: wgpu::Device, signal: Arc<CompletionSignal>) -> Self {
        let (sender, receiver) = mpsc::channel::<SubmissionToken>();
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = stop.clone();
        let thread = std::thread::spawn(move || {
            while !thread_stop.load(Ordering::Acquire) {
                let token = match receiver.recv_timeout(Duration::from_millis(10)) {
                    Ok(token) => token,
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                };
                loop {
                    if thread_stop.load(Ordering::Acquire) {
                        return;
                    }
                    match device.poll(wgpu::PollType::Wait {
                        submission_index: Some(token.submission_index.clone()),
                        timeout: Some(Duration::from_millis(10)),
                    }) {
                        Ok(_) => {
                            signal.send(Completion::Done(token));
                            break;
                        }
                        Err(wgpu::PollError::Timeout) => continue,
                        Err(error) => {
                            signal.send(Completion::Failed(RuntimeError::Poll(error.to_string())));
                            return;
                        }
                    }
                }
            }
        });
        Self {
            sender,
            stop,
            thread: Some(thread),
        }
    }
}

#[cfg(not(target_family = "wasm"))]
impl Drop for CompletionWorker {
    fn drop(&mut self) {
        // Bounded poll waits make shutdown independent of GPU queue progress.
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Drives the same app with a window, supplied targets, or no rendering at all.
///
/// Interactive methods never wait for the GPU. Drain completion notifications on
/// the owning thread; the worker and wgpu callbacks never call app methods.
/// `new` installs the device-lost callback on its owned GPU device.
pub struct AppRuntime<A: App> {
    pub gpu: GpuContext,
    pub app: A,
    pub camera: A::Camera,
    config: RuntimeConfig,
    progress: RuntimeProgress,
    pending: VecDeque<SubmissionToken>,
    runtime_id: u64,
    completion_receiver: mpsc::Receiver<Completion>,
    signal: Arc<CompletionSignal>,
    #[cfg(not(target_family = "wasm"))]
    worker: Option<CompletionWorker>,
    failure: Option<RuntimeError>,
    stopped: bool,
}

impl<A: App> AppRuntime<A> {
    pub fn new(gpu: GpuContext, app: A, camera: A::Camera) -> Self {
        let (sender, completion_receiver) = mpsc::channel();
        let signal = Arc::new(CompletionSignal {
            sender,
            host_waker: Mutex::new(None),
            task_waker: Mutex::new(None),
        });
        let loss_signal = signal.clone();
        gpu.device.set_device_lost_callback(move |reason, message| {
            loss_signal.send(Completion::Failed(RuntimeError::DeviceLost(format!(
                "{reason:?}: {message}"
            ))));
        });
        #[cfg(not(target_family = "wasm"))]
        let worker = Some(CompletionWorker::new(gpu.device.clone(), signal.clone()));
        Self {
            gpu,
            app,
            camera,
            config: RuntimeConfig::default(),
            progress: RuntimeProgress::default(),
            pending: VecDeque::new(),
            runtime_id: NEXT_RUNTIME_ID.fetch_add(1, Ordering::Relaxed),
            completion_receiver,
            signal,
            #[cfg(not(target_family = "wasm"))]
            worker,
            failure: None,
            stopped: false,
        }
    }

    pub fn config(&self) -> RuntimeConfig {
        self.config
    }
    pub fn set_config(&mut self, config: RuntimeConfig) -> Result<(), RuntimeError> {
        if config.max_in_flight_submissions == 0 {
            return Err(RuntimeError::InvalidConfig);
        }
        self.config = config;
        Ok(())
    }
    pub fn progress(&self) -> &RuntimeProgress {
        &self.progress
    }
    pub fn in_flight_submissions(&self) -> usize {
        self.pending.len()
    }
    pub fn available_submission_slots(&self) -> usize {
        self.config
            .max_in_flight_submissions
            .saturating_sub(self.pending.len())
    }
    pub fn set_waker(&mut self, waker: RuntimeWaker) {
        *self.signal.host_waker.lock().unwrap() = Some(waker);
    }

    /// Explicit native backpressure wait for checkpoint/offscreen hosts.
    #[cfg(not(target_family = "wasm"))]
    pub fn wait_for_capacity(&mut self) -> Result<(), RuntimeError> {
        self.poll_completions()?;
        while self.available_submission_slots() == 0 {
            let oldest = self.pending.front().unwrap().clone();
            self.wait_for(&oldest)?;
        }
        Ok(())
    }

    pub async fn wait_for_capacity_async(&mut self) -> Result<(), RuntimeError> {
        self.poll_completions()?;
        while self.available_submission_slots() == 0 {
            let oldest = self.pending.front().unwrap().clone();
            self.wait_for_async(&oldest).await?;
        }
        Ok(())
    }

    fn handle_completion(&mut self, completion: Completion) {
        match completion {
            Completion::Done(token) => {
                if self.failure.is_some() {
                    return;
                }
                // Completing a queue endpoint also completes every earlier entry.
                while self
                    .pending
                    .front()
                    .is_some_and(|front| front.id <= token.id)
                {
                    let done = self.pending.pop_front().unwrap();
                    self.progress.completed_submissions = done.id;
                    if let Some(tick) = done.tick_id {
                        self.progress.completed_ticks = tick;
                    }
                    if let Some(frame) = done.frame_id {
                        self.progress.completed_frames = frame;
                    }
                    self.app.after_complete(&self.gpu, &done);
                }
            }
            Completion::Failed(error) => {
                self.failure.get_or_insert(error);
            }
        }
    }

    pub fn poll_completions(&mut self) -> Result<usize, RuntimeError> {
        if self.stopped {
            return Err(RuntimeError::Stopped);
        }
        let before = self.progress.completed_submissions;
        while let Ok(completion) = self.completion_receiver.try_recv() {
            self.handle_completion(completion);
        }
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        Ok((self.progress.completed_submissions - before) as usize)
    }

    fn ensure_capacity(&mut self) -> Result<(), RuntimeError> {
        self.poll_completions()?;
        if self.available_submission_slots() == 0 {
            return Err(RuntimeError::WouldBlock);
        }
        Ok(())
    }

    fn submit(
        &mut self,
        encoder: wgpu::CommandEncoder,
        mut extra: Vec<wgpu::CommandBuffer>,
        tick_id: Option<u64>,
        frame_id: Option<u64>,
    ) -> Result<SubmissionToken, RuntimeError> {
        extra.push(encoder.finish());
        let submission_index = self.gpu.queue.submit(extra);
        let token = SubmissionToken {
            id: self.progress.submitted_submissions + 1,
            submission_index,
            tick_id,
            frame_id,
            runtime_id: self.runtime_id,
        };
        self.progress.submitted_submissions = token.id;
        if let Some(tick) = tick_id {
            self.progress.submitted_ticks = tick;
        }
        if let Some(frame) = frame_id {
            self.progress.submitted_frames = frame;
        }
        self.pending.push_back(token.clone());
        #[cfg(not(target_family = "wasm"))]
        if self
            .worker
            .as_ref()
            .unwrap()
            .sender
            .send(token.clone())
            .is_err()
        {
            self.failure = Some(RuntimeError::WorkerStopped);
            return Err(RuntimeError::WorkerStopped);
        }
        #[cfg(target_family = "wasm")]
        {
            let signal = self.signal.clone();
            let completion_token = token.clone();
            self.gpu
                .queue
                .on_submitted_work_done(move || signal.send(Completion::Done(completion_token)));
        }
        self.app.after_submit(&self.gpu, &token);
        Ok(token)
    }

    /// Execute and submit one tick, or return `WouldBlock` before changing the app.
    pub fn try_tick(&mut self, dt: Duration) -> Result<SubmissionToken, RuntimeError> {
        self.ensure_capacity()?;
        let tick_id = self.progress.encoded_ticks + 1;
        let elapsed = self.progress.elapsed + dt.as_secs_f64();
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mikage_tick"),
            });
        self.app.tick(&mut TickContext {
            gpu: &self.gpu,
            encoder: &mut encoder,
            tick_id,
            dt: dt.as_secs_f32(),
            elapsed,
        });
        self.progress.encoded_ticks = tick_id;
        self.progress.elapsed = elapsed;
        self.submit(encoder, Vec::new(), Some(tick_id), None)
    }

    /// Render without advancing simulation. Compose overlays/capture copies after
    /// the scene in the same encoder. GUI construction may run before this call.
    pub fn render_with(
        &mut self,
        target: RenderTarget<'_>,
        wall_dt: Duration,
        compose: impl FnOnce(&mut A, &mut RenderContext<'_, A::Camera>),
    ) -> Result<SubmissionToken, RuntimeError> {
        self.ensure_capacity()?;
        if target.size.width == 0
            || target.size.height == 0
            || !matches!(target.config.sample_count, 1 | 4)
            || (target.config.sample_count > 1) != target.resolve_target.is_some()
        {
            return Err(RuntimeError::InvalidTarget);
        }
        let frame_id = self.progress.encoded_frames + 1;
        self.app.prepare_render(&mut RenderUpdateContext {
            gpu: &self.gpu,
            camera: &self.camera,
            target_size: target.size,
            target_config: target.config,
            dt: wall_dt.as_secs_f32(),
            elapsed: self.progress.elapsed,
        });
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mikage_render"),
            });
        let mut extra_command_buffers = Vec::new();
        {
            let mut ctx = RenderContext {
                gpu: &self.gpu,
                encoder: &mut encoder,
                extra_command_buffers: &mut extra_command_buffers,
                target,
                camera: &self.camera,
                frame_id,
                completed_tick: self.progress.completed_ticks,
            };
            self.app.render(&mut ctx);
            compose(&mut self.app, &mut ctx);
        }
        self.progress.encoded_frames = frame_id;
        let tick_id = (self.progress.submitted_ticks > 0).then_some(self.progress.submitted_ticks);
        self.submit(encoder, extra_command_buffers, tick_id, Some(frame_id))
    }

    pub fn render(&mut self, target: RenderTarget<'_>) -> Result<SubmissionToken, RuntimeError> {
        self.render_with(target, Duration::ZERO, |_, _| {})
    }

    /// Submit diagnostic work at the current simulation boundary. The closure
    /// runs only after capacity is available, and does not increment tick/frame ids.
    pub fn submit_command<R>(
        &mut self,
        encode: impl FnOnce(&mut A, &mut CommandContext<'_>) -> R,
    ) -> Result<(R, SubmissionToken), RuntimeError> {
        self.ensure_capacity()?;
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mikage_command"),
            });
        let result = encode(
            &mut self.app,
            &mut CommandContext {
                gpu: &self.gpu,
                encoder: &mut encoder,
                tick_id: self.progress.submitted_ticks,
            },
        );
        let tick_id = (self.progress.submitted_ticks > 0).then_some(self.progress.submitted_ticks);
        let token = self.submit(encoder, Vec::new(), tick_id, None)?;
        Ok((result, token))
    }

    /// Record a successful host `present` call. Offscreen rendering must not call
    /// this. Repeated notification for the same frame has no effect.
    pub fn mark_presented(&mut self, token: &SubmissionToken) {
        if token.runtime_id != self.runtime_id {
            return;
        }
        if let Some(frame) = token.frame_id
            && self
                .progress
                .last_presented_frame
                .is_none_or(|previous| frame > previous)
        {
            self.progress.last_presented_frame = Some(frame);
            self.progress.presented_frames += 1;
        }
    }

    fn check_token(&self, token: &SubmissionToken) -> Result<(), RuntimeError> {
        if token.runtime_id != self.runtime_id {
            return Err(RuntimeError::InvalidToken);
        }
        Ok(())
    }

    /// Explicit native completion wait. This does not execute ticks or rendering.
    #[cfg(not(target_family = "wasm"))]
    pub fn wait_for(&mut self, token: &SubmissionToken) -> Result<(), RuntimeError> {
        self.wait_for_timeout(token, None)
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn wait_for_timeout(
        &mut self,
        token: &SubmissionToken,
        timeout: Option<Duration>,
    ) -> Result<(), RuntimeError> {
        self.check_token(token)?;
        let start = std::time::Instant::now();
        loop {
            self.poll_completions()?;
            if self.progress.completed_submissions >= token.id {
                return Ok(());
            }
            let remaining = match timeout {
                Some(timeout) => timeout
                    .checked_sub(start.elapsed())
                    .ok_or(RuntimeError::Timeout)?,
                None => Duration::from_millis(100),
            };
            match self
                .completion_receiver
                .recv_timeout(remaining.min(Duration::from_millis(100)))
            {
                Ok(completion) => self.handle_completion(completion),
                Err(mpsc::RecvTimeoutError::Timeout) => (),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(RuntimeError::WorkerStopped);
                }
            }
        }
    }

    /// Nonblocking async completion wait, including browser/WebGPU execution.
    pub async fn wait_for_async(&mut self, token: &SubmissionToken) -> Result<(), RuntimeError> {
        self.check_token(token)?;
        std::future::poll_fn(|context| {
            // Register before draining to avoid losing a concurrent completion wake.
            *self.signal.task_waker.lock().unwrap() = Some(context.waker().clone());
            if let Err(error) = self.poll_completions() {
                self.signal.task_waker.lock().unwrap().take();
                return Poll::Ready(Err(error));
            }
            if self.progress.completed_submissions >= token.id {
                self.signal.task_waker.lock().unwrap().take();
                Poll::Ready(Ok(()))
            } else {
                Poll::Pending
            }
        })
        .await
    }

    /// Submit exactly `count` additional ticks. Waits only when capacity is full;
    /// call `wait_for` on the returned endpoint to require all ticks completed.
    #[cfg(not(target_family = "wasm"))]
    pub fn advance_ticks(
        &mut self,
        count: u64,
        dt: Duration,
    ) -> Result<Option<SubmissionToken>, RuntimeError> {
        let mut last = None;
        for _ in 0..count {
            loop {
                match self.try_tick(dt) {
                    Ok(token) => {
                        last = Some(token);
                        break;
                    }
                    Err(RuntimeError::WouldBlock) => {
                        let oldest = self.pending.front().unwrap().clone();
                        self.wait_for(&oldest)?;
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        Ok(last)
    }

    pub async fn advance_ticks_async(
        &mut self,
        count: u64,
        dt: Duration,
    ) -> Result<Option<SubmissionToken>, RuntimeError> {
        let mut last = None;
        for _ in 0..count {
            loop {
                match self.try_tick(dt) {
                    Ok(token) => {
                        last = Some(token);
                        break;
                    }
                    Err(RuntimeError::WouldBlock) => {
                        let oldest = self.pending.front().unwrap().clone();
                        self.wait_for_async(&oldest).await?;
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        Ok(last)
    }

    /// Stop app-owned workers and the completion worker exactly once. Pending GPU
    /// work is not synchronously drained; request an explicit fence before shutdown
    /// when its results are required.
    pub fn shutdown(&mut self) {
        if self.stopped {
            return;
        }
        self.stopped = true;
        self.app.shutdown(&self.gpu);
        #[cfg(not(target_family = "wasm"))]
        self.worker.take();
        self.signal.host_waker.lock().unwrap().take();
        self.signal.task_waker.lock().unwrap().take();
        self.pending.clear();
    }
}

impl<A: App> Drop for AppRuntime<A> {
    fn drop(&mut self) {
        self.shutdown();
    }
}
