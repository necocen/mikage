//! Exact simulation checkpoints and optional scene rendering without a window.

use std::time::Duration;

use dpi::PhysicalSize;

use crate::app::{App, RenderTarget};
use crate::camera::InteractiveCamera;
use crate::capture::{CaptureRegistry, CaptureTarget};
use crate::context::{
    GpuContext, GpuDescriptor, GpuInitError, OffscreenTarget, RenderTargetConfig,
};
use crate::readback::{ReadbackError, ReadbackId, ReadbackMetadata, ReadbackResult, ReadbackRing};
use crate::runtime::{AppRuntime, RuntimeError, SubmissionToken};

#[derive(Debug)]
pub enum HeadlessError {
    Gpu(GpuInitError),
    Runtime(RuntimeError),
    Readback(ReadbackError),
    UnknownTarget(String),
    SceneNotRendered,
    MissingReadback,
}

impl std::fmt::Display for HeadlessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gpu(error) => error.fmt(f),
            Self::Runtime(error) => error.fmt(f),
            Self::Readback(error) => error.fmt(f),
            Self::UnknownTarget(name) => write!(f, "unknown capture target: {name}"),
            Self::SceneNotRendered => {
                f.write_str("render_once must run before capturing the scene")
            }
            Self::MissingReadback => {
                f.write_str("completed submission did not produce its readback result")
            }
        }
    }
}
impl std::error::Error for HeadlessError {}
impl From<GpuInitError> for HeadlessError {
    fn from(error: GpuInitError) -> Self {
        Self::Gpu(error)
    }
}
impl From<RuntimeError> for HeadlessError {
    fn from(error: RuntimeError) -> Self {
        Self::Runtime(error)
    }
}
impl From<ReadbackError> for HeadlessError {
    fn from(error: ReadbackError) -> Self {
        Self::Readback(error)
    }
}

/// A production app runtime with exact tick driving, lazy scene rendering, and
/// named texture/buffer capture. No window, GUI, or event loop is created.
///
/// `scene` names the most recently rendered offscreen scene. Other names come
/// from `App::capture_targets`; there is no composed `window` target here.
pub struct HeadlessHarness<A: App> {
    pub runtime: AppRuntime<A>,
    size: PhysicalSize<u32>,
    target_config: RenderTargetConfig,
    offscreen: Option<OffscreenTarget>,
    last_render: Option<SubmissionToken>,
    readback: ReadbackRing,
    pending_capture: Option<(ReadbackId, SubmissionToken)>,
}

impl<A: App> HeadlessHarness<A> {
    pub async fn new(
        descriptor: GpuDescriptor,
        target_config: RenderTargetConfig,
        size: PhysicalSize<u32>,
        mut camera: A::Camera,
        init: impl FnOnce(&GpuContext, RenderTargetConfig, PhysicalSize<u32>) -> A,
    ) -> Result<Self, HeadlessError> {
        let gpu = GpuContext::headless(descriptor).await?;
        if size.width == 0 || size.height == 0 {
            return Err(GpuInitError::InvalidSize(size).into());
        }
        let limit = gpu.device.limits().max_texture_dimension_2d;
        if size.width > limit || size.height > limit {
            return Err(GpuInitError::SizeExceedsLimit { size, limit }.into());
        }
        gpu.validate_render_target(target_config)?;
        camera.set_viewport_size(size.width, size.height);
        let app = init(&gpu, target_config, size);
        Ok(Self {
            runtime: AppRuntime::new(gpu, app, camera),
            size,
            target_config,
            offscreen: None,
            last_render: None,
            readback: ReadbackRing::new(1, 64 * 1024 * 1024)?,
            pending_capture: None,
        })
    }

    pub fn size(&self) -> PhysicalSize<u32> {
        self.size
    }
    pub fn render_target_config(&self) -> RenderTargetConfig {
        self.target_config
    }
    pub fn offscreen_target(&self) -> Option<&OffscreenTarget> {
        self.offscreen.as_ref()
    }

    /// Submit exactly N additional ticks. Completion remains an explicit fence.
    #[cfg(not(target_family = "wasm"))]
    pub fn advance_ticks(
        &mut self,
        count: u64,
        dt: Duration,
    ) -> Result<Option<SubmissionToken>, HeadlessError> {
        Ok(self.runtime.advance_ticks(count, dt)?)
    }

    pub async fn advance_ticks_async(
        &mut self,
        count: u64,
        dt: Duration,
    ) -> Result<Option<SubmissionToken>, HeadlessError> {
        Ok(self.runtime.advance_ticks_async(count, dt).await?)
    }

    /// Render the current app state once, without another simulation tick.
    /// Returns `WouldBlock` before rendering when the runtime is at capacity.
    pub fn render_once(&mut self) -> Result<SubmissionToken, HeadlessError> {
        self.runtime.poll_completions()?;
        if self.runtime.available_submission_slots() == 0 {
            return Err(RuntimeError::WouldBlock.into());
        }
        if self.offscreen.is_none() {
            self.offscreen = Some(OffscreenTarget::new(
                &self.runtime.gpu,
                self.size,
                self.target_config,
            )?);
        }
        let offscreen = self.offscreen.as_ref().unwrap();
        let token = self.runtime.render(RenderTarget {
            view: offscreen.color_view(),
            resolve_target: offscreen.resolve_target(),
            depth_view: Some(offscreen.depth_view()),
            size: offscreen.size(),
            config: offscreen.render_target_config(),
        })?;
        self.last_render = Some(token.clone());
        Ok(token)
    }

    fn begin_capture(
        &mut self,
        name: &str,
    ) -> Result<(ReadbackId, SubmissionToken), HeadlessError> {
        if name == "window" {
            return Err(HeadlessError::UnknownTarget(name.to_owned()));
        }
        let scene_endpoint = if name == "scene" {
            Some(
                self.last_render
                    .as_ref()
                    .ok_or(HeadlessError::SceneNotRendered)?
                    .clone(),
            )
        } else {
            None
        };
        let scene_texture = self
            .offscreen
            .as_ref()
            .map(|target| target.texture().clone());
        let metadata = ReadbackMetadata {
            target: name.to_owned(),
            tick_id: scene_endpoint.as_ref().and_then(|token| token.tick_id),
            frame_id: scene_endpoint.as_ref().and_then(|token| token.frame_id),
            ..Default::default()
        };
        let ring = self.readback.clone();
        let name = name.to_owned();
        let (result, token) = self.runtime.submit_command(move |app, ctx| {
            // Build after runtime completion hooks, which may replace app resources.
            let mut registry = CaptureRegistry::new();
            app.capture_targets(&mut registry);
            if let Some(texture) = scene_texture {
                registry.register_texture("scene", &texture);
            }
            let resource = registry
                .get(&name)
                .cloned()
                .ok_or_else(|| HeadlessError::UnknownTarget(name))?;
            match resource {
                CaptureTarget::Texture(texture) => {
                    ring.enqueue_texture(&ctx.gpu.device, ctx.encoder, &texture, metadata)
                }
                CaptureTarget::Buffer {
                    buffer,
                    offset,
                    size,
                } => ring.enqueue_buffer(
                    &ctx.gpu.device,
                    ctx.encoder,
                    &buffer,
                    offset,
                    size,
                    metadata,
                ),
            }
            .map_err(HeadlessError::Readback)
        })?;
        let id = result?;
        self.pending_capture = Some((id, token.clone()));
        Ok((id, token))
    }

    fn finish_capture(
        &mut self,
        id: ReadbackId,
        token: &SubmissionToken,
    ) -> Result<ReadbackResult, HeadlessError> {
        let mut result = self
            .readback
            .take_ready()
            .into_iter()
            .find(|result| result.id == id)
            .ok_or(HeadlessError::MissingReadback)?;
        result.metadata.submission_id = Some(token.id);
        if result.metadata.target != "scene" {
            result.metadata.tick_id = token.tick_id;
        }
        self.pending_capture = None;
        Ok(result)
    }

    /// Capture bytes after their copy endpoint completes. This explicit native
    /// checkpoint may wait; it never advances simulation or implicitly renders.
    #[cfg(not(target_family = "wasm"))]
    pub fn capture_named(&mut self, name: &str) -> Result<ReadbackResult, HeadlessError> {
        // Recover a slot if a previous asynchronous capture future was cancelled.
        if let Some((_, token)) = self.pending_capture.clone() {
            self.runtime.wait_for(&token)?;
            self.readback.take_ready();
            self.pending_capture = None;
        }
        let (id, token) = loop {
            match self.begin_capture(name) {
                Err(HeadlessError::Runtime(RuntimeError::WouldBlock)) => {
                    self.runtime.wait_for_capacity()?;
                }
                result => break result?,
            }
        };
        self.runtime.wait_for(&token)?;
        self.finish_capture(id, &token)
    }

    /// Async counterpart of `capture_named`, suitable for browser event loops.
    pub async fn capture_named_async(
        &mut self,
        name: &str,
    ) -> Result<ReadbackResult, HeadlessError> {
        if let Some((_, token)) = self.pending_capture.clone() {
            self.runtime.wait_for_async(&token).await?;
            self.readback.take_ready();
            self.pending_capture = None;
        }
        let (id, token) = loop {
            match self.begin_capture(name) {
                Err(HeadlessError::Runtime(RuntimeError::WouldBlock)) => {
                    self.runtime.wait_for_capacity_async().await?;
                }
                result => break result?,
            }
        };
        self.runtime.wait_for_async(&token).await?;
        self.finish_capture(id, &token)
    }
}
