//! HTTP requests are interpreted on the host thread; GPU copies keep runtime ordering.
use super::*;
use crate::agent::{
    AgentBridge, AgentCaptureWorker, AgentCommand, AgentRequestKind, AgentResponse, CaptureRequest,
    JobId,
};
use crate::{CaptureRegistry, CaptureTarget, ReadbackMetadata, RuntimeProgress, SubmissionToken};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::mpsc::Sender;

struct PendingCapture {
    request: CaptureRequest,
    at_tick: u64,
    respond: Sender<AgentResponse>,
    job: Option<JobId>,
}
struct PendingGpu {
    payload: Value,
    at_tick: u64,
    respond: Sender<AgentResponse>,
    job: Option<JobId>,
}
struct PendingFence {
    token: SubmissionToken,
    result: Value,
    app_result: bool,
    respond: Sender<AgentResponse>,
    job: Option<JobId>,
}
struct RunUntil {
    target: u64,
    dt: Duration,
    respond: Sender<AgentResponse>,
    job: Option<JobId>,
    last: Option<SubmissionToken>,
}

pub(super) struct Diagnostics {
    worker: AgentCaptureWorker,
    captures: VecDeque<PendingCapture>,
    commands: VecDeque<PendingGpu>,
    fences: Vec<PendingFence>,
    run_until: Option<RunUntil>,
    paused: bool,
    exact_fence: Option<SubmissionToken>,
    exact_in_frame: bool,
}
impl Diagnostics {
    pub(super) fn new() -> Self {
        Self {
            worker: AgentCaptureWorker::new(),
            captures: VecDeque::new(),
            commands: VecDeque::new(),
            fences: Vec::new(),
            run_until: None,
            paused: false,
            exact_fence: None,
            exact_in_frame: false,
        }
    }
    pub(super) fn permits_automatic_tick(&self, current: u64) -> bool {
        !self.paused && self.run_until.is_none() && !self.boundary_blocked(current)
    }
    fn boundary_blocked(&self, current: u64) -> bool {
        self.exact_fence.is_some()
            || self.captures.iter().any(|p| p.at_tick <= current)
            || self.commands.iter().any(|p| p.at_tick <= current)
    }
    pub(super) fn requires_frame<A: App>(&self, runtime: &AppRuntime<A>) -> bool {
        self.captures.iter().any(|p| {
            p.at_tick <= runtime.progress().submitted_ticks
                && matches!(p.request.target.as_str(), "scene" | "window")
        })
    }
    pub(super) fn encode_frame_captures<C: crate::Camera>(
        &mut self,
        name: &str,
        ctx: &mut crate::RenderContext<'_, C>,
        texture: &wgpu::Texture,
        progress: &RuntimeProgress,
    ) {
        let mut remaining = VecDeque::new();
        while let Some(pending) = self.captures.pop_front() {
            if pending.request.target != name || pending.at_tick > progress.submitted_ticks {
                remaining.push_back(pending);
                continue;
            }
            let metadata = ReadbackMetadata {
                target: name.into(),
                submission_id: Some(progress.submitted_submissions + 1),
                tick_id: Some(progress.submitted_ticks),
                frame_id: Some(ctx.frame_id),
                ..Default::default()
            };
            let exact = pending.request.exact;
            let respond = pending.respond.clone();
            match self.worker.enqueue(
                &ctx.gpu.device,
                ctx.encoder,
                &CaptureTarget::Texture(texture.clone()),
                pending.request,
                metadata,
                pending.respond,
            ) {
                Ok(_) => {
                    self.exact_in_frame |= exact;
                }
                Err(error) => {
                    let _ = respond.send(error);
                }
            }
        }
        self.captures = remaining;
    }
    pub(super) fn after_frame_submit(&mut self, token: &SubmissionToken) {
        if self.exact_in_frame {
            self.exact_fence = Some(token.clone());
            self.exact_in_frame = false;
        }
    }
}

pub(super) fn pump<A: App>(
    bridge: &mut AgentBridge,
    state: &mut RunState<A>,
    event_loop: &ActiveEventLoop,
) {
    if let Err(err) = state.runtime.poll_completions() {
        bridge.fail_all(&err.to_string());
        event_loop.exit();
        return;
    }
    let current = state.runtime.progress().submitted_ticks;
    for request in bridge.drain_requests() {
        let respond = request.respond_to;
        let job = request.job_id;
        match request.kind {
            AgentRequestKind::Screenshot => {
                enqueue_capture(state, CaptureRequest::default(), respond, job)
            }
            AgentRequestKind::Capture(capture) => enqueue_capture(state, capture, respond, job),
            AgentRequestKind::Command(command) => match command {
                AgentCommand::Shutdown => {
                    let _ = respond.send(AgentResponse::ok());
                    event_loop.exit();
                    return;
                }
                AgentCommand::Redraw => {
                    state.redraw_pending = true;
                    let _ = respond.send(AgentResponse::ok());
                }
                AgentCommand::Pause => {
                    state.diagnostics.paused = true;
                    let _ = respond.send(AgentResponse::ok());
                }
                AgentCommand::Resume => {
                    if state.diagnostics.run_until.is_some() {
                        let _ = respond
                            .send(AgentResponse::busy("a run_until_completed job is active"));
                    } else {
                        state.diagnostics.paused = false;
                        state.backlog = Duration::ZERO;
                        state.last_render = Instant::now();
                        state.last_schedule = Instant::now();
                        state.redraw_pending = true;
                        let _ = respond.send(AgentResponse::ok());
                    }
                }
                AgentCommand::AppCommand { payload } => {
                    let result = state
                        .runtime
                        .app
                        .on_agent_command(payload)
                        .map(AgentResponse::json)
                        .unwrap_or_else(AgentResponse::bad_request);
                    let _ = respond.send(result);
                    state.redraw_pending = true;
                }
                AgentCommand::AppGpuCommand { payload, at_tick } => {
                    let at_tick = at_tick.unwrap_or(current);
                    if at_tick < current {
                        let _ = respond.send(AgentResponse::bad_request(
                            "tick is already past; historical GPU state is not retained",
                        ));
                    } else {
                        state.diagnostics.commands.push_back(PendingGpu {
                            payload,
                            at_tick,
                            respond,
                            job,
                        });
                    }
                }
                AgentCommand::RunUntilCompleted { target_tick, dt } => {
                    if !dt.is_finite() || dt <= 0.0 {
                        let _ = respond
                            .send(AgentResponse::bad_request("dt must be finite and positive"));
                    } else if target_tick < current {
                        let _ =
                            respond.send(AgentResponse::bad_request("target_tick is already past"));
                    } else if state.diagnostics.run_until.is_some() {
                        let _ = respond.send(AgentResponse::busy(
                            "a run_until_completed job is already active",
                        ));
                    } else if let Ok(dt) = Duration::try_from_secs_f32(dt) {
                        state.diagnostics.run_until = Some(RunUntil {
                            target: target_tick,
                            dt,
                            respond,
                            job,
                            last: None,
                        });
                    } else {
                        let _ = respond.send(AgentResponse::bad_request(
                            "dt is outside the supported duration range",
                        ));
                    }
                }
                command => {
                    let result = state
                        .runtime
                        .camera
                        .apply_agent_command(&command)
                        .map(|_| AgentResponse::ok())
                        .unwrap_or_else(AgentResponse::bad_request);
                    let _ = respond.send(result);
                    state.redraw_pending = true;
                }
            },
        }
    }
    let diagnostics = &mut state.diagnostics;
    if state
        .surface
        .as_ref()
        .expect("active surface")
        .is_suspended()
        || state.occluded
        || state.suspended
    {
        diagnostics.captures.retain(|pending| {
            if matches!(pending.request.target.as_str(), "window" | "scene") {
                let _ = pending.respond.send(AgentResponse::unavailable(
                    "presentation suspended before the requested capture",
                ));
                false
            } else {
                true
            }
        });
    }
    diagnostics
        .captures
        .retain(|p| p.job.is_none_or(|id| bridge.is_job_pending(id)));
    diagnostics
        .commands
        .retain(|p| p.job.is_none_or(|id| bridge.is_job_pending(id)));
    diagnostics
        .fences
        .retain(|p| p.app_result || p.job.is_none_or(|id| bridge.is_job_pending(id)));
    if diagnostics
        .run_until
        .as_ref()
        .is_some_and(|p| p.job.is_some_and(|id| !bridge.is_job_pending(id)))
    {
        diagnostics.run_until = None;
    }
    if diagnostics
        .exact_fence
        .as_ref()
        .is_some_and(|t| t.id <= state.runtime.progress().completed_submissions)
    {
        diagnostics.exact_fence = None;
    }
    let mut pending_fences = Vec::new();
    for fence in diagnostics.fences.drain(..) {
        if fence.token.id > state.runtime.progress().completed_submissions {
            pending_fences.push(fence);
            continue;
        }
        let result = if fence.app_result {
            state
                .runtime
                .app
                .complete_agent_command(&state.runtime.gpu, fence.result, &fence.token)
        } else {
            Ok(fence.result)
        };
        let response = result.map(|value| AgentResponse::json(json!({"result":value,"submission_id":fence.token.id,"tick_id":fence.token.tick_id,"frame_id":fence.token.frame_id}))).unwrap_or_else(AgentResponse::bad_request);
        let _ = fence.respond.send(response);
    }
    diagnostics.fences = pending_fences;
    for _ in 0..8 {
        process_boundaries(&mut state.runtime, diagnostics);
        if diagnostics.requires_frame(&state.runtime) {
            state.redraw_pending = true;
        }
        let Some(run) = &mut diagnostics.run_until else {
            break;
        };
        if state.runtime.progress().submitted_ticks >= run.target {
            if run.last.is_none() && state.runtime.available_submission_slots() > 0 {
                match state.runtime.submit_command(|_, _| ()) {
                    Ok((_, token)) => run.last = Some(token),
                    Err(err) => {
                        let _ = run.respond.send(AgentResponse::internal(err.to_string()));
                        diagnostics.run_until = None;
                        break;
                    }
                }
            }
            if let Some(token) = run.last.clone() {
                let run = diagnostics.run_until.take().unwrap();
                diagnostics.paused = true;
                diagnostics.fences.push(PendingFence {
                    token,
                    result: json!({"completed_tick":run.target}),
                    app_result: false,
                    respond: run.respond,
                    job: run.job,
                });
            }
            break;
        }
        if diagnostics.boundary_blocked(state.runtime.progress().submitted_ticks)
            || state.runtime.available_submission_slots() <= 1
        {
            break;
        }
        let run = diagnostics.run_until.as_mut().unwrap();
        match state.runtime.try_tick(run.dt) {
            Ok(token) => run.last = Some(token),
            Err(RuntimeError::WouldBlock) => break,
            Err(err) => {
                let _ = run.respond.send(AgentResponse::internal(err.to_string()));
                diagnostics.run_until = None;
                break;
            }
        }
    }
}

fn enqueue_capture<A: App>(
    state: &mut RunState<A>,
    request: CaptureRequest,
    respond: Sender<AgentResponse>,
    job: Option<JobId>,
) {
    let at_tick = request
        .at_tick
        .unwrap_or(state.runtime.progress().submitted_ticks);
    if at_tick < state.runtime.progress().submitted_ticks {
        let _ = respond.send(AgentResponse::bad_request(
            "tick is already past; historical GPU state is not retained",
        ));
        return;
    }
    if matches!(request.target.as_str(), "window" | "scene")
        && (state
            .surface
            .as_ref()
            .expect("active surface")
            .is_suspended()
            || state.suspended
            || state.occluded)
    {
        let _ = respond.send(AgentResponse::unavailable(
            "window presentation is suspended; named resource capture remains available",
        ));
        return;
    }
    state.diagnostics.captures.push_back(PendingCapture {
        request,
        at_tick,
        respond,
        job,
    });
}

fn process_boundaries<A: App>(runtime: &mut AppRuntime<A>, diagnostics: &mut Diagnostics) {
    let current = runtime.progress().submitted_ticks;
    let mut captures = VecDeque::new();
    while let Some(pending) = diagnostics.captures.pop_front() {
        if pending.at_tick > current
            || matches!(pending.request.target.as_str(), "scene" | "window")
            || runtime.available_submission_slots() == 0
        {
            captures.push_back(pending);
            continue;
        }
        let mut registry = CaptureRegistry::new();
        runtime.app.capture_targets(&mut registry);
        let Some(target) = registry.get(&pending.request.target) else {
            let _ = pending
                .respond
                .send(AgentResponse::bad_request("unknown capture target"));
            continue;
        };
        let metadata = ReadbackMetadata {
            target: pending.request.target.clone(),
            tick_id: Some(current),
            submission_id: Some(runtime.progress().submitted_submissions + 1),
            ..Default::default()
        };
        let exact = pending.request.exact;
        let respond = pending.respond.clone();
        let result = runtime.submit_command(|_, ctx| {
            diagnostics.worker.enqueue(
                &ctx.gpu.device,
                ctx.encoder,
                target,
                pending.request,
                metadata,
                pending.respond,
            )
        });
        match result {
            Ok((Ok(_), token)) => {
                if exact {
                    diagnostics.exact_fence = Some(token);
                }
            }
            Ok((Err(error), _)) => {
                let _ = respond.send(error);
            }
            Err(err) => {
                let _ = respond.send(AgentResponse::internal(err.to_string()));
            }
        }
    }
    diagnostics.captures = captures;
    let mut commands = VecDeque::new();
    while let Some(pending) = diagnostics.commands.pop_front() {
        if pending.at_tick > current || runtime.available_submission_slots() == 0 {
            commands.push_back(pending);
            continue;
        }
        match runtime.submit_command(|app, ctx| app.encode_agent_command(pending.payload, ctx)) {
            Ok((Ok(result), token)) => diagnostics.fences.push(PendingFence {
                token,
                result,
                app_result: true,
                respond: pending.respond,
                job: pending.job,
            }),
            Ok((Err(error), _)) => {
                let _ = pending.respond.send(AgentResponse::bad_request(error));
            }
            Err(err) => {
                let _ = pending
                    .respond
                    .send(AgentResponse::internal(err.to_string()));
            }
        }
    }
    diagnostics.commands = commands;
}

pub(super) fn publish<A: App>(bridge: &AgentBridge, state: &RunState<A>) {
    let mut targets = CaptureRegistry::new();
    state.runtime.app.capture_targets(&mut targets);
    bridge.update_snapshot(|snapshot| {
        snapshot.ready = true;
        snapshot.window_size = Some([
            state.surface.as_ref().expect("active surface").size().width,
            state
                .surface
                .as_ref()
                .expect("active surface")
                .size()
                .height,
        ]);
        snapshot.frame_count = state.runtime.progress().submitted_frames;
        snapshot.elapsed = state.runtime.progress().elapsed;
        snapshot.progress = state.runtime.progress().clone();
        snapshot.discarded_wall_time = state.discarded_wall_time.as_secs_f64();
        snapshot.screenshot_supported = !(state
            .surface
            .as_ref()
            .expect("active surface")
            .is_suspended()
            || state.occluded
            || state.suspended);
        snapshot.camera = Some(state.runtime.camera.agent_snapshot());
        snapshot.app = state.runtime.app.agent_status();
        snapshot.capture_targets = targets
            .names()
            .map(str::to_owned)
            .chain(["scene".into(), "window".into()])
            .collect();
        snapshot.gpu = crate::agent::gpu_snapshot(&state.runtime.gpu);
    });
}
