//! Real browser-window coverage for runtime automatic tick control.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use mikage::{
    App, GpuContext, PresentationPolicy, RedrawPolicy, RedrawRequest, RenderContext,
    RenderUpdateContext, RunConfig, SimulationControl, SimulationPolicy, SubmissionToken,
    TickContext, wgpu,
};
use wasm_bindgen::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum PauseHook {
    Tick,
    Submit,
    Complete,
    Gui,
}

#[derive(Default)]
struct State {
    started: bool,
    ticks: u64,
    tick_submits: u64,
    frames: u64,
    completed_frames: u64,
    submitted: u64,
    completed: u64,
    elapsed: f64,
    last_dt: f32,
    snapshot: u32,
    rendered_snapshot: u32,
    pause_hook: Option<PauseHook>,
    pause_boundary: u64,
    hook_pauses: u64,
}

thread_local! {
    static STATE: Rc<RefCell<State>> = Rc::new(RefCell::new(State::default()));
    static CONTROL: RefCell<Option<SimulationControl>> = const { RefCell::new(None) };
    static REDRAW: RefCell<Option<RedrawRequest>> = const { RefCell::new(None) };
}

struct Counter {
    state: Rc<RefCell<State>>,
    prepared_snapshot: u32,
}

impl Counter {
    fn pause_at(&self, hook: PauseHook) {
        let should_pause = {
            let mut state = self.state.borrow_mut();
            if state.pause_hook == Some(hook) {
                state.pause_hook = None;
                state.pause_boundary = state.ticks;
                state.hook_pauses += 1;
                true
            } else {
                false
            }
        };
        if should_pause {
            CONTROL.with(|control| control.borrow().as_ref().unwrap().pause());
        }
    }
}

impl App for Counter {
    type Camera = ();

    fn set_redraw_request(&mut self, request: RedrawRequest) {
        REDRAW.with(|redraw| *redraw.borrow_mut() = Some(request));
    }

    fn set_simulation_control(&mut self, control: SimulationControl) {
        CONTROL.with(|slot| *slot.borrow_mut() = Some(control));
    }

    fn tick(&mut self, ctx: &mut TickContext<'_>) {
        {
            let mut state = self.state.borrow_mut();
            state.ticks = ctx.tick_id;
            state.last_dt = ctx.dt;
            state.elapsed = ctx.elapsed;
        }
        // Deliberately leave this encoder empty, as in the Worker-driven mode.
        self.pause_at(PauseHook::Tick);
    }

    fn prepare_render(&mut self, _: &mut RenderUpdateContext<'_, ()>) {
        self.prepared_snapshot = self.state.borrow().snapshot;
    }

    fn render(&mut self, ctx: &mut RenderContext<'_, ()>) {
        let attachment = ctx.color_attachment(wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color {
                r: f64::from(self.prepared_snapshot % 256) / 255.0,
                g: 0.1,
                b: 0.25,
                a: 1.0,
            }),
            store: wgpu::StoreOp::Store,
        });
        let _pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("simulation_control_snapshot"),
            color_attachments: &[Some(attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.state.borrow_mut().rendered_snapshot = self.prepared_snapshot;
    }

    fn gui(&mut self, ui: &mut mikage::egui::Ui) {
        ui.label("Simulation control regression");
        self.pause_at(PauseHook::Gui);
    }

    fn after_submit(&mut self, _: &GpuContext, token: &SubmissionToken) {
        {
            let mut state = self.state.borrow_mut();
            state.submitted = token.id;
            if let Some(frame) = token.frame_id {
                state.frames = frame;
            } else {
                state.tick_submits += 1;
            }
        }
        self.pause_at(PauseHook::Submit);
    }

    fn after_complete(&mut self, _: &GpuContext, token: &SubmissionToken) {
        {
            let mut state = self.state.borrow_mut();
            state.completed = token.id;
            if let Some(frame) = token.frame_id {
                state.completed_frames = frame;
            }
        }
        self.pause_at(PauseHook::Complete);
    }
}

fn with_control(action: impl FnOnce(&SimulationControl)) -> Result<(), JsValue> {
    CONTROL.with(|slot| {
        let control = slot.borrow();
        action(
            control
                .as_ref()
                .ok_or_else(|| JsValue::from_str("runner is not ready"))?,
        );
        Ok(())
    })
}

#[wasm_bindgen]
pub fn start(policy: &str) -> Result<(), JsValue> {
    let policy = match policy {
        "fixed" => SimulationPolicy::Fixed {
            dt: Duration::from_micros(400),
            max_ticks_per_wake: 4,
            max_backlog: Duration::from_micros(1600),
        },
        "per-redraw" => SimulationPolicy::PerRedraw,
        "manual" => SimulationPolicy::Manual,
        _ => return Err(JsValue::from_str("unknown simulation policy")),
    };
    let state = STATE.with(Rc::clone);
    if state.borrow().started {
        return Err(JsValue::from_str("start must be called once per page"));
    }
    state.borrow_mut().started = true;
    let mut config = RunConfig::new("Simulation control regression")
        .with_camera(())
        .with_size(640, 360)
        .with_simulation_policy(policy)
        .with_redraw_policy(RedrawPolicy::Reactive)
        .with_presentation_policy(PresentationPolicy::Vsync);
    config.canvas = Some("#canvas".into());
    mikage::run(
        move |_, _, _| Counter {
            state,
            prepared_snapshot: 0,
        },
        config,
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))
}

#[wasm_bindgen]
pub fn pause() -> Result<(), JsValue> {
    with_control(SimulationControl::pause)
}

#[wasm_bindgen]
pub fn resume() -> Result<(), JsValue> {
    with_control(SimulationControl::resume)
}

#[wasm_bindgen]
pub fn pause_in_hook(hook: &str) -> Result<(), JsValue> {
    let hook = match hook {
        "tick" => PauseHook::Tick,
        "submit" => PauseHook::Submit,
        "complete" => PauseHook::Complete,
        "gui" => PauseHook::Gui,
        _ => return Err(JsValue::from_str("unknown pause hook")),
    };
    STATE.with(|state| state.borrow_mut().pause_hook = Some(hook));
    Ok(())
}

#[wasm_bindgen]
pub fn publish_snapshot(value: u32) -> Result<(), JsValue> {
    STATE.with(|state| state.borrow_mut().snapshot = value);
    REDRAW.with(|slot| {
        slot.borrow()
            .as_ref()
            .ok_or_else(|| JsValue::from_str("runner is not ready"))?
            .request_redraw();
        Ok(())
    })
}

#[wasm_bindgen]
pub fn snapshot() -> String {
    let ready = CONTROL.with(|slot| slot.borrow().is_some());
    let paused = CONTROL.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(SimulationControl::is_paused)
    });
    STATE.with(|shared| {
        let state = shared.borrow();
        format!(
            "{{\"ready\":{ready},\"paused\":{paused},\"ticks\":{},\"tickSubmits\":{},\"frames\":{},\"completedFrames\":{},\"submitted\":{},\"completed\":{},\"elapsed\":{},\"lastDt\":{},\"snapshot\":{},\"renderedSnapshot\":{},\"pauseBoundary\":{},\"hookPauses\":{}}}",
            state.ticks,
            state.tick_submits,
            state.frames,
            state.completed_frames,
            state.submitted,
            state.completed,
            state.elapsed,
            state.last_dt,
            state.snapshot,
            state.rendered_snapshot,
            state.pause_boundary,
            state.hook_pauses,
        )
    })
}
