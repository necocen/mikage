//! Browser regression probe: async progress without a window or host polling.

use mikage::{App, AppRuntime, GpuContext, GpuDescriptor, RuntimeConfig, RuntimeError};
use std::time::Duration;
use wasm_bindgen::prelude::*;

#[derive(Default)]
struct Counter {
    ticks: u64,
    submitted: u64,
    completed: u64,
}

impl App for Counter {
    type Camera = ();

    fn tick(&mut self, _: &mut mikage::TickContext<'_>) {
        self.ticks += 1;
    }

    fn after_submit(&mut self, _: &GpuContext, token: &mikage::SubmissionToken) {
        assert_eq!(token.id, self.submitted + 1);
        self.submitted = token.id;
    }

    fn after_complete(&mut self, _: &GpuContext, token: &mikage::SubmissionToken) {
        assert_eq!(token.id, self.completed + 1);
        self.completed = token.id;
    }
}

fn error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[wasm_bindgen]
pub async fn run(case: &str) -> Result<(), JsValue> {
    let gpu = GpuContext::headless(GpuDescriptor::default())
        .await
        .map_err(error)?;
    let mut runtime = AppRuntime::new(gpu, Counter::default(), ());
    runtime
        .set_config(RuntimeConfig {
            max_in_flight_submissions: 2,
        })
        .map_err(error)?;
    let dt = Duration::from_micros(400);
    let first = runtime.try_tick(dt).map_err(error)?;
    match case {
        "timer-error" => {
            assert!(matches!(
                runtime.poll_completions(),
                Err(RuntimeError::Poll(_))
            ));
            return Ok(());
        }
        "shutdown" => {
            runtime.shutdown();
            runtime.shutdown();
            assert_eq!(runtime.poll_completions(), Err(RuntimeError::Stopped));
            return Ok(());
        }
        "drop" => return Ok(()),
        "device-loss" => {
            runtime.gpu.device.destroy();
            assert!(matches!(
                runtime.wait_for_async(&first).await,
                Err(RuntimeError::DeviceLost(_))
            ));
            return Ok(());
        }
        _ => {}
    }
    let second = runtime.try_tick(dt).map_err(error)?;
    assert!(matches!(
        runtime.try_tick(dt),
        Err(RuntimeError::WouldBlock)
    ));
    assert_eq!(runtime.app.ticks, 2);
    assert_eq!(runtime.in_flight_submissions(), 2);
    runtime.wait_for_capacity_async().await.map_err(error)?;
    runtime.wait_for_async(&second).await.map_err(error)?;
    let last = runtime
        .advance_ticks_async(4, dt)
        .await
        .map_err(error)?
        .unwrap();
    runtime.wait_for_async(&last).await.map_err(error)?;
    let progress = runtime.progress();
    assert_eq!(progress.submitted_ticks, 6);
    assert_eq!(progress.completed_ticks, 6);
    assert_eq!(progress.submitted_submissions, 6);
    assert_eq!(progress.completed_submissions, 6);
    assert_eq!(progress.submitted_frames, 0);
    assert_eq!(runtime.app.submitted, 6);
    assert_eq!(runtime.app.completed, 6);
    assert_eq!(runtime.in_flight_submissions(), 0);
    Ok(())
}
