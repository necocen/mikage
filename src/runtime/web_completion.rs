//! Workaround for browser WebGPU implementations that deliver work-done
//! notifications only on a slow periodic poll once ordinary submissions stop.

use std::time::Duration;
#[cfg(not(target_family = "wasm"))]
use std::time::Instant;
#[cfg(target_family = "wasm")]
use web_time::Instant;

// Nominal inactivity threshold, not a timing guarantee. Browsers may clamp or
// throttle nested timers; correctness and submission limits do not depend on it.
const RETRY_DELAY: Duration = Duration::from_millis(2);

/// Queue endpoints, rather than callback counts: a later completion covers all
/// earlier work, even if its individual callbacks have not arrived yet.
#[derive(Default)]
struct Progress {
    submitted: u64,
    completed: u64,
    last_activity: Option<Instant>,
    stopped: bool,
}

impl Progress {
    fn submitted(&mut self, id: u64, now: Instant) {
        self.submitted = id;
        self.last_activity = Some(now);
    }

    fn completed(&mut self, id: u64, now: Instant) {
        if id > self.completed {
            self.completed = id;
            self.last_activity = Some(now);
        }
    }

    fn delay(&self, now: Instant) -> Option<Duration> {
        if self.stopped || self.completed >= self.submitted {
            return None;
        }
        Some(RETRY_DELAY.saturating_sub(now.duration_since(self.last_activity.unwrap())))
    }

    fn probe(&mut self, now: Instant) -> bool {
        if self.delay(now) != Some(Duration::ZERO) {
            return false;
        }
        self.last_activity = Some(now);
        true
    }
}

#[cfg(target_family = "wasm")]
pub(super) use browser::CompletionPump;

#[cfg(target_family = "wasm")]
mod browser {
    use super::*;
    use crate::runtime::{Completion, CompletionSignal, RuntimeError};
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};
    use wasm_bindgen::prelude::*;

    // Global bindings also work in an offscreen Web Worker. AppRuntime does not
    // require a Window or winit event loop.
    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_name = setTimeout, catch)]
        fn set_timeout(callback: &Closure<dyn FnMut()>, delay: i32) -> Result<i32, JsValue>;
        #[wasm_bindgen(js_name = clearTimeout)]
        fn clear_timeout(id: i32);
    }

    #[derive(Default)]
    struct State {
        progress: Progress,
        timeout: Option<i32>,
    }

    /// Only plain Send state is captured by wgpu's Send completion callbacks.
    #[derive(Clone, Default)]
    pub(crate) struct CompletionProgress(Arc<Mutex<State>>);

    impl CompletionProgress {
        pub fn completed(&self, id: u64) {
            let mut state = self.0.lock().unwrap();
            state.progress.completed(id, Instant::now());
            if state.progress.delay(Instant::now()).is_none()
                && let Some(timeout) = state.timeout.take()
            {
                clear_timeout(timeout);
            }
        }

        pub fn stop(&self) {
            let mut state = self.0.lock().unwrap();
            state.progress.stopped = true;
            if let Some(timeout) = state.timeout.take() {
                clear_timeout(timeout);
            }
        }
    }

    pub(crate) struct CompletionPump(Rc<Timer>);

    struct Timer {
        queue: wgpu::Queue,
        progress: CompletionProgress,
        signal: Arc<CompletionSignal>,
        callback: Closure<dyn FnMut()>,
    }

    impl CompletionPump {
        pub fn new(queue: wgpu::Queue, signal: Arc<CompletionSignal>) -> Self {
            Self(Rc::new_cyclic(|weak: &std::rc::Weak<Timer>| {
                let weak = weak.clone();
                Timer {
                    queue,
                    progress: CompletionProgress::default(),
                    signal,
                    callback: Closure::new(move || {
                        if let Some(timer) = weak.upgrade() {
                            timer.wake();
                        }
                    }),
                }
            }))
        }

        pub fn progress(&self) -> CompletionProgress {
            self.0.progress.clone()
        }

        pub fn submitted(&self, id: u64) {
            self.0
                .progress
                .0
                .lock()
                .unwrap()
                .progress
                .submitted(id, Instant::now());
            self.0.arm();
        }
    }

    impl Drop for CompletionPump {
        fn drop(&mut self) {
            self.0.progress.stop();
        }
    }

    impl Timer {
        fn arm(&self) {
            let mut state = self.progress.0.lock().unwrap();
            if state.timeout.is_some() {
                return;
            }
            let Some(delay) = state.progress.delay(Instant::now()) else {
                return;
            };
            // Round up so early timer delivery never creates a zero-delay spin.
            let millis = delay.as_nanos().div_ceil(1_000_000).max(1) as i32;
            match set_timeout(&self.callback, millis) {
                Ok(timeout) => state.timeout = Some(timeout),
                Err(error) => {
                    state.progress.stopped = true;
                    drop(state);
                    self.signal
                        .send(Completion::Failed(RuntimeError::Poll(format!(
                            "WebGPU completion timer failed: {error:?}"
                        ))));
                }
            }
        }

        fn wake(&self) {
            let probe = {
                let mut state = self.progress.0.lock().unwrap();
                state.timeout = None;
                state.progress.probe(Instant::now())
            };
            if probe {
                // Encourage delivery of existing notifications without adding
                // app work, tokens, callbacks, or releasing submission capacity.
                // Never hold the state lock across a queue call.
                self.queue.submit([]);
            }
            self.arm();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_completion_and_idle_never_probe() {
        let now = Instant::now();
        let mut progress = Progress::default();
        assert_eq!(progress.delay(now), None);
        progress.submitted(1, now);
        progress.completed(1, now + RETRY_DELAY / 4);
        assert_eq!(progress.delay(now + RETRY_DELAY), None);
        assert!(!progress.probe(now + Duration::from_secs(60)));
    }

    #[test]
    fn delayed_frame_completion_probes_even_with_spare_capacity() {
        let now = Instant::now();
        let mut progress = Progress::default();
        // A single VSync frame can block rendering with seven slots still free.
        progress.submitted(1, now);
        assert!(!progress.probe(now + RETRY_DELAY - Duration::from_nanos(1)));
        assert!(progress.probe(now + RETRY_DELAY));
        assert!(!progress.probe(now + RETRY_DELAY));
        assert_eq!(progress.delay(now + RETRY_DELAY), Some(RETRY_DELAY));
        assert!(progress.probe(now + RETRY_DELAY * 2));
        progress.completed(1, now + RETRY_DELAY * 2);
        assert_eq!(progress.delay(now + RETRY_DELAY * 3), None);
    }

    #[test]
    fn real_submissions_and_completion_progress_defer_probes() {
        let now = Instant::now();
        let mut progress = Progress::default();
        progress.submitted(1, now);
        progress.submitted(2, now + RETRY_DELAY * 7 / 8);
        assert!(!progress.probe(now + RETRY_DELAY));
        progress.completed(1, now + RETRY_DELAY * 14 / 8);
        assert!(!progress.probe(now + RETRY_DELAY * 15 / 8));
        assert!(progress.probe(now + RETRY_DELAY * 22 / 8));
    }

    #[test]
    fn probes_never_complete_work_or_bypass_real_gpu_backpressure() {
        let now = Instant::now();
        let mut progress = Progress::default();
        progress.submitted(8, now);
        for retry in 1..=100 {
            assert!(progress.probe(now + RETRY_DELAY * retry));
            assert_eq!(progress.submitted - progress.completed, 8);
        }
        progress.completed(8, now + RETRY_DELAY * 100);
        assert_eq!(progress.delay(now + RETRY_DELAY * 101), None);
        progress.submitted(9, now + RETRY_DELAY * 102);
        assert!(progress.probe(now + RETRY_DELAY * 103));
    }

    #[test]
    fn late_callbacks_cannot_defer_new_work_or_regress_completion() {
        let now = Instant::now();
        let mut progress = Progress::default();
        progress.submitted(8, now);
        progress.completed(8, now);
        progress.submitted(9, now);
        progress.completed(3, now + RETRY_DELAY * 7 / 8);
        assert_eq!(progress.completed, 8);
        assert!(progress.probe(now + RETRY_DELAY));
        progress.completed(9, now + RETRY_DELAY);
        assert_eq!(progress.delay(now + RETRY_DELAY), None);
    }

    #[test]
    fn device_loss_or_shutdown_stops_retries_despite_late_callbacks() {
        let now = Instant::now();
        let mut progress = Progress::default();
        progress.submitted(1, now);
        progress.stopped = true;
        assert!(!progress.probe(now + RETRY_DELAY));
        progress.completed(1, now + RETRY_DELAY);
        assert_eq!(progress.delay(now + RETRY_DELAY), None);
    }
}
