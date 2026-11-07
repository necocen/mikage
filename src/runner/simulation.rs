use super::Instant;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

const PAUSED: u8 = 1;
const CHANGED: u8 = 2;

/// A cloneable request to pause or resume the window runner's automatic ticks.
///
/// Supplied once through [`crate::App::set_simulation_control`]. Both `Fixed`
/// and `PerRedraw` ticks pause; input, cameras, GUI, redraws and GPU completion
/// processing continue. Explicit runtime/agent stepping remains available.
/// Requests wake the host and apply between ticks, without cancelling a tick or
/// submission already in progress. Resume preserves the configured policy and
/// fixed dt, and discards paused wall time and any pre-pause backlog.
#[derive(Clone)]
pub struct SimulationControl {
    state: Arc<AtomicU8>,
    wake: Arc<dyn Fn() + Send + Sync>,
}

impl SimulationControl {
    pub(super) fn new(wake: Arc<dyn Fn() + Send + Sync>) -> Self {
        Self {
            state: Arc::new(AtomicU8::new(0)),
            wake,
        }
    }

    /// Request a pause of automatic ticks at the next host scheduling boundary.
    /// This does not wait for outstanding GPU work to finish.
    pub fn pause(&self) {
        self.set_paused(true);
    }

    /// Request automatic ticks to resume without catching up paused wall time.
    /// Other stop reasons, such as an agent pause or OS suspension, still apply.
    pub fn resume(&self) {
        self.set_paused(false);
    }

    /// The application's requested pause state, not a GPU fence or an
    /// acknowledgement that the host has applied it. Other stop reasons are
    /// independent of this value.
    pub fn is_paused(&self) -> bool {
        self.state.load(Ordering::Acquire) & PAUSED != 0
    }

    fn set_paused(&self, paused: bool) {
        let requested = if paused { PAUSED } else { 0 };
        if let Ok(previous) =
            self.state
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                    (state & PAUSED != requested).then_some(requested | CHANGED)
                })
            && previous & CHANGED == 0
        {
            (self.wake)();
        }
    }

    fn take_pending(&self) -> Option<bool> {
        let previous = self.state.fetch_and(!CHANGED, Ordering::AcqRel);
        (previous & CHANGED != 0).then_some(previous & PAUSED != 0)
    }

    // A pending pause -> resume must reset the clock before another tick, even
    // when its final requested state is the same as the applied state.
    pub(super) fn permits_tick(&self) -> bool {
        self.state.load(Ordering::Acquire) == 0
    }
}

pub(super) struct AutomaticSimulation {
    pub control: SimulationControl,
    paused: bool,
    resumed_at: Instant,
}

impl AutomaticSimulation {
    pub fn new(control: SimulationControl, now: Instant) -> Self {
        Self {
            control,
            paused: false,
            resumed_at: now,
        }
    }

    pub fn apply_pending(
        &mut self,
        now: Instant,
        backlog: &mut Duration,
        last_schedule: &mut Instant,
    ) -> bool {
        let Some(paused) = self.control.take_pending() else {
            return false;
        };
        self.paused = paused;
        self.resumed_at = now;
        *backlog = Duration::ZERO;
        *last_schedule = now;
        true
    }

    pub fn permits_tick(&self) -> bool {
        !self.paused && self.control.permits_tick()
    }

    pub fn per_redraw_dt(&self, now: Instant, last_render: Instant) -> Duration {
        now.duration_since(last_render.max(self.resumed_at))
            .min(Duration::from_millis(250))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn coalesced_transitions_reset_backlog_even_when_running_again() {
        let now = Instant::now();
        let wakes = Arc::new(AtomicUsize::new(0));
        let count = wakes.clone();
        let control = SimulationControl::new(Arc::new(move || {
            count.fetch_add(1, Ordering::Relaxed);
        }));
        let mut automatic = AutomaticSimulation::new(control.clone(), now);
        let mut backlog = Duration::from_millis(100);
        let mut last_schedule = now;
        control.pause();
        control.pause();
        control.resume();
        assert!(!control.is_paused());
        assert!(!automatic.permits_tick());
        assert_eq!(wakes.load(Ordering::Relaxed), 1);
        let resumed = now + Duration::from_secs(60);
        assert!(automatic.apply_pending(resumed, &mut backlog, &mut last_schedule));
        assert!(automatic.permits_tick());
        assert_eq!(backlog, Duration::ZERO);
        assert_eq!(last_schedule, resumed);
        assert_eq!(automatic.per_redraw_dt(resumed, now), Duration::ZERO);
        assert_eq!(
            automatic.per_redraw_dt(resumed + Duration::from_millis(5), now),
            Duration::from_millis(5)
        );
        assert!(!automatic.apply_pending(resumed, &mut backlog, &mut last_schedule));
        control.resume();
        assert_eq!(wakes.load(Ordering::Relaxed), 1);
        let worker = control.clone();
        std::thread::spawn(move || worker.pause()).join().unwrap();
        assert_eq!(wakes.load(Ordering::Relaxed), 2);
        assert!(automatic.apply_pending(resumed, &mut backlog, &mut last_schedule));
        assert!(!automatic.permits_tick());
        assert!(control.is_paused());
        control.resume();
        assert!(automatic.apply_pending(resumed, &mut backlog, &mut last_schedule));
        assert!(automatic.permits_tick());
    }
}
