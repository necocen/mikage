#[cfg(not(target_family = "wasm"))]
use std::time::Instant;
#[cfg(target_family = "wasm")]
use web_time::Instant;

/// Frame timing.
///
/// The framework calls [`tick`](FrameTime::tick) automatically each frame,
/// updating delta time and elapsed time. Applications access these values
/// through [`UpdateContext::dt`](crate::UpdateContext::dt) and
/// [`UpdateContext::elapsed`](crate::UpdateContext::elapsed).
pub struct FrameTime {
    last_instant: Instant,
    pub dt: f32,
    pub elapsed: f64,
    pub frame_count: u64,
}

impl FrameTime {
    pub fn new() -> Self {
        Self {
            last_instant: Instant::now(),
            dt: 0.0,
            elapsed: 0.0,
            frame_count: 0,
        }
    }

    /// Updates `dt` and `elapsed`. Called automatically at the start of each frame.
    pub fn tick(&mut self) {
        let now = Instant::now();
        self.dt = now.duration_since(self.last_instant).as_secs_f32();
        self.elapsed += self.dt as f64;
        self.frame_count += 1;
        self.last_instant = now;
    }
}

impl Default for FrameTime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state() {
        let ft = FrameTime::new();
        assert_eq!(ft.dt, 0.0);
        assert_eq!(ft.elapsed, 0.0);
        assert_eq!(ft.frame_count, 0);
    }

    #[test]
    fn tick_increments_count() {
        let mut ft = FrameTime::new();
        ft.tick();
        assert_eq!(ft.frame_count, 1);
        ft.tick();
        assert_eq!(ft.frame_count, 2);
    }

    #[test]
    fn tick_elapsed_nonnegative() {
        let mut ft = FrameTime::new();
        ft.tick();
        assert!(ft.dt >= 0.0);
        assert!(ft.elapsed >= 0.0);
    }
}
