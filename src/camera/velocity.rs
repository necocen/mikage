//! Time-based drag velocity estimation and consecutive-flick momentum compounding.
//!
//! Converts incoming drag deltas into a 60fps reference-frame velocity independent
//! of input sampling rate (60Hz mouse, 120Hz/240Hz mobile touch, or batched browser
//! events), resists lift-off micro-jitter, drops inertia when the pointer pauses
//! before release, and compounds momentum on rapid same-direction consecutive flicks.

use std::collections::VecDeque;
use std::time::Duration;

use glam::{Vec2, Vec3};

#[cfg(not(target_family = "wasm"))]
pub(crate) use std::time::Instant;
#[cfg(target_family = "wasm")]
pub(crate) use web_time::Instant;

/// Reference frame duration (60fps) used for camera inertia velocities.
pub(crate) const REFERENCE_DT: f32 = 1.0 / 60.0;

const MAX_SAMPLES: usize = 32;
const WINDOW: Duration = Duration::from_millis(100);
const COALESCE_INTERVAL: Duration = Duration::from_micros(500);
const MIN_SPAN: Duration = Duration::from_millis(4);
const DECAY_TAU_SECS: f32 = 0.050;
const HOLD_GRACE: Duration = Duration::from_millis(40);
const HOLD_STOP: Duration = Duration::from_millis(100);
const EMA_SMOOTHING: f32 = 0.5;
const MAX_COMPOUND_DRAG_DURATION: Duration = Duration::from_millis(350);
const COMPOUND_CARRY_FACTOR: f32 = 0.8;
const COMPOUND_ALIGNMENT_MIN: f32 = 0.5;
const MAX_COMPOUND_MULTIPLIER: f32 = 3.0;
const MIN_VELOCITY_LEN_SQ: f32 = 1e-12;

pub(crate) trait VelocityVector: Copy {
    const ZERO: Self;
    fn add(self, other: Self) -> Self;
    fn scale(self, factor: f32) -> Self;
    fn dot(self, other: Self) -> f32;
    fn length_squared(self) -> f32;
}

impl VelocityVector for Vec2 {
    const ZERO: Self = Vec2::ZERO;

    fn add(self, other: Self) -> Self {
        self + other
    }

    fn scale(self, factor: f32) -> Self {
        self * factor
    }

    fn dot(self, other: Self) -> f32 {
        Vec2::dot(self, other)
    }

    fn length_squared(self) -> f32 {
        Vec2::length_squared(self)
    }
}

impl VelocityVector for Vec3 {
    const ZERO: Self = Vec3::ZERO;

    fn add(self, other: Self) -> Self {
        self + other
    }

    fn scale(self, factor: f32) -> Self {
        self * factor
    }

    fn dot(self, other: Self) -> f32 {
        Vec3::dot(self, other)
    }

    fn length_squared(self) -> f32 {
        Vec3::length_squared(self)
    }
}

#[derive(Clone, Copy)]
struct Sample<V> {
    time: Instant,
    delta: V,
}

/// Estimates release velocity (in units per 60fps reference frame) from drag deltas.
pub(crate) struct DragVelocityTracker<V: VelocityVector> {
    samples: VecDeque<Sample<V>>,
    active: bool,
    drag_start: Option<Instant>,
    carry_velocity: V,
    ema_delta: V,
}

impl<V: VelocityVector> Default for DragVelocityTracker<V> {
    fn default() -> Self {
        Self {
            samples: VecDeque::new(),
            active: false,
            drag_start: None,
            carry_velocity: V::ZERO,
            ema_delta: V::ZERO,
        }
    }
}

impl<V: VelocityVector> DragVelocityTracker<V> {
    /// Records a drag delta at the current time and returns the live velocity estimate
    /// in units per 60fps reference frame.
    pub(crate) fn record(&mut self, delta: V, current_velocity: V) -> V {
        self.record_at(delta, current_velocity, Instant::now())
    }

    /// Deterministic variant of [`Self::record`] using an explicit timestamp.
    pub(crate) fn record_at(&mut self, delta: V, current_velocity: V, now: Instant) -> V {
        if !self.active {
            self.active = true;
            self.drag_start = Some(now);
            self.carry_velocity = current_velocity;
            self.samples.clear();
            self.ema_delta = current_velocity;
        }

        while let Some(front) = self.samples.front() {
            if now.saturating_duration_since(front.time) > WINDOW {
                self.samples.pop_front();
            } else {
                break;
            }
        }
        // If all earlier samples expired while the pointer sat still mid-drag,
        // drop pre-drag carry momentum and stale EMA state.
        if self.samples.is_empty()
            && self
                .drag_start
                .is_some_and(|start| now.saturating_duration_since(start) > WINDOW)
        {
            self.carry_velocity = V::ZERO;
            self.ema_delta = V::ZERO;
        }

        self.ema_delta = self
            .ema_delta
            .scale(EMA_SMOOTHING)
            .add(delta.scale(1.0 - EMA_SMOOTHING));

        if let Some(last) = self.samples.back_mut()
            && now.saturating_duration_since(last.time) < COALESCE_INTERVAL
        {
            last.delta = last.delta.add(delta);
        } else {
            if self.samples.len() == MAX_SAMPLES {
                self.samples.pop_front();
            }
            self.samples.push_back(Sample { time: now, delta });
        }

        self.estimate_velocity(now)
    }

    /// Finalizes an active drag at the current time, returning the release velocity
    /// (including hold-before-lift decay and same-direction momentum compounding).
    /// Returns `None` if no drag was active on this tracker.
    pub(crate) fn finish(&mut self) -> Option<V> {
        self.finish_at(Instant::now())
    }

    /// Deterministic variant of [`Self::finish`] using an explicit timestamp.
    pub(crate) fn finish_at(&mut self, now: Instant) -> Option<V> {
        if !self.active {
            return None;
        }
        self.active = false;
        let base_velocity = self.estimate_velocity(now);
        let final_velocity = self.compound_momentum(base_velocity, now);
        self.samples.clear();
        self.drag_start = None;
        self.carry_velocity = V::ZERO;
        self.ema_delta = V::ZERO;
        Some(final_velocity)
    }

    /// Discards any in-progress gesture samples without producing a release velocity.
    pub(crate) fn cancel(&mut self) {
        self.active = false;
        self.samples.clear();
        self.drag_start = None;
        self.carry_velocity = V::ZERO;
        self.ema_delta = V::ZERO;
    }

    fn estimate_velocity(&self, now: Instant) -> V {
        let (Some(first), Some(last)) = (self.samples.front(), self.samples.back()) else {
            return V::ZERO;
        };

        let silence = now.saturating_duration_since(last.time);
        if silence >= HOLD_STOP {
            return V::ZERO;
        }
        let hold_scale = if silence <= HOLD_GRACE {
            1.0
        } else {
            (HOLD_STOP - silence).as_secs_f32() / (HOLD_STOP - HOLD_GRACE).as_secs_f32()
        };

        let span = last.time.saturating_duration_since(first.time);
        if self.samples.len() >= 2 && span >= MIN_SPAN {
            let t_end = last.time;
            let mut weighted_disp = V::ZERO;
            let mut weighted_time = 0.0_f32;
            for i in 1..self.samples.len() {
                let prev = self.samples[i - 1];
                let curr = self.samples[i];
                let dt = curr.time.saturating_duration_since(prev.time).as_secs_f32();
                if dt <= 0.0 {
                    continue;
                }
                let age = t_end.saturating_duration_since(curr.time).as_secs_f32() + 0.5 * dt;
                let weight = (-age / DECAY_TAU_SECS).exp();
                weighted_disp = weighted_disp.add(curr.delta.scale(weight));
                weighted_time += dt * weight;
            }
            if weighted_time > 1e-6 {
                let per_frame = weighted_disp.scale(REFERENCE_DT / weighted_time);
                return per_frame.scale(hold_scale);
            }
        }

        // Fallback for single-sample or sub-MIN_SPAN synthetic drags.
        self.ema_delta.scale(hold_scale)
    }

    fn compound_momentum(&self, new_vel: V, now: Instant) -> V {
        let Some(start) = self.drag_start else {
            return new_vel;
        };
        if now.saturating_duration_since(start) > MAX_COMPOUND_DRAG_DURATION {
            return new_vel;
        }
        let new_len_sq = new_vel.length_squared();
        let carry_len_sq = self.carry_velocity.length_squared();
        if new_len_sq <= MIN_VELOCITY_LEN_SQ || carry_len_sq <= MIN_VELOCITY_LEN_SQ {
            return new_vel;
        }
        let new_len = new_len_sq.sqrt();
        let carry_len = carry_len_sq.sqrt();
        let dot = new_vel.dot(self.carry_velocity);
        let alignment = dot / (new_len * carry_len);
        if alignment < COMPOUND_ALIGNMENT_MIN {
            return new_vel;
        }
        let carry_along_new = dot / new_len;
        let boosted_speed = (new_len + carry_along_new * COMPOUND_CARRY_FACTOR)
            .min(new_len * MAX_COMPOUND_MULTIPLIER);
        new_vel.scale(boosted_speed / new_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simulate_constant_swipe(hz: u32, speed_per_sec: Vec2, duration_ms: u64) -> Vec2 {
        let mut tracker = DragVelocityTracker::<Vec2>::default();
        let start = Instant::now();
        let step_us = 1_000_000 / u64::from(hz);
        let delta = speed_per_sec * (step_us as f32 / 1_000_000.0);
        let mut t = start;
        let end = start + Duration::from_millis(duration_ms);
        while t <= end {
            tracker.record_at(delta, Vec2::ZERO, t);
            t += Duration::from_micros(step_us);
        }
        tracker.finish_at(t).unwrap()
    }

    #[test]
    fn sampling_rate_independent_across_60_120_240_hz() {
        let speed = Vec2::new(1200.0, -450.0);
        let expected = speed * REFERENCE_DT;
        for hz in [60, 120, 240] {
            let v = simulate_constant_swipe(hz, speed, 80);
            let rel_err = (v - expected).length() / expected.length();
            assert!(
                rel_err < 0.02,
                "hz={hz}: expected {expected:?}, got {v:?} (rel_err={rel_err:.4})"
            );
        }
    }

    #[test]
    fn coalesces_batched_events_in_same_frame() {
        let mut tracker = DragVelocityTracker::<Vec2>::default();
        let t0 = Instant::now();
        // Two 50-unit events delivered together at t=0, then two more at t=16ms and t=32ms.
        // True rate is 100 units per 16ms (= 6250 units/s).
        for step in 0..3 {
            let t = t0 + Duration::from_millis(step * 16);
            tracker.record_at(Vec2::new(50.0, 0.0), Vec2::ZERO, t);
            tracker.record_at(
                Vec2::new(50.0, 0.0),
                Vec2::ZERO,
                t + Duration::from_micros(100),
            );
        }
        let v = tracker.finish_at(t0 + Duration::from_millis(36)).unwrap();
        let expected = 100.0 * (REFERENCE_DT / 0.016);
        assert!(
            (v.x - expected).abs() / expected < 0.02,
            "expected ~{expected:.2}, got {:.2}",
            v.x
        );
    }

    #[test]
    fn resists_liftoff_micro_jitter() {
        let mut tracker = DragVelocityTracker::<Vec2>::default();
        let t0 = Instant::now();
        // 120Hz brisk swipe at 1200 units/s (10 units per 8.33ms) for 75ms.
        for i in 0..10 {
            let t = t0 + Duration::from_micros(i * 8333);
            tracker.record_at(Vec2::new(10.0, 0.0), Vec2::ZERO, t);
        }
        // One tiny noisy lift-off event right before release.
        let t_jitter = t0 + Duration::from_micros(10 * 8333);
        tracker.record_at(Vec2::new(0.5, 0.0), Vec2::ZERO, t_jitter);
        let v = tracker
            .finish_at(t_jitter + Duration::from_millis(4))
            .unwrap();
        let full_speed = 1200.0 * REFERENCE_DT; // 20.0
        assert!(
            v.x > full_speed * 0.80,
            "lift-off jitter crushed release speed: got {:.2}, full {:.2}",
            v.x,
            full_speed
        );
    }

    #[test]
    fn hold_before_lift_stops_inertia() {
        let mut tracker = DragVelocityTracker::<Vec2>::default();
        let t0 = Instant::now();
        for i in 0..8 {
            tracker.record_at(
                Vec2::new(15.0, 0.0),
                Vec2::ZERO,
                t0 + Duration::from_millis(i * 8),
            );
        }
        // Hold still for 110ms before releasing.
        let v = tracker
            .finish_at(t0 + Duration::from_millis(56 + 110))
            .unwrap();
        assert_eq!(v, Vec2::ZERO);
    }

    #[test]
    fn brief_pause_between_grace_and_stop_fades_smoothly() {
        let run = |silence_ms: u64| {
            let mut tracker = DragVelocityTracker::<Vec2>::default();
            let t0 = Instant::now();
            for i in 0..8 {
                tracker.record_at(
                    Vec2::new(10.0, 0.0),
                    Vec2::ZERO,
                    t0 + Duration::from_millis(i * 8),
                );
            }
            tracker
                .finish_at(t0 + Duration::from_millis(56 + silence_ms))
                .unwrap()
                .x
        };
        let v_fast = run(10);
        let v_mid = run(70);
        let v_stop = run(100);
        assert!(v_fast > v_mid && v_mid > 0.0);
        assert_eq!(v_stop, 0.0);
    }

    #[test]
    fn consecutive_same_direction_flicks_compound_momentum() {
        let mut tracker = DragVelocityTracker::<Vec2>::default();
        let t0 = Instant::now();
        let carry = Vec2::new(15.0, 0.0);
        // Quick 48ms flick in the same direction (+X) at 1200 units/s (20 units/frame).
        for i in 0..7 {
            tracker.record_at(
                Vec2::new(10.0, 0.0),
                carry,
                t0 + Duration::from_micros(i * 8333),
            );
        }
        let v = tracker.finish_at(t0 + Duration::from_millis(54)).unwrap();
        let single_speed = 20.0;
        let expected = single_speed + carry.x * COMPOUND_CARRY_FACTOR;
        assert!(
            (v.x - expected).abs() < 0.5,
            "expected compounded ~{expected:.2}, got {:.2}",
            v.x
        );
    }

    #[test]
    fn opposite_or_slow_drag_does_not_compound_momentum() {
        let t0 = Instant::now();
        let carry = Vec2::new(20.0, 0.0);

        // Opposite direction flick should NOT keep previous +X momentum.
        let mut rev = DragVelocityTracker::<Vec2>::default();
        for i in 0..6 {
            rev.record_at(
                Vec2::new(-10.0, 0.0),
                carry,
                t0 + Duration::from_micros(i * 8333),
            );
        }
        let v_rev = rev.finish_at(t0 + Duration::from_millis(45)).unwrap();
        assert!(v_rev.x < -19.0 && v_rev.x > -21.0, "got {v_rev:?}");

        // Long slow drag (> 350ms) should NOT inherit carry from before the drag.
        let mut long_drag = DragVelocityTracker::<Vec2>::default();
        for i in 0..50 {
            long_drag.record_at(
                Vec2::new(10.0, 0.0),
                carry,
                t0 + Duration::from_millis(i * 10),
            );
        }
        let v_long = long_drag
            .finish_at(t0 + Duration::from_millis(500))
            .unwrap();
        let unboosted = 10.0 * (REFERENCE_DT / 0.010);
        assert!(
            (v_long.x - unboosted).abs() / unboosted < 0.05,
            "long drag should not compound: expected ~{unboosted:.2}, got {:.2}",
            v_long.x
        );
    }

    #[test]
    fn compound_speed_is_capped() {
        let mut tracker = DragVelocityTracker::<Vec2>::default();
        let t0 = Instant::now();
        let huge_carry = Vec2::new(500.0, 0.0);
        for i in 0..6 {
            tracker.record_at(
                Vec2::new(10.0, 0.0),
                huge_carry,
                t0 + Duration::from_micros(i * 8333),
            );
        }
        let v = tracker.finish_at(t0 + Duration::from_millis(46)).unwrap();
        let single_speed = 20.0;
        assert!(
            v.x <= single_speed * MAX_COMPOUND_MULTIPLIER + 0.5,
            "compounded speed must be bounded: got {:.2}",
            v.x
        );
    }

    #[test]
    fn tracks_3d_pan_velocity_for_orbit_camera() {
        let mut tracker = DragVelocityTracker::<Vec3>::default();
        let t0 = Instant::now();
        let speed = Vec3::new(6.0, -3.0, 1.5);
        for i in 0..10 {
            tracker.record_at(speed * 0.008, Vec3::ZERO, t0 + Duration::from_millis(i * 8));
        }
        let v = tracker.finish_at(t0 + Duration::from_millis(76)).unwrap();
        let expected = speed * REFERENCE_DT;
        let rel_err = (v - expected).length() / expected.length();
        assert!(rel_err < 0.02, "expected {expected:?}, got {v:?}");
    }
}
