use glam::{Mat4, Vec2, Vec3};

use super::{Camera, InteractiveCamera};

/// 2D orthographic camera with pan and zoom.
///
/// Projects the XY plane using an orthographic projection.
/// Provides [`viewport_bounds`](Camera2d::viewport_bounds) to query the
/// visible world-space rectangle (useful for culling or grid generation).
///
/// # Controls
/// - **Left drag**: pan
/// - **Scroll**: zoom
///
/// # Damping
///
/// Set `damping` to a value between 0.0 and 1.0 to enable inertial follow.
/// - `0.0` (default): no damping, camera stops immediately when input stops
/// - `0.9`: smooth deceleration after releasing the mouse
///
/// # Zoom smoothing
///
/// Set `zoom_smoothing` to a value between 0.0 and 1.0 to enable smooth
/// zoom animation. The value controls what fraction of the remaining zoom
/// distance is covered per 1/60s reference frame.
/// - `0.0` (default): instant zoom
/// - `0.2`: responsive smooth zoom
/// - `0.3`: noticeably smooth zoom
///
/// # Example
/// ```
/// use mikage::Camera2d;
///
/// let mut camera = Camera2d::default();
/// camera.position = glam::Vec2::new(0.0, 0.0);
/// camera.zoom = 1.0;
/// camera.damping = 0.85;
/// camera.zoom_smoothing = 0.2;
/// ```
pub struct Camera2d {
    /// Camera center in world coordinates.
    pub position: Vec2,
    /// Zoom level (1.0 = one world unit per viewport half-height).
    pub zoom: f32,
    /// Near clip plane. Default: -1.0.
    pub near: f32,
    /// Far clip plane. Default: 1.0.
    pub far: f32,
    /// Zoom speed multiplier.
    pub zoom_speed: f32,
    /// Minimum zoom level.
    pub min_zoom: f32,
    /// Maximum zoom level.
    pub max_zoom: f32,
    /// Damping factor for pan inertia (0.0 = instant stop, 0.9 = smooth).
    pub damping: f32,
    /// Smoothing factor for zoom animation (0.0 = instant, 0.2–0.3 = smooth).
    /// Controls what fraction of the remaining zoom distance is covered per
    /// 1/60s reference frame. Frame-rate independent.
    pub zoom_smoothing: f32,

    /// Whether camera input is enabled. Use [`set_enabled`](InteractiveCamera::set_enabled)
    /// / [`is_enabled`](InteractiveCamera::is_enabled) to toggle.
    /// When disabled, input events are ignored but inertial motion continues.
    enabled: bool,

    // Internal state
    velocity: Vec2,
    is_dragging: bool,
    /// Viewport width in physical pixels (updated via `set_viewport_size`).
    viewport_width: f32,
    /// Viewport height in physical pixels (updated via `set_viewport_size`).
    viewport_height: f32,
    cursor_pos: Vec2,
    /// Target zoom for smooth animation. `None` when no animation is active.
    target_zoom: Option<f32>,
    /// Screen-space anchor point for zoom animation (preserves world point here).
    zoom_anchor: Vec2,
}

impl Default for Camera2d {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            zoom: 1.0,
            near: -1.0,
            far: 1.0,
            zoom_speed: 0.1,
            min_zoom: 0.01,
            max_zoom: 1000.0,
            damping: 0.0,
            zoom_smoothing: 0.0,
            enabled: true,
            velocity: Vec2::ZERO,
            is_dragging: false,
            viewport_width: 1280.0,
            viewport_height: 720.0,
            cursor_pos: Vec2::ZERO,
            target_zoom: None,
            zoom_anchor: Vec2::new(640.0, 360.0),
        }
    }
}

impl Camera2d {
    /// Returns the world-space bounds of the current viewport.
    ///
    /// Returns `(min, max)` in world coordinates where `min` is the
    /// bottom-left corner and `max` is the top-right corner.
    pub fn viewport_bounds(&self, aspect: f32) -> (Vec2, Vec2) {
        let half_w = aspect / self.zoom;
        let half_h = 1.0 / self.zoom;
        let min = self.position - Vec2::new(half_w, half_h);
        let max = self.position + Vec2::new(half_w, half_h);
        (min, max)
    }

    /// Zooms around a screen-space point (pixels, top-left origin),
    /// preserving the world coordinate under that point.
    ///
    /// When `zoom_smoothing > 0`, sets `target_zoom` and defers the actual
    /// zoom change to [`update`](InteractiveCamera::update). Otherwise applies instantly.
    fn zoom_around(&mut self, delta: f32, origin: Vec2) {
        let base = self.target_zoom.unwrap_or(self.zoom);
        let new_target =
            (base * (1.0 + delta * self.zoom_speed)).clamp(self.min_zoom, self.max_zoom);
        self.zoom_anchor = origin;

        if self.zoom_smoothing <= 0.0 {
            // Instant zoom
            let before = self.screen_to_world(origin);
            self.zoom = new_target;
            let after = self.screen_to_world(origin);
            self.position += before - after;
            self.target_zoom = None;
        } else {
            self.target_zoom = Some(new_target);
        }
    }

    /// Converts a screen-space point (pixels, top-left origin) to world coordinates.
    fn screen_to_world(&self, screen: Vec2) -> Vec2 {
        let aspect = self.viewport_width / self.viewport_height;
        let ndc_x = (screen.x / self.viewport_width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen.y / self.viewport_height) * 2.0;
        Vec2::new(
            ndc_x * (aspect / self.zoom) + self.position.x,
            ndc_y * (1.0 / self.zoom) + self.position.y,
        )
    }
}

impl Camera for Camera2d {
    fn view_matrix(&self) -> Mat4 {
        Mat4::from_translation(-self.position.extend(0.0))
    }

    fn projection_matrix(&self, aspect: f32) -> Mat4 {
        let half_w = aspect / self.zoom;
        let half_h = 1.0 / self.zoom;
        Mat4::orthographic_rh(-half_w, half_w, -half_h, half_h, self.near, self.far)
    }

    fn position(&self) -> Vec3 {
        self.position.extend(0.0)
    }
}

/// EMA retain factor for velocity smoothing during drag.
/// Higher values weight previous velocity more, producing a smoother
/// direction estimate at the cost of responsiveness.
/// 0.8 gives a half-life of ~3 events (~50ms at 60fps).
const VELOCITY_SMOOTHING: f32 = 0.8;

impl InteractiveCamera for Camera2d {
    fn on_mouse_drag(&mut self, dx: f64, dy: f64, left: bool, _right: bool, _middle: bool) {
        if !self.enabled {
            return;
        }

        if left {
            // Clear residual velocity from previous fling on drag start.
            if !self.is_dragging {
                self.velocity = Vec2::ZERO;
                self.is_dragging = true;
            }
            // Pan: convert pixel delta to world units.
            // The orthographic projection maps viewport_height pixels to 2/zoom world units.
            let world_per_px = 2.0 / (self.viewport_height * self.zoom);
            let delta = Vec2::new(-dx as f32 * world_per_px, dy as f32 * world_per_px);
            self.position += delta;
            // Exponential moving average for smoother inertia direction on release.
            self.velocity = self.velocity * VELOCITY_SMOOTHING + delta * (1.0 - VELOCITY_SMOOTHING);
        }
    }

    fn on_scroll(&mut self, delta: f32) {
        if !self.enabled {
            return;
        }
        self.zoom_around(delta, self.cursor_pos);
    }

    fn set_viewport_size(&mut self, width: u32, height: u32) {
        self.viewport_width = width as f32;
        self.viewport_height = height as f32;
    }

    fn set_cursor_position(&mut self, x: f64, y: f64) {
        self.cursor_pos = Vec2::new(x as f32, y as f32);
    }

    fn on_pinch_pan(
        &mut self,
        zoom_delta: f32,
        pan_dx: f64,
        pan_dy: f64,
        focus: Option<(f64, f64)>,
    ) {
        if !self.enabled {
            return;
        }
        if zoom_delta.abs() > 1e-4 {
            let origin = focus
                .map(|(x, y)| Vec2::new(x as f32, y as f32))
                .unwrap_or(self.cursor_pos);
            self.zoom_around(zoom_delta, origin);
        }
        if pan_dx.abs() > 0.5 || pan_dy.abs() > 0.5 {
            // Camera2d pans on left drag, not right drag.
            self.on_mouse_drag(pan_dx, pan_dy, true, false, false);
        }
    }

    fn on_drag_end(&mut self) {
        self.is_dragging = false;
    }

    fn update(&mut self, dt: f32) {
        const REFERENCE_DT: f32 = 1.0 / 60.0;

        // Smooth zoom animation (runs regardless of drag/damping state).
        if let Some(target) = self.target_zoom {
            let gap = self.zoom - target;
            if gap.abs() > 1e-6 {
                let before = self.screen_to_world(self.zoom_anchor);
                let factor = dt / REFERENCE_DT;
                let retain = (1.0 - self.zoom_smoothing).powf(factor);
                self.zoom = target + gap * retain;
                let after = self.screen_to_world(self.zoom_anchor);
                self.position += before - after;
            }
            if (self.zoom - target).abs() <= 1e-6 {
                self.zoom = target;
                self.target_zoom = None;
            }
        }

        // Pan inertia
        if self.damping <= 0.0 || self.is_dragging {
            return;
        }

        // Apply inertial velocity (frame-rate independent).
        // velocity is a per-reference-frame delta (reference = 60fps).
        // Use exact geometric series to compute displacement over `factor`
        // reference frames, ensuring identical results at any frame rate.
        if self.velocity.length_squared() > 1e-12 {
            let factor = dt / REFERENCE_DT;
            let decay = self.damping.powf(factor);
            let movement_scale = if (1.0 - self.damping).abs() > 1e-6 {
                (1.0 - decay) / (1.0 - self.damping)
            } else {
                factor // limit as damping → 1.0
            };
            self.position += self.velocity * movement_scale;
            self.velocity *= decay;
        }
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;
    use glam::{Mat4, Vec2, Vec3};

    macro_rules! assert_approx {
        ($a:expr, $b:expr) => {
            assert!(
                ($a - $b).abs() < 1e-5,
                "{} ≠ {} (diff={})",
                $a,
                $b,
                ($a - $b).abs()
            );
        };
    }

    #[test]
    fn default_values() {
        let cam = Camera2d::default();
        assert_eq!(cam.position, Vec2::ZERO);
        assert_approx!(cam.zoom, 1.0);
        assert_approx!(cam.near, -1.0);
        assert_approx!(cam.far, 1.0);
        assert!(cam.enabled);
        assert_approx!(cam.damping, 0.0);
        assert_approx!(cam.viewport_height, 720.0);
    }

    #[test]
    fn position_returns_vec3() {
        let mut cam = Camera2d::default();
        cam.position = Vec2::new(3.0, 4.0);
        assert_eq!(cam.position(), Vec3::new(3.0, 4.0, 0.0));
    }

    #[test]
    fn view_matrix_is_translation() {
        let mut cam = Camera2d::default();
        cam.position = Vec2::new(3.0, 4.0);
        let expected = Mat4::from_translation(-cam.position.extend(0.0));
        assert_eq!(cam.view_matrix(), expected);
    }

    #[test]
    fn projection_matrix_orthographic() {
        let cam = Camera2d::default(); // zoom=1.0
        let aspect = 2.0_f32;
        let expected = Mat4::orthographic_rh(-2.0, 2.0, -1.0, 1.0, -1.0, 1.0);
        assert_eq!(cam.projection_matrix(aspect), expected);
    }

    #[test]
    fn view_projection_composition() {
        let mut cam = Camera2d::default();
        cam.position = Vec2::new(1.0, 2.0);
        let aspect = 1.5_f32;
        let expected = cam.projection_matrix(aspect) * cam.view_matrix();
        assert_eq!(cam.view_projection_matrix(aspect), expected);
    }

    #[test]
    fn viewport_bounds_default() {
        let cam = Camera2d::default(); // position=0, zoom=1
        let (min, max) = cam.viewport_bounds(1.0);
        assert_approx!(min.x, -1.0);
        assert_approx!(min.y, -1.0);
        assert_approx!(max.x, 1.0);
        assert_approx!(max.y, 1.0);
    }

    #[test]
    fn viewport_bounds_with_zoom() {
        let mut cam = Camera2d::default();
        cam.zoom = 2.0;
        let (min, max) = cam.viewport_bounds(1.0);
        assert_approx!(min.x, -0.5);
        assert_approx!(min.y, -0.5);
        assert_approx!(max.x, 0.5);
        assert_approx!(max.y, 0.5);
    }

    #[test]
    fn viewport_bounds_with_offset() {
        let mut cam = Camera2d::default();
        cam.position = Vec2::new(5.0, 3.0);
        cam.zoom = 1.0;
        let (min, max) = cam.viewport_bounds(1.0);
        assert_approx!(min.x, 4.0);
        assert_approx!(min.y, 2.0);
        assert_approx!(max.x, 6.0);
        assert_approx!(max.y, 4.0);
    }

    #[test]
    fn viewport_bounds_aspect_ratio() {
        let cam = Camera2d::default(); // position=0, zoom=1
        let (min, max) = cam.viewport_bounds(2.0);
        assert_approx!(min.x, -2.0);
        assert_approx!(min.y, -1.0);
        assert_approx!(max.x, 2.0);
        assert_approx!(max.y, 1.0);
    }

    #[test]
    fn pan_drag_moves_position() {
        let mut cam = Camera2d::default();
        let before = cam.position;
        cam.on_mouse_drag(100.0, 50.0, true, false, false);
        // Pan inverts: dx>0 → position.x decreases, dy>0 → position.y increases
        assert!(cam.position.x < before.x);
        assert!(cam.position.y > before.y);
    }

    #[test]
    fn pan_pixel_to_world_accuracy() {
        // A 1600x800 window with zoom=1 has visible height = 2 world units.
        // 1 pixel = 2/(800*1) = 0.0025 world units.
        let mut cam = Camera2d::default();
        cam.set_viewport_size(1600, 800);

        cam.on_mouse_drag(400.0, 0.0, true, false, false);
        // 400 px * 0.0025 = 1.0 world unit to the left
        assert_approx!(cam.position.x, -1.0);
        assert_approx!(cam.position.y, 0.0);
    }

    #[test]
    fn pan_accuracy_with_zoom() {
        // zoom=2: 1 pixel = 2/(800*2) = 0.00125 world units
        let mut cam = Camera2d::default();
        cam.set_viewport_size(1600, 800);
        cam.zoom = 2.0;

        cam.on_mouse_drag(0.0, 800.0, true, false, false);
        // 800 px * 0.00125 = 1.0 world unit upward
        assert_approx!(cam.position.x, 0.0);
        assert_approx!(cam.position.y, 1.0);
    }

    #[test]
    fn scroll_changes_zoom() {
        let mut cam = Camera2d::default();
        let initial_zoom = cam.zoom;
        cam.on_scroll(1.0);
        assert!(cam.zoom > initial_zoom);

        // Scroll down repeatedly should clamp at min_zoom
        for _ in 0..1000 {
            cam.on_scroll(-100.0);
        }
        assert_approx!(cam.zoom, cam.min_zoom);
    }

    #[test]
    fn zoom_to_cursor_preserves_world_point() {
        let mut cam = Camera2d::default();
        cam.set_viewport_size(1600, 800);
        // Place cursor at top-right quadrant
        cam.set_cursor_position(1200.0, 200.0);

        let world_before = cam.screen_to_world(cam.cursor_pos);
        cam.on_scroll(1.0); // zoom in
        let world_after = cam.screen_to_world(cam.cursor_pos);

        // The world point under the cursor should be preserved
        assert_approx!(world_before.x, world_after.x);
        assert_approx!(world_before.y, world_after.y);
    }

    #[test]
    fn zoom_at_center_does_not_move_position() {
        let mut cam = Camera2d::default();
        cam.set_viewport_size(1600, 800);
        // Cursor at window center
        cam.set_cursor_position(800.0, 400.0);

        let pos_before = cam.position;
        cam.on_scroll(1.0);
        // Zooming at center should not move the camera
        assert_approx!(cam.position.x, pos_before.x);
        assert_approx!(cam.position.y, pos_before.y);
    }

    #[test]
    fn disabled_ignores_input() {
        let mut cam = Camera2d::default();
        cam.set_enabled(false);
        let pos_before = cam.position;
        let zoom_before = cam.zoom;
        cam.on_mouse_drag(100.0, 50.0, true, false, false);
        cam.on_scroll(1.0);
        assert_eq!(cam.position, pos_before);
        assert_approx!(cam.zoom, zoom_before);
    }

    #[test]
    fn damping_decays_velocity() {
        let mut cam = Camera2d::default();
        cam.damping = 0.9;

        // Perform a drag to set velocity
        cam.on_mouse_drag(100.0, 0.0, true, false, false);
        cam.on_drag_end();

        assert!(cam.velocity.length_squared() > 0.0);

        // Simulate several update ticks; velocity should decay each time
        let mut prev_speed = cam.velocity.length();
        for _ in 0..100 {
            cam.update(1.0 / 60.0);
            let speed = cam.velocity.length();
            assert!(speed < prev_speed);
            prev_speed = speed;
        }
        // After many iterations velocity should be very small
        assert!(cam.velocity.length() < 1e-4);
    }

    #[test]
    fn damping_is_framerate_independent() {
        // Same initial velocity, different frame rates → same final position after 1 second
        let make_cam = || {
            let mut cam = Camera2d::default();
            cam.damping = 0.9;
            cam.set_viewport_size(1600, 800);
            cam.on_mouse_drag(100.0, 0.0, true, false, false);
            cam.on_drag_end();
            cam
        };

        // 60 fps
        let mut cam60 = make_cam();
        for _ in 0..60 {
            cam60.update(1.0 / 60.0);
        }

        // 30 fps
        let mut cam30 = make_cam();
        for _ in 0..30 {
            cam30.update(1.0 / 30.0);
        }

        // Positions should be approximately equal (within 1% of the 60fps position)
        let diff = (cam60.position - cam30.position).length();
        let magnitude = cam60.position.length().max(0.001);
        assert!(
            diff / magnitude < 0.01,
            "30fps pos {:?} vs 60fps pos {:?}, diff={diff}",
            cam30.position,
            cam60.position
        );
    }

    #[test]
    fn set_viewport_size_updates_fields() {
        let mut cam = Camera2d::default();
        cam.set_viewport_size(1920, 1080);
        assert_approx!(cam.viewport_width, 1920.0);
        assert_approx!(cam.viewport_height, 1080.0);
    }

    #[test]
    fn pinch_pan_moves_camera2d() {
        let mut cam = Camera2d::default();
        cam.set_viewport_size(1600, 800);
        let before = cam.position;
        // Two-finger pan: no zoom, just pan, no focal point
        cam.on_pinch_pan(0.0, 100.0, 0.0, None);
        // Should have moved (left drag path)
        assert!(cam.position.x < before.x, "pinch pan should move camera");
    }

    #[test]
    fn pinch_zoom_uses_focal_point() {
        let mut cam = Camera2d::default();
        cam.set_viewport_size(1600, 800);
        let focus = (1200.0, 200.0);
        let focus_screen = Vec2::new(focus.0 as f32, focus.1 as f32);
        let world_before = cam.screen_to_world(focus_screen);
        // Two-finger zoom with focal point
        cam.on_pinch_pan(1.0, 0.0, 0.0, Some(focus));
        let world_after = cam.screen_to_world(focus_screen);
        // World point under focal point should be preserved
        assert_approx!(world_before.x, world_after.x);
        assert_approx!(world_before.y, world_after.y);
    }

    #[test]
    fn pinch_zoom_without_focus_uses_cursor() {
        let mut cam = Camera2d::default();
        cam.set_viewport_size(1600, 800);
        cam.set_cursor_position(1200.0, 200.0);
        let world_before = cam.screen_to_world(cam.cursor_pos);
        // Two-finger zoom without focal point falls back to cursor
        cam.on_pinch_pan(1.0, 0.0, 0.0, None);
        let world_after = cam.screen_to_world(cam.cursor_pos);
        assert_approx!(world_before.x, world_after.x);
        assert_approx!(world_before.y, world_after.y);
    }

    #[test]
    fn smooth_zoom_converges() {
        let mut cam = Camera2d::default();
        cam.set_viewport_size(1600, 800);
        cam.zoom_smoothing = 0.2;
        cam.set_cursor_position(800.0, 400.0);

        cam.on_scroll(1.0);
        let target = cam.target_zoom.unwrap();
        assert!(
            target > cam.zoom,
            "target should be higher than current zoom"
        );

        // After many frames, zoom should converge to target
        for _ in 0..120 {
            cam.update(1.0 / 60.0);
        }
        assert_approx!(cam.zoom, target);
        assert!(cam.target_zoom.is_none(), "animation should have completed");
    }

    #[test]
    fn smooth_zoom_preserves_anchor_point() {
        let mut cam = Camera2d::default();
        cam.set_viewport_size(1600, 800);
        cam.zoom_smoothing = 0.2;
        cam.set_cursor_position(1200.0, 200.0);

        let anchor = cam.cursor_pos;
        let world_before = cam.screen_to_world(anchor);
        cam.on_scroll(2.0);

        // Animate to completion
        for _ in 0..120 {
            cam.update(1.0 / 60.0);
        }

        let world_after = cam.screen_to_world(anchor);
        assert_approx!(world_before.x, world_after.x);
        assert_approx!(world_before.y, world_after.y);
    }

    #[test]
    fn smooth_zoom_is_framerate_independent() {
        let make_cam = || {
            let mut cam = Camera2d::default();
            cam.set_viewport_size(1600, 800);
            cam.zoom_smoothing = 0.2;
            cam.set_cursor_position(800.0, 400.0);
            cam.on_scroll(2.0);
            cam
        };

        let mut cam60 = make_cam();
        for _ in 0..60 {
            cam60.update(1.0 / 60.0);
        }

        let mut cam30 = make_cam();
        for _ in 0..30 {
            cam30.update(1.0 / 30.0);
        }

        let diff = (cam60.zoom - cam30.zoom).abs();
        let magnitude = cam60.zoom.max(0.001);
        assert!(
            diff / magnitude < 0.01,
            "30fps zoom {} vs 60fps zoom {}, diff={diff}",
            cam30.zoom,
            cam60.zoom
        );
    }

    #[test]
    fn velocity_ema_smooths_direction() {
        let mut cam = Camera2d::default();
        cam.set_viewport_size(1600, 800);
        cam.damping = 0.9;

        // Simulate diagonal drag: mostly (100, 100) deltas
        for _ in 0..10 {
            cam.on_mouse_drag(100.0, 100.0, true, false, false);
        }
        // Last event with horizontal jitter (mouse deceleration artifact)
        cam.on_mouse_drag(50.0, 2.0, true, false, false);
        cam.on_drag_end();

        // Velocity direction should still be roughly diagonal, not horizontal.
        // arctan(vy/vx) should be > 20 degrees (0.35 rad).
        let angle = cam.velocity.y.atan2(cam.velocity.x.abs());
        assert!(
            angle.abs() > 0.35,
            "velocity angle {angle:.3} rad is too horizontal; velocity = {:?}",
            cam.velocity
        );
    }

    #[test]
    fn direct_zoom_set_not_animated_back() {
        // Setting zoom directly should not cause animation back to default
        let mut cam = Camera2d::default();
        cam.zoom_smoothing = 0.2;
        cam.zoom = 5.0;

        cam.update(1.0 / 60.0);
        // target_zoom is None, so no animation should occur
        assert_approx!(cam.zoom, 5.0);
    }
}
