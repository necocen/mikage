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
/// # Example
/// ```
/// use mikage::Camera2d;
///
/// let mut camera = Camera2d::default();
/// camera.position = glam::Vec2::new(0.0, 0.0);
/// camera.zoom = 1.0;
/// camera.damping = 0.85;
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
    /// Pan speed multiplier.
    pub pan_speed: f32,
    /// Zoom speed multiplier.
    pub zoom_speed: f32,
    /// Minimum zoom level.
    pub min_zoom: f32,
    /// Maximum zoom level.
    pub max_zoom: f32,
    /// Damping factor for pan inertia (0.0 = instant stop, 0.9 = smooth).
    pub damping: f32,
    /// Whether camera input is enabled.
    pub enabled: bool,

    // Internal velocity state for damping
    velocity: Vec2,
    is_dragging: bool,
}

impl Default for Camera2d {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            zoom: 1.0,
            near: -1.0,
            far: 1.0,
            pan_speed: 0.005,
            zoom_speed: 0.1,
            min_zoom: 0.01,
            max_zoom: 1000.0,
            damping: 0.0,
            enabled: true,
            velocity: Vec2::ZERO,
            is_dragging: false,
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

impl InteractiveCamera for Camera2d {
    fn on_mouse_drag(&mut self, dx: f64, dy: f64, left: bool, _right: bool, _middle: bool) {
        if !self.enabled {
            return;
        }

        if left {
            // Pan: move camera opposite to drag direction
            let delta = Vec2::new(
                -dx as f32 * self.pan_speed / self.zoom,
                dy as f32 * self.pan_speed / self.zoom,
            );
            self.position += delta;
            self.velocity = delta;
            self.is_dragging = true;
        }
    }

    fn on_scroll(&mut self, delta: f32) {
        if !self.enabled {
            return;
        }
        self.zoom *= 1.0 + delta * self.zoom_speed;
        self.zoom = self.zoom.clamp(self.min_zoom, self.max_zoom);
    }

    fn on_drag_end(&mut self) {
        self.is_dragging = false;
    }

    fn update(&mut self, _dt: f32) {
        if self.damping <= 0.0 || self.is_dragging {
            return;
        }

        // Apply inertial velocity
        if self.velocity.length_squared() > 1e-12 {
            self.position += self.velocity;
            self.velocity *= self.damping;
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
    fn disabled_ignores_input() {
        let mut cam = Camera2d::default();
        cam.enabled = false;
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
}
