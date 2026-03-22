use glam::{Mat4, Vec2, Vec3};

use super::{Camera, CameraController};

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

impl CameraController for Camera2d {
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
