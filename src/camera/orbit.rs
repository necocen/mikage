use glam::{Mat4, Vec3};

use super::{Camera, InteractiveCamera};

/// 3D orbit camera that rotates around a target point.
///
/// Supports orbit (rotation), pan (target translation), and zoom (distance).
///
/// # Controls
/// - **Left drag**: orbit (yaw/pitch rotation)
/// - **Right drag / middle drag**: pan (translate target)
/// - **Scroll**: zoom (adjust distance)
///
/// # Damping
///
/// Set `damping` to a value between 0.0 and 1.0 to enable inertial follow.
/// - `0.0` (default): no damping, camera stops immediately when input stops
/// - `0.9`: smooth deceleration after releasing the mouse
///
/// # Example
/// ```
/// use mikage::OrbitCamera;
///
/// let mut camera = OrbitCamera::default();
/// camera.target = glam::Vec3::new(0.0, 1.0, 0.0);
/// camera.distance = 5.0;
/// camera.yaw = 0.5;
/// camera.pitch = 0.3;
/// camera.damping = 0.85; // enable smooth inertia
/// ```
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
    pub orbit_speed: f32,
    pub pan_speed: f32,
    pub zoom_speed: f32,
    pub min_distance: f32,
    pub max_distance: f32,
    pub min_pitch: f32,
    pub max_pitch: f32,
    /// Damping factor for orbit inertia (0.0 = instant stop, 0.9 = smooth).
    pub damping: f32,
    pub enabled: bool,

    // Internal velocity state for damping
    velocity_yaw: f32,
    velocity_pitch: f32,
    is_dragging: bool,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 5.0,
            yaw: 0.0,
            pitch: 0.3,
            fov_y: std::f32::consts::FRAC_PI_4,
            near: 0.01,
            far: 100.0,
            orbit_speed: 0.005,
            pan_speed: 0.005,
            zoom_speed: 0.1,
            min_distance: 0.1,
            max_distance: 100.0,
            min_pitch: -std::f32::consts::FRAC_PI_2 + 0.01,
            max_pitch: std::f32::consts::FRAC_PI_2 - 0.01,
            damping: 0.0,
            enabled: true,
            velocity_yaw: 0.0,
            velocity_pitch: 0.0,
            is_dragging: false,
        }
    }
}

impl OrbitCamera {
    fn compute_position(&self) -> Vec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.cos();
        self.target + Vec3::new(x, y, z)
    }

    fn right(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin())
    }

    fn up(&self) -> Vec3 {
        let forward = (self.target - self.compute_position()).normalize();
        self.right().cross(forward).normalize()
    }
}

impl Camera for OrbitCamera {
    fn view_matrix(&self) -> Mat4 {
        let eye = self.compute_position();
        Mat4::look_at_rh(eye, self.target, Vec3::Y)
    }

    fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect, self.near, self.far)
    }

    fn position(&self) -> Vec3 {
        self.compute_position()
    }
}

impl InteractiveCamera for OrbitCamera {
    fn on_mouse_drag(&mut self, dx: f64, dy: f64, left: bool, right: bool, middle: bool) {
        if !self.enabled {
            return;
        }

        if left && !right {
            // Orbit
            let dyaw = -dx as f32 * self.orbit_speed;
            let dpitch = dy as f32 * self.orbit_speed;
            self.yaw += dyaw;
            self.pitch += dpitch;
            self.pitch = self.pitch.clamp(self.min_pitch, self.max_pitch);
            // Track velocity for damping
            self.velocity_yaw = dyaw;
            self.velocity_pitch = dpitch;
            self.is_dragging = true;
        } else if right || middle {
            // Pan
            let right_vec = self.right();
            let up_vec = self.up();
            let pan = right_vec * (-dx as f32 * self.pan_speed * self.distance)
                + up_vec * (dy as f32 * self.pan_speed * self.distance);
            self.target += pan;
            self.is_dragging = true;
        }
    }

    fn on_scroll(&mut self, delta: f32) {
        if !self.enabled {
            return;
        }
        self.distance *= 1.0 - delta * self.zoom_speed;
        self.distance = self.distance.clamp(self.min_distance, self.max_distance);
    }

    fn on_drag_end(&mut self) {
        self.is_dragging = false;
    }

    fn update(&mut self, _dt: f32) {
        if self.damping <= 0.0 || self.is_dragging {
            return;
        }

        // Apply inertial velocity
        if self.velocity_yaw.abs() > 1e-6 || self.velocity_pitch.abs() > 1e-6 {
            self.yaw += self.velocity_yaw;
            self.pitch += self.velocity_pitch;
            self.pitch = self.pitch.clamp(self.min_pitch, self.max_pitch);

            self.velocity_yaw *= self.damping;
            self.velocity_pitch *= self.damping;
        }
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}
