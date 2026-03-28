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

    fn update(&mut self, dt: f32) {
        if self.damping <= 0.0 || self.is_dragging {
            return;
        }

        // Apply inertial velocity (frame-rate independent).
        // Velocities are per-reference-frame deltas (reference = 60fps).
        // Use exact geometric series to compute displacement over `factor`
        // reference frames, ensuring identical results at any frame rate.
        if self.velocity_yaw.abs() > 1e-6 || self.velocity_pitch.abs() > 1e-6 {
            const REFERENCE_DT: f32 = 1.0 / 60.0;
            let factor = dt / REFERENCE_DT;
            let decay = self.damping.powf(factor);
            let movement_scale = if (1.0 - self.damping).abs() > 1e-6 {
                (1.0 - decay) / (1.0 - self.damping)
            } else {
                factor // limit as damping → 1.0
            };
            self.yaw += self.velocity_yaw * movement_scale;
            self.pitch += self.velocity_pitch * movement_scale;
            self.pitch = self.pitch.clamp(self.min_pitch, self.max_pitch);

            self.velocity_yaw *= decay;
            self.velocity_pitch *= decay;
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
    use glam::{Mat4, Vec3};
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

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
        let cam = OrbitCamera::default();
        assert_approx!(cam.distance, 5.0);
        assert_approx!(cam.yaw, 0.0);
        assert_approx!(cam.pitch, 0.3);
        assert_approx!(cam.fov_y, FRAC_PI_4);
        assert!(
            (cam.target - Vec3::ZERO).length() < 1e-5,
            "target should be ZERO"
        );
        assert!(cam.enabled);
    }

    #[test]
    fn position_at_default() {
        let cam = OrbitCamera::default();
        let pos = cam.position();
        let expected = Vec3::new(0.0, 5.0 * 0.3_f32.sin(), 5.0 * 0.3_f32.cos());
        assert!(
            (pos - expected).length() < 1e-5,
            "position {pos} ≠ expected {expected}"
        );
    }

    #[test]
    fn position_at_yaw_pi_half() {
        let mut cam = OrbitCamera::default();
        cam.yaw = FRAC_PI_2;
        let pos = cam.position();
        let expected_x = cam.distance * cam.pitch.cos();
        let expected_y = cam.distance * cam.pitch.sin();
        assert_approx!(pos.x, expected_x);
        assert_approx!(pos.y, expected_y);
        assert!(pos.z.abs() < 1e-5, "z should be ≈0, got {}", pos.z);
    }

    #[test]
    fn position_with_nonzero_target() {
        let mut cam = OrbitCamera::default();
        cam.target = Vec3::new(1.0, 2.0, 3.0);
        let pos = cam.position();

        // Spherical offset from compute_position with yaw=0, pitch=0.3, distance=5
        let offset = Vec3::new(
            cam.distance * cam.pitch.cos() * cam.yaw.sin(),
            cam.distance * cam.pitch.sin(),
            cam.distance * cam.pitch.cos() * cam.yaw.cos(),
        );
        let expected = cam.target + offset;
        assert!(
            (pos - expected).length() < 1e-5,
            "position {pos} ≠ expected {expected}"
        );
    }

    #[test]
    fn view_matrix_looks_at_target() {
        let cam = OrbitCamera::default();
        let view = cam.view_matrix();
        let expected = Mat4::look_at_rh(cam.position(), cam.target, Vec3::Y);
        let a = view.to_cols_array();
        let b = expected.to_cols_array();
        for i in 0..16 {
            assert_approx!(a[i], b[i]);
        }
    }

    #[test]
    fn projection_matrix_perspective() {
        let cam = OrbitCamera::default();
        let aspect = 1.5_f32;
        let proj = cam.projection_matrix(aspect);
        let expected = Mat4::perspective_rh(cam.fov_y, aspect, cam.near, cam.far);
        let a = proj.to_cols_array();
        let b = expected.to_cols_array();
        for i in 0..16 {
            assert_approx!(a[i], b[i]);
        }
    }

    #[test]
    fn view_projection_composition() {
        let cam = OrbitCamera::default();
        let aspect = 1.5_f32;
        let vp = cam.view_projection_matrix(aspect);
        let expected = cam.projection_matrix(aspect) * cam.view_matrix();
        let a = vp.to_cols_array();
        let b = expected.to_cols_array();
        for i in 0..16 {
            assert_approx!(a[i], b[i]);
        }
    }

    #[test]
    fn orbit_drag_updates_yaw_pitch() {
        let mut cam = OrbitCamera::default();
        let yaw_before = cam.yaw;
        let pitch_before = cam.pitch;
        let speed = cam.orbit_speed;

        cam.on_mouse_drag(100.0, 50.0, true, false, false);

        assert_approx!(cam.yaw, yaw_before + (-100.0 * speed));
        assert_approx!(cam.pitch, pitch_before + (50.0 * speed));
    }

    #[test]
    fn pitch_clamped_on_drag() {
        // Clamp to max_pitch with large positive dy
        let mut cam = OrbitCamera::default();
        cam.on_mouse_drag(0.0, 100_000.0, true, false, false);
        assert_approx!(cam.pitch, cam.max_pitch);

        // Clamp to min_pitch with large negative dy
        let mut cam = OrbitCamera::default();
        cam.on_mouse_drag(0.0, -100_000.0, true, false, false);
        assert_approx!(cam.pitch, cam.min_pitch);
    }

    #[test]
    fn pan_drag_moves_target() {
        let mut cam = OrbitCamera::default();
        let target_before = cam.target;

        cam.on_mouse_drag(10.0, 10.0, false, true, false);

        assert!(
            (cam.target - target_before).length() > 1e-6,
            "target should have moved after pan drag"
        );
    }

    #[test]
    fn scroll_zooms_distance() {
        let mut cam = OrbitCamera::default();
        let d0 = cam.distance;

        cam.on_scroll(1.0);
        assert!(
            cam.distance < d0,
            "scroll +1 should zoom in (decrease distance)"
        );

        let d1 = cam.distance;
        cam.on_scroll(-1.0);
        assert!(
            cam.distance > d1,
            "scroll -1 should zoom out (increase distance)"
        );
    }

    #[test]
    fn scroll_distance_clamped() {
        // Clamp at min_distance
        let mut cam = OrbitCamera::default();
        cam.distance = cam.min_distance + 0.01;
        cam.on_scroll(100.0);
        assert_approx!(cam.distance, cam.min_distance);

        // Clamp at max_distance
        let mut cam = OrbitCamera::default();
        cam.distance = cam.max_distance - 0.01;
        cam.on_scroll(-100.0);
        assert_approx!(cam.distance, cam.max_distance);
    }

    #[test]
    fn disabled_camera_ignores_input() {
        let mut cam = OrbitCamera::default();
        cam.set_enabled(false);

        let yaw = cam.yaw;
        let pitch = cam.pitch;
        let distance = cam.distance;
        let target = cam.target;

        cam.on_mouse_drag(100.0, 100.0, true, false, false);
        cam.on_scroll(5.0);

        assert_approx!(cam.yaw, yaw);
        assert_approx!(cam.pitch, pitch);
        assert_approx!(cam.distance, distance);
        assert!((cam.target - target).length() < 1e-5);
    }

    #[test]
    fn damping_inertia_decays() {
        let mut cam = OrbitCamera::default();
        cam.damping = 0.9;

        // Perform a drag to build velocity
        cam.on_mouse_drag(100.0, 0.0, true, false, false);
        let velocity_after_drag = cam.velocity_yaw;
        assert!(
            velocity_after_drag.abs() > 1e-6,
            "should have velocity after drag"
        );

        cam.on_drag_end();
        assert!(!cam.is_dragging);

        // Accumulate yaw changes over several update frames
        let yaw_after_drag = cam.yaw;
        for _ in 0..5 {
            cam.update(0.016);
        }

        // Yaw should have continued to change
        assert!(
            (cam.yaw - yaw_after_drag).abs() > 1e-6,
            "yaw should keep changing due to inertia"
        );

        // Velocity should have decayed
        assert!(
            cam.velocity_yaw.abs() < velocity_after_drag.abs(),
            "velocity should decay over time"
        );
    }

    #[test]
    fn no_damping_stops_immediately() {
        let mut cam = OrbitCamera::default();
        // damping is 0.0 by default

        cam.on_mouse_drag(100.0, 50.0, true, false, false);
        cam.on_drag_end();

        let yaw_after = cam.yaw;
        let pitch_after = cam.pitch;

        cam.update(0.016);
        cam.update(0.016);

        assert_approx!(cam.yaw, yaw_after);
        assert_approx!(cam.pitch, pitch_after);
    }

    #[test]
    fn enabled_toggle() {
        let mut cam = OrbitCamera::default();
        assert!(cam.is_enabled());

        cam.set_enabled(false);
        assert!(!cam.is_enabled());

        cam.set_enabled(true);
        assert!(cam.is_enabled());
    }
}
