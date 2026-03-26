//! Camera system.
//!
//! Provides the [`Camera`] trait (read-only view/projection) and
//! [`InteractiveCamera`] trait (input handling). The built-in [`OrbitCamera`]
//! implements both.
//!
//! To use a custom camera, implement [`InteractiveCamera`] (which extends
//! [`Camera`]) and pass it via [`RunConfig::camera`](crate::RunConfig::camera).
//! If you only need a read-only camera (no framework-driven input), implement
//! [`Camera`] alone and handle input yourself in [`App::update`](crate::App::update).

pub mod camera2d;
pub mod orbit;

pub use camera2d::Camera2d;
pub use orbit::OrbitCamera;

/// Read-only camera interface: view and projection matrices.
///
/// This trait is exposed to [`App::encode`](crate::App::encode) via
/// [`FrameContext::camera`](crate::FrameContext::camera).
pub trait Camera: Send + Sync {
    /// Returns the view matrix (world-to-camera transform).
    fn view_matrix(&self) -> glam::Mat4;

    /// Returns the projection matrix. `aspect` is the window width/height ratio.
    fn projection_matrix(&self, aspect: f32) -> glam::Mat4;

    /// Returns the combined view-projection matrix. Default: projection * view.
    fn view_projection_matrix(&self, aspect: f32) -> glam::Mat4 {
        self.projection_matrix(aspect) * self.view_matrix()
    }

    /// Returns the camera position in world space.
    fn position(&self) -> glam::Vec3;
}

/// Camera input controller.
///
/// Extends [`Camera`] with mouse/scroll/touch input handling and per-frame
/// updates. The framework calls these methods automatically, forwarding
/// input events unless egui is capturing them.
///
/// Exposed to [`App::update`](crate::App::update) via
/// [`UpdateContext::camera`](crate::UpdateContext::camera).
pub trait InteractiveCamera: Camera {
    /// Called on mouse drag. `dx`/`dy` are pixel deltas.
    fn on_mouse_drag(&mut self, dx: f64, dy: f64, left: bool, right: bool, middle: bool);

    /// Called on scroll/zoom. Positive values zoom in.
    fn on_scroll(&mut self, delta: f32);

    /// Called every frame. Use for damping or animation.
    fn update(&mut self, dt: f32);

    /// Called when a mouse drag ends (button released).
    ///
    /// Use this to transition from active dragging to inertial motion.
    /// Default implementation does nothing.
    fn on_drag_end(&mut self) {}

    /// Enables or disables camera input.
    fn set_enabled(&mut self, enabled: bool);

    /// Returns whether camera input is enabled.
    fn is_enabled(&self) -> bool;
}
