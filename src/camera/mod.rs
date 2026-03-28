//! Camera system.
//!
//! Provides the [`Camera`] trait (read-only view/projection) and
//! [`InteractiveCamera`] trait (input handling). The built-in [`OrbitCamera`]
//! and [`Camera2d`] implement both.
//!
//! [`App::Camera`](crate::App::Camera) requires [`InteractiveCamera`].
//! For a camera that ignores all input, implement `InteractiveCamera` with
//! no-op methods (the framework calls them automatically).

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

    /// Called on a one-finger touch drag. Default: forwards as left-button mouse drag.
    fn on_touch_drag(&mut self, dx: f64, dy: f64) {
        self.on_mouse_drag(dx, dy, true, false, false);
    }

    /// Called when a one-finger touch drag ends. Default: forwards to [`on_drag_end`](Self::on_drag_end).
    fn on_touch_drag_end(&mut self) {
        self.on_drag_end();
    }

    /// Called on a two-finger pinch/pan gesture (touch or trackpad).
    ///
    /// `zoom_delta`: positive = zoom in, negative = zoom out.
    /// `pan_dx`/`pan_dy`: pixel deltas of the gesture midpoint.
    /// Default: forwards zoom to [`on_scroll`](Self::on_scroll) and pan as right-button drag.
    fn on_pinch_pan(&mut self, zoom_delta: f32, pan_dx: f64, pan_dy: f64) {
        if zoom_delta.abs() > 1e-4 {
            self.on_scroll(zoom_delta);
        }
        if pan_dx.abs() > 0.5 || pan_dy.abs() > 0.5 {
            self.on_mouse_drag(pan_dx, pan_dy, false, true, false);
        }
    }

    /// Notifies the camera of the current viewport size in physical pixels.
    ///
    /// Called on window resize. Default implementation does nothing.
    fn set_viewport_size(&mut self, _width: u32, _height: u32) {}

    /// Notifies the camera of the current cursor position in physical pixels (top-left origin).
    ///
    /// Called on every cursor move. Default implementation does nothing.
    fn set_cursor_position(&mut self, _x: f64, _y: f64) {}

    /// Enables or disables camera input.
    fn set_enabled(&mut self, enabled: bool);

    /// Returns whether camera input is enabled.
    fn is_enabled(&self) -> bool;
}
