use std::collections::HashSet;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::KeyCode;

/// Per-frame input state.
///
/// Tracks keyboard, mouse, and scroll state. Updated automatically by the
/// framework. Access via [`UpdateContext::input`](crate::UpdateContext::input).
///
/// Events consumed by egui (e.g. keyboard input while a text field is focused,
/// or pointer input over an egui window) are **not** reflected here. This
/// ensures that `update()` and `on_window_event()` see a consistent,
/// egui-filtered view of input.
///
/// `keys_pressed` / `mouse_buttons_pressed` contain keys/buttons that were
/// newly pressed this frame (trigger detection).
/// `keys_down` / `mouse_buttons_down` contain keys/buttons currently held (continuous detection).
#[derive(Default)]
pub struct InputState {
    pub keys_down: HashSet<KeyCode>,
    pub keys_pressed: HashSet<KeyCode>,
    pub keys_released: HashSet<KeyCode>,
    pub mouse_position: (f64, f64),
    /// Mouse movement delta since last frame (pixels).
    pub mouse_delta: (f64, f64),
    pub mouse_buttons_down: MouseButtons,
    pub mouse_buttons_pressed: MouseButtons,
    pub mouse_buttons_released: MouseButtons,
    pub scroll_delta: f32,

    // Per-event deltas (used by runner for camera dispatch, not accumulated)
    pub(crate) event_mouse_delta: (f64, f64),
    pub(crate) event_scroll_delta: f32,

    // Internal: previous mouse position for delta calculation
    prev_mouse_position: Option<(f64, f64)>,
}

/// Mouse button state.
#[derive(Default, Clone, Copy)]
pub struct MouseButtons {
    pub left: bool,
    pub right: bool,
    pub middle: bool,
}

impl InputState {
    /// Resets per-frame transient state (pressed/released/deltas).
    ///
    /// Called automatically at the end of each frame, after rendering.
    /// Continuous state (`keys_down`, `mouse_buttons_down`, `mouse_position`)
    /// is preserved across frames.
    pub(crate) fn end_frame(&mut self) {
        self.keys_pressed.clear();
        self.keys_released.clear();
        self.mouse_buttons_pressed = MouseButtons::default();
        self.mouse_buttons_released = MouseButtons::default();
        self.scroll_delta = 0.0;
        self.mouse_delta = (0.0, 0.0);
    }

    /// Updates input state from a winit `WindowEvent`.
    pub fn handle_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let winit::keyboard::PhysicalKey::Code(key) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            if self.keys_down.insert(key) {
                                self.keys_pressed.insert(key);
                            }
                        }
                        ElementState::Released => {
                            self.keys_down.remove(&key);
                            self.keys_released.insert(key);
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = (position.x, position.y);
                let delta = if let Some(prev) = self.prev_mouse_position {
                    (new_pos.0 - prev.0, new_pos.1 - prev.1)
                } else {
                    (0.0, 0.0)
                };
                self.event_mouse_delta = delta;
                self.mouse_delta.0 += delta.0;
                self.mouse_delta.1 += delta.1;
                self.prev_mouse_position = Some(new_pos);
                self.mouse_position = new_pos;
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = *state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        self.mouse_buttons_down.left = pressed;
                        if pressed {
                            self.mouse_buttons_pressed.left = true;
                        } else {
                            self.mouse_buttons_released.left = true;
                        }
                    }
                    MouseButton::Right => {
                        self.mouse_buttons_down.right = pressed;
                        if pressed {
                            self.mouse_buttons_pressed.right = true;
                        } else {
                            self.mouse_buttons_released.right = true;
                        }
                    }
                    MouseButton::Middle => {
                        self.mouse_buttons_down.middle = pressed;
                        if pressed {
                            self.mouse_buttons_pressed.middle = true;
                        } else {
                            self.mouse_buttons_released.middle = true;
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 50.0,
                };
                self.event_scroll_delta = scroll;
                self.scroll_delta += scroll;
            }
            _ => {}
        }
    }

    /// Clears all keyboard state. Called when egui captures keyboard input
    /// to prevent stuck keys.
    pub(crate) fn clear_keyboard(&mut self) {
        self.keys_down.clear();
        self.keys_pressed.clear();
        self.keys_released.clear();
    }

    /// Clears all pointer/mouse state. Called when egui captures pointer input
    /// to prevent stuck buttons.
    pub(crate) fn clear_pointer(&mut self) {
        self.mouse_buttons_down = MouseButtons::default();
        self.mouse_buttons_pressed = MouseButtons::default();
        self.mouse_buttons_released = MouseButtons::default();
        self.mouse_delta = (0.0, 0.0);
        self.scroll_delta = 0.0;
        self.event_mouse_delta = (0.0, 0.0);
        self.event_scroll_delta = 0.0;
        // Reset prev position to avoid jump delta after egui capture ends
        self.prev_mouse_position = None;
    }

    pub fn is_key_down(&self, key: KeyCode) -> bool {
        self.keys_down.contains(&key)
    }

    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::dpi::PhysicalPosition;
    use winit::event::KeyEvent;
    use winit::event::{
        DeviceId, ElementState, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent,
    };
    use winit::keyboard::{KeyCode, NativeKey, PhysicalKey};

    fn key_event(code: KeyCode, state: ElementState) -> WindowEvent {
        // KeyEvent has a pub(crate) platform_specific field, so we cannot construct it directly.
        // We create a zeroed one via unsafe and set the public fields.
        let mut ke: KeyEvent = unsafe { std::mem::zeroed() };
        ke.physical_key = PhysicalKey::Code(code);
        ke.logical_key = winit::keyboard::Key::Unidentified(NativeKey::Unidentified);
        ke.text = None;
        ke.location = winit::keyboard::KeyLocation::Standard;
        ke.state = state;
        ke.repeat = false;
        WindowEvent::KeyboardInput {
            device_id: DeviceId::dummy(),
            event: ke,
            is_synthetic: false,
        }
    }

    fn cursor_moved(x: f64, y: f64) -> WindowEvent {
        WindowEvent::CursorMoved {
            device_id: DeviceId::dummy(),
            position: PhysicalPosition::new(x, y),
        }
    }

    fn mouse_button(button: MouseButton, state: ElementState) -> WindowEvent {
        WindowEvent::MouseInput {
            device_id: DeviceId::dummy(),
            state,
            button,
        }
    }

    fn mouse_scroll_line(y: f32) -> WindowEvent {
        WindowEvent::MouseWheel {
            device_id: DeviceId::dummy(),
            delta: MouseScrollDelta::LineDelta(0.0, y),
            phase: TouchPhase::Moved,
        }
    }

    #[test]
    fn initial_state() {
        let state = InputState::default();
        assert!(state.keys_down.is_empty());
        assert!(state.keys_pressed.is_empty());
        assert!(state.keys_released.is_empty());
        assert_eq!(state.mouse_position, (0.0, 0.0));
        assert_eq!(state.mouse_delta, (0.0, 0.0));
        assert_eq!(state.scroll_delta, 0.0);
        assert!(!state.mouse_buttons_down.left);
        assert!(!state.mouse_buttons_down.right);
        assert!(!state.mouse_buttons_down.middle);
    }

    #[test]
    fn key_press_tracked() {
        let mut state = InputState::default();
        state.handle_event(&key_event(KeyCode::KeyW, ElementState::Pressed));
        assert!(state.is_key_down(KeyCode::KeyW));
        assert!(state.is_key_pressed(KeyCode::KeyW));
    }

    #[test]
    fn key_release_tracked() {
        let mut state = InputState::default();
        state.handle_event(&key_event(KeyCode::KeyW, ElementState::Pressed));
        state.handle_event(&key_event(KeyCode::KeyW, ElementState::Released));
        assert!(!state.is_key_down(KeyCode::KeyW));
        assert!(state.keys_released.contains(&KeyCode::KeyW));
    }

    #[test]
    fn repeated_press_not_double_counted() {
        let mut state = InputState::default();
        state.handle_event(&key_event(KeyCode::KeyW, ElementState::Pressed));
        assert!(state.keys_pressed.contains(&KeyCode::KeyW));
        // Second press without release — keys_down.insert returns false, so
        // keys_pressed is NOT called again. The key should still be in keys_pressed
        // from the first press, and keys_down should contain it exactly once.
        state.handle_event(&key_event(KeyCode::KeyW, ElementState::Pressed));
        assert!(state.is_key_down(KeyCode::KeyW));
        assert!(state.keys_pressed.contains(&KeyCode::KeyW));
        assert_eq!(state.keys_down.len(), 1);
    }

    #[test]
    fn end_frame_clears_transient() {
        let mut state = InputState::default();
        state.handle_event(&key_event(KeyCode::KeyW, ElementState::Pressed));
        state.end_frame();
        assert!(state.keys_pressed.is_empty());
        assert!(state.is_key_down(KeyCode::KeyW));
        assert_eq!(state.mouse_delta, (0.0, 0.0));
        assert_eq!(state.scroll_delta, 0.0);
    }

    #[test]
    fn mouse_position_updated() {
        let mut state = InputState::default();
        state.handle_event(&cursor_moved(100.0, 200.0));
        assert_eq!(state.mouse_position, (100.0, 200.0));
    }

    #[test]
    fn mouse_delta_calculated() {
        let mut state = InputState::default();
        state.handle_event(&cursor_moved(100.0, 200.0));
        state.handle_event(&cursor_moved(110.0, 220.0));
        assert_eq!(state.mouse_delta, (10.0, 20.0));
    }

    #[test]
    fn mouse_delta_resets_on_end_frame() {
        let mut state = InputState::default();
        state.handle_event(&cursor_moved(100.0, 200.0));
        state.handle_event(&cursor_moved(110.0, 220.0));
        state.end_frame();
        assert_eq!(state.mouse_delta, (0.0, 0.0));
    }

    #[test]
    fn mouse_button_press_and_release() {
        let mut state = InputState::default();
        state.handle_event(&mouse_button(MouseButton::Left, ElementState::Pressed));
        assert!(state.mouse_buttons_down.left);
        assert!(state.mouse_buttons_pressed.left);
        state.handle_event(&mouse_button(MouseButton::Left, ElementState::Released));
        assert!(!state.mouse_buttons_down.left);
        assert!(state.mouse_buttons_released.left);
    }

    #[test]
    fn scroll_line_delta() {
        let mut state = InputState::default();
        state.handle_event(&mouse_scroll_line(3.0));
        assert_eq!(state.scroll_delta, 3.0);
    }

    #[test]
    fn scroll_pixel_delta() {
        let mut state = InputState::default();
        let event = WindowEvent::MouseWheel {
            device_id: DeviceId::dummy(),
            delta: MouseScrollDelta::PixelDelta(PhysicalPosition::new(0.0, 150.0)),
            phase: TouchPhase::Moved,
        };
        state.handle_event(&event);
        assert_eq!(state.scroll_delta, 150.0 / 50.0);
    }

    #[test]
    fn scroll_accumulates() {
        let mut state = InputState::default();
        state.handle_event(&mouse_scroll_line(2.0));
        state.handle_event(&mouse_scroll_line(3.0));
        assert_eq!(state.scroll_delta, 5.0);
    }

    #[test]
    fn event_mouse_delta_is_per_event() {
        let mut state = InputState::default();
        state.handle_event(&cursor_moved(100.0, 200.0));
        state.handle_event(&cursor_moved(110.0, 220.0));
        // Accumulated delta is total
        assert_eq!(state.mouse_delta, (10.0, 20.0));
        // Per-event delta is only the last event's contribution
        assert_eq!(state.event_mouse_delta, (10.0, 20.0));

        // Third move
        state.handle_event(&cursor_moved(115.0, 225.0));
        // Accumulated: (10+5, 20+5) = (15, 25)
        assert_eq!(state.mouse_delta, (15.0, 25.0));
        // Per-event: only (5, 5)
        assert_eq!(state.event_mouse_delta, (5.0, 5.0));
    }

    #[test]
    fn event_scroll_delta_is_per_event() {
        let mut state = InputState::default();
        state.handle_event(&mouse_scroll_line(2.0));
        assert_eq!(state.event_scroll_delta, 2.0);
        assert_eq!(state.scroll_delta, 2.0);

        state.handle_event(&mouse_scroll_line(3.0));
        // Per-event: only 3.0
        assert_eq!(state.event_scroll_delta, 3.0);
        // Accumulated: 5.0
        assert_eq!(state.scroll_delta, 5.0);
    }

    #[test]
    fn clear_pointer_resets_prev_position() {
        let mut state = InputState::default();
        state.handle_event(&cursor_moved(100.0, 200.0));
        state.clear_pointer();

        // After clear, next move should not produce a jump delta
        state.handle_event(&cursor_moved(500.0, 500.0));
        assert_eq!(state.mouse_delta, (0.0, 0.0));
        assert_eq!(state.event_mouse_delta, (0.0, 0.0));
    }
}
