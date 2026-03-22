use std::collections::HashSet;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::KeyCode;

/// Per-frame input state.
///
/// Tracks keyboard, mouse, and scroll state. Updated automatically by the
/// framework. Access via [`UpdateContext::input`](crate::UpdateContext::input).
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
    /// Resets per-frame state. Called automatically at the start of each frame.
    pub fn begin_frame(&mut self) {
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
                if let Some(prev) = self.prev_mouse_position {
                    self.mouse_delta.0 += new_pos.0 - prev.0;
                    self.mouse_delta.1 += new_pos.1 - prev.1;
                }
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
                self.scroll_delta += match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 50.0,
                };
            }
            _ => {}
        }
    }

    pub fn is_key_down(&self, key: KeyCode) -> bool {
        self.keys_down.contains(&key)
    }

    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }
}
