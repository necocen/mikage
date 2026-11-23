//! Touch gesture tracking and pointer-input cancellation for the window runner.

use crate::InteractiveCamera;
use crate::input::InputState;

/// Tracks touch gestures: one-finger drag (orbit) and two-finger pinch/pan.
pub(super) struct TouchTracker {
    /// Active camera-owned touch points; GUI/unknown fingers are never added.
    touches: std::collections::HashMap<u64, (f64, f64)>,
    /// Previous distance between two fingers (for pinch detection).
    prev_pinch_distance: Option<f64>,
    /// Previous midpoint of two fingers (for two-finger pan).
    prev_midpoint: Option<(f64, f64)>,
    pinch_sensitivity: f32,
}

impl Default for TouchTracker {
    fn default() -> Self {
        Self {
            touches: std::collections::HashMap::new(),
            prev_pinch_distance: None,
            prev_midpoint: None,
            pinch_sensitivity: Self::DEFAULT_PINCH_SENSITIVITY,
        }
    }
}

pub(super) enum TouchGestureAction {
    /// One-finger drag (orbit): dx, dy in pixels.
    OneDrag { dx: f64, dy: f64 },
    /// Camera touch gesture released or cancelled.
    OneDragEnd,
    /// Two-finger gesture: pinch zoom + pan.
    TwoFinger {
        scroll_delta: f32,
        pan_dx: f64,
        pan_dy: f64,
        /// Midpoint of the two fingers in physical pixels.
        midpoint: (f64, f64),
    },
}

impl TouchGestureAction {
    pub(super) fn dispatch(self, camera: &mut impl InteractiveCamera) {
        match self {
            Self::OneDrag { dx, dy } => camera.on_touch_drag(dx, dy),
            Self::OneDragEnd => camera.on_touch_drag_end(),
            Self::TwoFinger {
                scroll_delta,
                pan_dx,
                pan_dy,
                midpoint,
            } => camera.on_pinch_pan(scroll_delta, pan_dx, pan_dy, Some(midpoint)),
        }
    }
}

pub(super) struct TouchResponse {
    pub(super) suppressed: bool,
    pub(super) action: Option<TouchGestureAction>,
}

pub(super) fn cancel_pointer_input(
    input: &mut InputState,
    touches: &mut TouchTracker,
    camera: &mut impl InteractiveCamera,
) {
    let buttons = input.mouse_buttons_down;
    let touch_end = touches.cancel();
    input.clear_pointer();
    // The default touch-end hook already forwards to on_drag_end. Call the
    // mouse hook separately only for a held mouse button or a mouse-only reset.
    if touch_end.is_none() || buttons.left || buttons.right || buttons.middle {
        camera.on_drag_end();
    }
    if let Some(action) = touch_end {
        action.dispatch(camera);
    }
}

impl TouchTracker {
    const DEFAULT_PINCH_SENSITIVITY: f32 = 5.0;

    pub(super) fn new(pinch_sensitivity: f32) -> Self {
        Self {
            pinch_sensitivity,
            ..Default::default()
        }
    }

    /// Only a non-GUI Started event may give a finger to the camera. Once
    /// cancelled, its remaining Moved/Ended events cannot restart the gesture.
    pub(super) fn handle_touch(
        &mut self,
        touch: &winit::event::Touch,
        captured: bool,
    ) -> TouchResponse {
        use winit::event::TouchPhase;
        if captured && touch.phase == TouchPhase::Started {
            return TouchResponse {
                suppressed: true,
                action: self.cancel(),
            };
        }
        let suppressed =
            touch.phase != TouchPhase::Started && !self.touches.contains_key(&touch.id);
        TouchResponse {
            suppressed,
            action: self.update_touch(touch),
        }
    }

    fn cancel(&mut self) -> Option<TouchGestureAction> {
        let had_touches = !self.touches.is_empty();
        self.touches.clear();
        self.prev_pinch_distance = None;
        self.prev_midpoint = None;
        had_touches.then_some(TouchGestureAction::OneDragEnd)
    }

    fn update_touch(&mut self, touch: &winit::event::Touch) -> Option<TouchGestureAction> {
        use winit::event::TouchPhase;
        match touch.phase {
            TouchPhase::Started => {
                self.touches
                    .insert(touch.id, (touch.location.x, touch.location.y));
                if self.touches.len() == 2 {
                    let (dist, mid) = self.two_finger_state();
                    self.prev_pinch_distance = Some(dist);
                    self.prev_midpoint = Some(mid);
                }
                None
            }
            TouchPhase::Moved => {
                let new_pos = (touch.location.x, touch.location.y);
                let prev_pos = *self.touches.get(&touch.id)?;
                self.touches.insert(touch.id, new_pos);

                match self.touches.len() {
                    1 => {
                        // One-finger drag
                        let dx = new_pos.0 - prev_pos.0;
                        let dy = new_pos.1 - prev_pos.1;
                        Some(TouchGestureAction::OneDrag { dx, dy })
                    }
                    2 => {
                        // Two-finger pinch + pan
                        let (dist, mid) = self.two_finger_state();
                        let scroll_delta = self
                            .prev_pinch_distance
                            .map(|prev| ((dist / prev) - 1.0) as f32 * self.pinch_sensitivity)
                            .unwrap_or(0.0);
                        let (pan_dx, pan_dy) = self
                            .prev_midpoint
                            .map(|(px, py)| (mid.0 - px, mid.1 - py))
                            .unwrap_or((0.0, 0.0));

                        self.prev_pinch_distance = Some(dist);
                        self.prev_midpoint = Some(mid);
                        Some(TouchGestureAction::TwoFinger {
                            scroll_delta,
                            pan_dx,
                            pan_dy,
                            midpoint: mid,
                        })
                    }
                    _ => None,
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.touches.remove(&touch.id)?;
                self.prev_pinch_distance = None;
                self.prev_midpoint = None;
                if self.touches.is_empty() {
                    Some(TouchGestureAction::OneDragEnd)
                } else {
                    None
                }
            }
        }
    }

    fn two_finger_state(&self) -> (f64, (f64, f64)) {
        let mut iter = self.touches.values();
        let &(x0, y0) = iter.next().unwrap();
        let &(x1, y1) = iter.next().unwrap();
        let dx = x1 - x0;
        let dy = y1 - y0;
        let dist = (dx * dx + dy * dy).sqrt().max(1.0);
        let mid = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
        (dist, mid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(id: u64, phase: winit::event::TouchPhase, x: f64, y: f64) -> winit::event::Touch {
        winit::event::Touch {
            device_id: winit::event::DeviceId::dummy(),
            phase,
            location: winit::dpi::PhysicalPosition::new(x, y),
            force: None,
            id,
        }
    }

    #[test]
    fn gui_taps_with_delayed_capture_leave_no_fingers() {
        use winit::event::TouchPhase::*;
        let mut tracker = TouchTracker::default();
        // Even if Started slips through before egui consumes its queued input,
        // Ended must finish the gesture and remove the ID despite GUI capture.
        for id in 0..3 {
            tracker.handle_touch(&touch(id, Started, 1000.0, 30.0), false);
            let ended = tracker.handle_touch(&touch(id, Ended, 1000.0, 30.0), true);
            assert!(!ended.suppressed);
            assert!(matches!(ended.action, Some(TouchGestureAction::OneDragEnd)));
            assert!(tracker.touches.is_empty());
        }
        tracker.handle_touch(&touch(10, Started, 100.0, 100.0), false);
        let moved = tracker.handle_touch(&touch(10, Moved, 140.0, 120.0), false);
        assert!(!moved.suppressed);
        assert!(matches!(
            moved.action,
            Some(TouchGestureAction::OneDrag { dx: 40.0, dy: 20.0 })
        ));
        tracker.handle_touch(&touch(10, Ended, 140.0, 120.0), false);

        tracker.handle_touch(&touch(11, Started, 100.0, 100.0), false);
        tracker.handle_touch(&touch(12, Started, 200.0, 100.0), false);
        let moved = tracker.handle_touch(&touch(11, Moved, 120.0, 100.0), false);
        assert!(matches!(
            moved.action,
            Some(TouchGestureAction::TwoFinger {
                pan_dx: 10.0,
                pan_dy: 0.0,
                ..
            })
        ));
    }

    #[test]
    fn gui_fingers_and_orphan_events_cannot_join_camera_gestures() {
        use winit::event::TouchPhase::*;
        let mut tracker = TouchTracker::default();
        assert!(
            tracker
                .handle_touch(&touch(1, Started, 1000.0, 30.0), true)
                .suppressed
        );
        for phase in [Moved, Moved, Ended, Cancelled] {
            let response = tracker.handle_touch(&touch(1, phase, 100.0, 100.0), false);
            assert!(response.suppressed);
            assert!(response.action.is_none());
            assert!(tracker.touches.is_empty());
        }
        tracker.handle_touch(&touch(2, Started, 100.0, 100.0), false);
        for phase in [Moved, Ended, Cancelled] {
            let response = tracker.handle_touch(&touch(1, phase, 300.0, 100.0), false);
            assert!(response.suppressed);
            assert!(response.action.is_none());
            assert_eq!(tracker.touches.len(), 1);
        }
        let response = tracker.handle_touch(&touch(2, Moved, 110.0, 100.0), false);
        assert!(matches!(
            response.action,
            Some(TouchGestureAction::OneDrag { dx: 10.0, .. })
        ));
    }

    #[test]
    fn gui_capture_cancels_camera_fingers_until_a_fresh_start() {
        use winit::event::TouchPhase::*;
        let mut tracker = TouchTracker::default();
        tracker.handle_touch(&touch(1, Started, 100.0, 100.0), false);
        tracker.handle_touch(&touch(1, Moved, 110.0, 100.0), false);
        let captured = tracker.handle_touch(&touch(2, Started, 1000.0, 30.0), true);
        assert!(matches!(
            captured.action,
            Some(TouchGestureAction::OneDragEnd)
        ));
        let moved = tracker.handle_touch(&touch(1, Moved, 120.0, 100.0), false);
        assert!(moved.suppressed);
        assert!(moved.action.is_none());
        tracker.handle_touch(&touch(1, Ended, 120.0, 100.0), false);
        tracker.handle_touch(&touch(2, Ended, 1000.0, 30.0), false);
        tracker.handle_touch(&touch(3, Started, 100.0, 100.0), false);
        let moved = tracker.handle_touch(&touch(3, Moved, 130.0, 100.0), false);
        assert!(!moved.suppressed);
        assert!(matches!(
            moved.action,
            Some(TouchGestureAction::OneDrag { dx: 30.0, .. })
        ));
    }

    #[cfg(feature = "window-gui")]
    #[test]
    fn scene_touch_keeps_ownership_while_egui_still_has_old_gui_hover() {
        use winit::event::TouchPhase::*;
        let ctx = egui::Context::default();
        let output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280.0, 720.0),
                )),
                events: vec![egui::Event::PointerMoved(egui::pos2(1050.0, 28.0))],
                ..Default::default()
            },
            |ui| {
                egui::Panel::right("controls")
                    .exact_size(300.0)
                    .show(ui, |ui| {
                        let _ = ui.button("Control");
                    });
            },
        );
        output.drop_without_applying_deltas();
        assert!(ctx.egui_wants_pointer_input());
        let scene_pos = egui::pos2(520.0, 300.0);
        let mut tracker = TouchTracker::default();
        let start = tracker.handle_touch(
            &touch(1, Started, 520.0, 300.0),
            crate::egui_integration::position_over_gui(&ctx, scene_pos),
        );
        assert!(!start.suppressed);
        // No GUI frame between Started and Moved: the global query still
        // describes the old hover, but cannot steal an accepted scene finger.
        let moved = tracker.handle_touch(
            &touch(1, Moved, 560.0, 300.0),
            ctx.egui_wants_pointer_input(),
        );
        assert!(!moved.suppressed);
        assert!(matches!(
            moved.action,
            Some(TouchGestureAction::OneDrag { dx: 40.0, dy: 0.0 })
        ));
        let ended = tracker.handle_touch(
            &touch(1, Ended, 560.0, 300.0),
            ctx.egui_wants_pointer_input(),
        );
        assert!(matches!(ended.action, Some(TouchGestureAction::OneDragEnd)));
        assert!(tracker.touches.is_empty());
    }

    #[test]
    fn touch_cancellation_resets_pinch_baselines_and_ends_drag_once() {
        use winit::event::TouchPhase::*;
        for captured in [false, true] {
            let mut tracker = TouchTracker::default();
            tracker.handle_touch(&touch(1, Started, 100.0, 100.0), false);
            let ended = tracker.handle_touch(&touch(1, Cancelled, 100.0, 100.0), captured);
            assert!(matches!(ended.action, Some(TouchGestureAction::OneDragEnd)));
            assert!(tracker.cancel().is_none());

            tracker.handle_touch(&touch(2, Started, 100.0, 100.0), false);
            tracker.handle_touch(&touch(3, Started, 200.0, 100.0), false);
            tracker.handle_touch(&touch(3, Moved, 220.0, 100.0), false);
            // The same reset is used on focus loss and application suspension.
            assert!(matches!(
                tracker.cancel(),
                Some(TouchGestureAction::OneDragEnd)
            ));
            assert!(tracker.touches.is_empty());
            assert!(tracker.prev_pinch_distance.is_none());
            assert!(tracker.prev_midpoint.is_none());
            assert!(
                tracker
                    .handle_touch(&touch(2, Moved, 200.0, 200.0), false)
                    .action
                    .is_none()
            );

            tracker.handle_touch(&touch(4, Started, 0.0, 0.0), false);
            tracker.handle_touch(&touch(5, Started, 100.0, 0.0), false);
            let moved = tracker.handle_touch(&touch(5, Moved, 120.0, 0.0), false);
            match moved.action {
                Some(TouchGestureAction::TwoFinger {
                    scroll_delta,
                    pan_dx,
                    pan_dy,
                    ..
                }) => {
                    assert!((scroll_delta - 1.0).abs() < 1e-5);
                    assert_eq!((pan_dx, pan_dy), (10.0, 0.0));
                }
                _ => panic!("fresh two-finger gesture must use its own baseline"),
            }
        }
    }

    #[test]
    fn touch_finger_count_transitions_do_not_jump() {
        use winit::event::TouchPhase::*;
        let mut tracker = TouchTracker::default();
        tracker.handle_touch(&touch(1, Started, 100.0, 100.0), false);
        tracker.handle_touch(&touch(2, Started, 200.0, 100.0), false);
        tracker.handle_touch(&touch(3, Started, 300.0, 100.0), false);
        assert!(
            tracker
                .handle_touch(&touch(3, Moved, 400.0, 100.0), false)
                .action
                .is_none()
        );
        tracker.handle_touch(&touch(3, Ended, 400.0, 100.0), false);
        assert!(matches!(
            tracker
                .handle_touch(&touch(2, Moved, 210.0, 100.0), false)
                .action,
            Some(TouchGestureAction::TwoFinger {
                scroll_delta: 0.0,
                pan_dx: 0.0,
                pan_dy: 0.0,
                ..
            })
        ));
        tracker.handle_touch(&touch(2, Ended, 210.0, 100.0), false);
        assert!(matches!(
            tracker
                .handle_touch(&touch(1, Moved, 110.0, 100.0), false)
                .action,
            Some(TouchGestureAction::OneDrag { dx: 10.0, dy: 0.0 })
        ));
    }

    #[test]
    fn pointer_reset_notifies_custom_touch_end_without_duplicate_mouse_end() {
        #[derive(Default)]
        struct EndCamera {
            touch_ends: usize,
            drag_ends: usize,
        }
        impl crate::Camera for EndCamera {
            fn view_matrix(&self) -> glam::Mat4 {
                glam::Mat4::IDENTITY
            }
            fn projection_matrix(&self, _: f32) -> glam::Mat4 {
                glam::Mat4::IDENTITY
            }
            fn position(&self) -> glam::Vec3 {
                glam::Vec3::ZERO
            }
        }
        impl InteractiveCamera for EndCamera {
            fn on_mouse_drag(&mut self, _: f64, _: f64, _: bool, _: bool, _: bool) {}
            fn on_scroll(&mut self, _: f32) {}
            fn update(&mut self, _: f32) {}
            fn set_enabled(&mut self, _: bool) {}
            fn is_enabled(&self) -> bool {
                true
            }
            fn on_drag_end(&mut self) {
                self.drag_ends += 1;
            }
            fn on_touch_drag_end(&mut self) {
                self.touch_ends += 1;
                self.on_drag_end();
            }
        }
        let mut camera = EndCamera::default();
        let mut input = InputState::default();
        let mut tracker = TouchTracker::default();
        tracker.handle_touch(
            &touch(1, winit::event::TouchPhase::Started, 100.0, 100.0),
            false,
        );
        cancel_pointer_input(&mut input, &mut tracker, &mut camera);
        assert_eq!((camera.touch_ends, camera.drag_ends), (1, 1));
        assert!(tracker.touches.is_empty());
        cancel_pointer_input(&mut input, &mut tracker, &mut camera);
        assert_eq!((camera.touch_ends, camera.drag_ends), (1, 2));
        input.mouse_buttons_down.left = true;
        cancel_pointer_input(&mut input, &mut tracker, &mut camera);
        assert_eq!((camera.touch_ends, camera.drag_ends), (1, 3));
        assert!(!input.mouse_buttons_down.left);
    }
}
