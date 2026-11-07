//! Bounded presentation work. Application hooks always stay on the host thread.

/// Controls when the window runner prepares another frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PresentationPolicy {
    /// Acquire and present on the window thread, preserving the original scheduler.
    #[default]
    Immediate,
    /// Coalesce invalidations behind VSync surface availability, with at most one
    /// rendered frame in flight. Native acquisition/presentation use a worker;
    /// browsers use winit's requestAnimationFrame redraws. Requires AutoVsync,
    /// Fifo or FifoRelaxed. No fixed refresh-rate timer is used.
    Vsync,
}

/// Tracks redraw delivery separately from invalidation and GPU completion.
pub(super) struct Presentation {
    pub policy: PresentationPolicy,
    redraw_queued: bool,
    last_submission: Option<u64>,
    #[cfg(not(target_family = "wasm"))]
    pub worker: Option<SurfaceWorker>,
}

impl Presentation {
    pub fn new(policy: PresentationPolicy, wake: crate::runtime::RuntimeWaker) -> Self {
        #[cfg(target_family = "wasm")]
        let _ = wake;
        Self {
            policy,
            redraw_queued: false,
            last_submission: None,
            #[cfg(not(target_family = "wasm"))]
            worker: (policy == PresentationPolicy::Vsync).then(|| SurfaceWorker::new(wake)),
        }
    }

    pub fn frame_in_flight(&self, completed: u64) -> bool {
        self.policy == PresentationPolicy::Vsync
            && self.last_submission.is_some_and(|id| id > completed)
    }

    pub fn submitted(&mut self, id: u64) {
        self.last_submission = Some(id);
    }

    pub fn queue_redraw(&mut self) -> bool {
        if self.policy == PresentationPolicy::Immediate {
            return true;
        }
        if self.redraw_queued {
            return false;
        }
        self.redraw_queued = true;
        true
    }

    pub fn redraw_received(&mut self) {
        self.redraw_queued = false;
    }

    pub fn presenting(&self) -> bool {
        #[cfg(not(target_family = "wasm"))]
        if let Some(worker) = &self.worker {
            return worker.presenting;
        }
        false
    }

    pub fn tick_reserve(&self, completed: u64) -> usize {
        if self.policy == PresentationPolicy::Immediate {
            return 1;
        }
        if self.frame_in_flight(completed) {
            return 0;
        }
        #[cfg(not(target_family = "wasm"))]
        if self.worker.as_ref().is_some_and(SurfaceWorker::busy) {
            return 0;
        }
        // Once a frame can be prepared, reserve one slot so a stream of ticks
        // cannot starve it. Acquisition waits themselves consume no slots.
        1
    }
}

#[cfg(not(target_family = "wasm"))]
pub(super) use native::SurfaceWorker;
#[cfg(not(target_family = "wasm"))]
mod native {
    use crate::{RuntimeError, SubmissionToken, runtime::RuntimeWaker};
    use std::sync::{Arc, mpsc};

    enum Command {
        Acquire(Arc<wgpu::Surface<'static>>),
        Present(wgpu::Queue, wgpu::SurfaceTexture, SubmissionToken),
    }
    enum Reply {
        Acquired(wgpu::CurrentSurfaceTexture),
        Presented(SubmissionToken),
    }

    /// A single operation, with no backlog of obsolete acquisition/present work.
    struct Worker<I, O> {
        sender: Option<mpsc::SyncSender<I>>,
        receiver: mpsc::Receiver<O>,
        thread: Option<std::thread::JoinHandle<()>>,
        pending: bool,
    }

    impl<I: Send + 'static, O: Send + 'static> Worker<I, O> {
        fn new(wake: RuntimeWaker, mut execute: impl FnMut(I) -> O + Send + 'static) -> Self {
            let (sender, commands) = mpsc::sync_channel(1);
            let (replies, receiver) = mpsc::sync_channel(1);
            let thread = std::thread::spawn(move || {
                struct WakeOnExit(RuntimeWaker);
                impl Drop for WakeOnExit {
                    fn drop(&mut self) {
                        (self.0)();
                    }
                }
                // Also wake the owner on a worker panic/disconnection.
                let _exit = WakeOnExit(wake.clone());
                while let Ok(command) = commands.recv() {
                    if replies.send(execute(command)).is_err() {
                        break;
                    }
                    wake();
                }
            });
            Self {
                sender: Some(sender),
                receiver,
                thread: Some(thread),
                pending: false,
            }
        }

        fn send(&mut self, command: I) -> Result<(), RuntimeError> {
            if self.pending {
                return Err(RuntimeError::WouldBlock);
            }
            self.sender
                .as_ref()
                .ok_or(RuntimeError::WorkerStopped)?
                .try_send(command)
                .map_err(|_| RuntimeError::WorkerStopped)?;
            self.pending = true;
            Ok(())
        }

        fn poll(&mut self) -> Result<Option<O>, RuntimeError> {
            match self.receiver.try_recv() {
                Ok(reply) => {
                    self.pending = false;
                    Ok(Some(reply))
                }
                Err(mpsc::TryRecvError::Empty) => Ok(None),
                Err(mpsc::TryRecvError::Disconnected) => Err(RuntimeError::WorkerStopped),
            }
        }
    }

    impl<I, O> Drop for Worker<I, O> {
        fn drop(&mut self) {
            self.sender.take();
            // An acquisition already in progress may take the platform timeout.
            // Release its result before releasing the native surface/window.
            if self.pending {
                let _ = self.receiver.recv();
            }
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    pub(crate) struct SurfaceWorker {
        worker: Worker<Command, Reply>,
        pub presenting: bool,
        pub ready: Option<wgpu::CurrentSurfaceTexture>,
    }

    impl SurfaceWorker {
        pub fn new(wake: RuntimeWaker) -> Self {
            let worker = Worker::new(wake, |command| match command {
                Command::Acquire(surface) => {
                    let frame = surface.get_current_texture();
                    // Lost requires releasing the old swapchain before attaching another.
                    drop(surface);
                    Reply::Acquired(frame)
                }
                Command::Present(queue, frame, token) => {
                    queue.present(frame);
                    Reply::Presented(token)
                }
            });
            Self {
                worker,
                presenting: false,
                ready: None,
            }
        }

        pub fn busy(&self) -> bool {
            self.worker.pending
        }

        pub fn acquire(
            &mut self,
            surface: Arc<wgpu::Surface<'static>>,
        ) -> Result<(), RuntimeError> {
            debug_assert!(self.ready.is_none());
            self.worker.send(Command::Acquire(surface))
        }

        pub fn present(
            &mut self,
            queue: wgpu::Queue,
            frame: wgpu::SurfaceTexture,
            token: SubmissionToken,
        ) -> Result<(), RuntimeError> {
            debug_assert!(self.ready.is_none());
            self.worker.send(Command::Present(queue, frame, token))?;
            self.presenting = true;
            Ok(())
        }

        pub fn poll(&mut self) -> Result<Option<SubmissionToken>, RuntimeError> {
            match self.worker.poll()? {
                Some(Reply::Acquired(frame)) => self.ready = Some(frame),
                Some(Reply::Presented(token)) => {
                    self.presenting = false;
                    return Ok(Some(token));
                }
                None => {}
            }
            Ok(None)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::time::Duration;

        #[test]
        fn blocked_acquisition_keeps_host_free_and_never_queues_obsolete_work() {
            let (started, start) = mpsc::channel();
            let (release, released) = mpsc::channel();
            let (wake, wakes) = mpsc::channel();
            let mut worker = Worker::new(
                Arc::new(move || {
                    let _ = wake.send(());
                }),
                move |tick| {
                    started.send(()).unwrap();
                    released.recv().unwrap();
                    tick
                },
            );
            worker.send(0).unwrap();
            start.recv_timeout(Duration::from_secs(5)).unwrap();
            let mut ticks = 0;
            // The native acquisition can be held across an entire display interval
            // while the owner advances physics and coalesces more invalidations.
            for _ in 0..42 {
                ticks += 1;
                assert_eq!(worker.poll().unwrap(), None);
                assert_eq!(worker.send(ticks), Err(RuntimeError::WouldBlock));
            }
            release.send(()).unwrap();
            wakes.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(worker.poll().unwrap(), Some(0));
            worker.send(ticks).unwrap();
            release.send(()).unwrap();
            wakes.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(worker.poll().unwrap(), Some(42));
        }

        #[test]
        fn worker_failure_wakes_owner_instead_of_leaving_it_asleep() {
            let (wake, wakes) = mpsc::channel();
            let mut worker = Worker::new(
                Arc::new(move || {
                    let _ = wake.send(());
                }),
                |()| {
                    panic!("simulated presentation failure");
                },
            );
            worker.send(()).unwrap();
            wakes.recv_timeout(Duration::from_secs(5)).unwrap();
            // WakeOnExit runs during unwind, before the channels are dropped.
            worker.thread.take().unwrap().join().unwrap_err();
            assert!(matches!(worker.poll(), Err(RuntimeError::WorkerStopped)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn ticks_keep_capacity_until_a_frame_is_ready_then_yield_to_rendering() {
        let mut presentation = Presentation::new(PresentationPolicy::Vsync, Arc::new(|| {}));
        presentation.submitted(7);
        for completed in 0..7 {
            assert!(presentation.frame_in_flight(completed));
            assert_eq!(presentation.tick_reserve(completed), 0);
        }
        assert!(!presentation.frame_in_flight(7));
        assert_eq!(presentation.tick_reserve(7), 1);
        assert!(presentation.queue_redraw());
        for _ in 0..42 {
            assert!(!presentation.queue_redraw());
        }
        presentation.redraw_received();
        assert!(presentation.queue_redraw());
    }

    #[test]
    fn immediate_policy_preserves_uncapped_submission_behavior() {
        let mut presentation = Presentation::new(PresentationPolicy::Immediate, Arc::new(|| {}));
        presentation.submitted(7);
        assert!(!presentation.frame_in_flight(0));
        assert_eq!(presentation.tick_reserve(0), 1);
        assert!(presentation.queue_redraw());
        assert!(presentation.queue_redraw());
    }
}
