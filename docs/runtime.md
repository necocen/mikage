# Simulation, rendering, and GPU completion

`AppRuntime` owns the application, camera, GPU context, and submission bookkeeping.
It does not create a window, surface, or event loop. The window runner uses the
same lifecycle as a headless test; an external host can also use `GpuContext` and
`SurfaceContext` directly without an `AppRuntime`.

## Application lifecycle

| Hook | Responsibility |
| --- | --- |
| `on_input` (`window` feature) | Convert a filtered host input snapshot into app-owned controls and pending actions. |
| `tick` | Advance CPU simulation and encode GPU compute for one explicit timestep. |
| `gui` (`gui` feature) | Build UI; set parameters or pending simulation actions. |
| `prepare_render` | Upload camera matrices and other view-dependent state. |
| `render` | Encode a scene into the supplied target. |
| `after_submit` | Associate app bookkeeping with a queue endpoint. |
| `after_complete` | Consume completion notifications on the runtime's owner thread. |
| `shutdown` | Stop app-owned workers before runtime resources are released. |

Ticks have no window or input dependency. The host delivers transient input once,
including when zero ticks run. Store a reset button or similar one-shot action in
app state until `tick` consumes it. Multiple ticks must not repeat a key press.
Rendering does not advance simulation; a paused app can still move its camera and
redraw its UI. Camera damping is driven by the host's presentation time.

Each tick has its own encoder and queue submission. In particular, repeated
`queue.write_buffer` uploads to the same uniform buffer remain ordered with the
corresponding tick's compute passes. Combining those ticks into one submission
without changing the upload strategy could make all dispatches read the final
uniform value. There is no GPU wait between ticks unless an explicitly blocking
headless operation reaches its configured in-flight capacity.

## Reactive window rendering

Select `RedrawPolicy::Reactive` to draw on input, explicit invalidation, camera
animation, and GUI repaint requests. `Camera2d` reports smooth zoom and released
pan inertia; `OrbitCamera` reports released orbit inertia. Held drags do not
request animation frames on their own. Disabled input still allows existing
inertia to finish. Once animations settle, the runner waits for the next event.
Custom cameras should override `InteractiveCamera::needs_redraw()` when their
`update(dt)` has more animation to perform; the default is `false`.

The camera uses its own clock. New animation after idle starts from the input
event, and continuous animation uses elapsed presentation time capped at 250 ms.
Time spent idle, suspended, or occluded is not integrated into a new animation.
Simulation timing remains controlled separately by `SimulationPolicy`:
`PerRedraw` ticks on camera animation frames, `Fixed` has its own timer, and
`Manual` never ticks automatically. A static interactive viewer can combine
`Reactive` and `Manual`.

Apps receive a cloneable `RedrawRequest` once at window startup, before lifecycle
hooks run. Store it to invalidate app-owned visible state from a hook or a worker:

```rust
use mikage::{App, Camera2d, RedrawRequest};

#[derive(Default)]
struct Viewer {
    redraw: Option<RedrawRequest>,
}

impl App for Viewer {
    type Camera = Camera2d;

    fn set_redraw_request(&mut self, request: RedrawRequest) {
        self.redraw = Some(request);
    }
}

impl Viewer {
    fn invalidate(&self) {
        if let Some(request) = &self.redraw {
            request.request_redraw();
        }
    }
}
```

The handle works with the `window` feature alone and can be cloned into a worker
after arranging synchronization for that worker's app data. Requests coalesce
until handled, wake an idle event loop, and remain pending when requested during
rendering or while the surface is suspended. A redraw respects submission
capacity and does not introduce a tick in `Manual` mode. External/headless hosts
continue to own their presentation scheduling.

## Touch input and GUI capture

The window runner maps one camera-owned finger to drag and two to pinch/pan.
A touch that starts on an egui panel or window belongs to the GUI, even before
the next GUI frame processes the event. Moving that finger outside the GUI does
not start a camera gesture. New touches use physical coordinates converted with
the GUI's pixel scale and the last rendered layout.

Starting a GUI touch or transferring mouse/trackpad input to egui cancels active
camera touch gestures. Losing window focus or suspending the application does
the same. Cancellation clears finger IDs and pinch/pan baselines; remaining
moves from those fingers are ignored, and camera control resumes with fresh
touch starts. Custom cameras receive `on_touch_drag_end` when an active touch
gesture is cancelled.

## Independent VSync presentation

In 0.9, `RunConfig` gains the public `presentation_policy` field. When migrating
an exhaustive struct literal from 0.8, add
`presentation_policy: PresentationPolicy::Immediate`. Configurations built with
constructors or `..Default::default()` select that value automatically.

`RedrawPolicy` chooses what invalidates a window. `PresentationPolicy` chooses
when that invalidation can become another frame. The default `Immediate` retains
the original acquisition/submission behavior and supports uncapped present modes.
Opt into `PresentationPolicy::Vsync` to let surface availability pace rendering
while fixed simulation continues on its own wall clock:

```rust
use mikage::{PresentationPolicy, RedrawPolicy, RunConfig, SimulationPolicy, wgpu};
use std::time::Duration;

let config = RunConfig::default()
    .with_present_mode(wgpu::PresentMode::AutoVsync)
    .with_redraw_policy(RedrawPolicy::Reactive)
    .with_presentation_policy(PresentationPolicy::Vsync)
    .with_simulation_policy(SimulationPolicy::Fixed {
        dt: Duration::from_micros(400),
        max_ticks_per_wake: 4,
        max_backlog: Duration::from_micros(1600),
    });
```

The policy requires `AutoVsync`, `Fifo`, or `FifoRelaxed`; incompatible present
modes return `RunError::InvalidConfig` before starting the event loop. It requests
the backend's minimum supported presentation latency (desired maximum latency 1).
There is no fixed refresh-rate timer or fixed ticks-per-frame ratio: changing
between 60 Hz and 120 Hz leaves `dt`, the wall-time limit and catch-up limits
unchanged. Slower CPU/GPU work or host timers can reduce simulation throughput.
`Fixed` rendering never executes another tick; `PerRedraw` retains its documented
one-tick-per-ordinary-redraw behavior, and `Manual` remains explicit.

On native hosts, a single worker acquires the next surface texture and presents
submitted frames. While acquisition waits for the presentation engine, the host
continues fixed ticks, completion processing and input delivery. Tick, camera,
app and GUI invalidations coalesce before GUI construction, `prepare_render` and
`render`; OS-originated redraws also pass the same availability check. Application
and camera hooks remain on the window thread, so apps need not implement `Send`.
The worker holds at most one acquisition/present operation. At most one rendered
frame is in flight in the runtime, in addition to the presentation engine's
backend-dependent buffering; its successor renders the latest simulation state.

Acquisition waits reserve no runtime submission slots. When a frame becomes
available, one slot is reserved so continuous ticks cannot starve rendering.
When capacity is exhausted, completion notifications wake the runner instead of
polling an expired deadline. Native wgpu serializes `Queue::present` with submits;
the host defers new tick/diagnostic submissions during that operation and resumes
on the worker's notification, retaining only bounded wall-time backlog. Rendering
and compute still share the device queue and GPU execution time. This policy
does not guarantee real-time physics or GPU preemption of long render work.

Resize/reconfiguration waits until acquired textures are released. Input and
camera state update while acquisition is pending, and the next frame uses the
new size/state. Paused apps still redraw; exact-tick diagnostic captures hold the
simulation boundary until the requested frame is encoded. Suspended/occluded
windows stop automatic work. Reactive idle has no presentation polling timer.
On WASM, winit supplies display-aligned `requestAnimationFrame` events; the same
coalescing and frame-in-flight bound apply without a native worker. Headless and
explicit `AppRuntime::render` calls retain host-owned scheduling.

The capture example can exercise this path, including minimal submission
capacity and the issue's 0.4 ms clock:

```sh
cargo build --example agent_capture --features agent
python3 tests/smoke_agent.py --vsync --tick-micros 400 --submission-slots 2
```

Validation on macOS/Metal passed 152 CPU tests, 37 rendering/compute tests,
four headless runtime tests, six doctests, strict Clippy and the native/WASM
feature matrix. A native run with two submission slots advanced 934 fixed
0.4 ms ticks and submitted 44 frames in 0.507 seconds, representing 0.3736 seconds
of simulated time. These are host submission counts, not measured scanout times.
Both single-sample and 4x MSAA capture tests passed pause/resume, exact boundaries,
reactive idle and explicit redraw. A separate native probe verified smooth zoom
settling, resize/capture dimensions, minimize/restore, GUI repaints, all app hooks
on the owner thread and no second render preparation before frame completion.
It coalesced 200 app invalidations into one frame and used about 0.01 CPU seconds
during 0.5 seconds of idle. WASM was compile-checked; other native window backends
and physical refresh-rate switching were not exercised.

## Exact headless stepping

Create the GPU and app without a window, then explicitly choose both tick count
and duration. `advance_ticks` submits exactly that many additional ticks; it does
not automatically wait for the final tick to finish. For example, with an existing
`AppRuntime` named `runtime`:

```rust,ignore
let target_tick = runtime.progress().submitted_ticks + 120;
let endpoint = runtime.advance_ticks(120, std::time::Duration::from_secs_f64(1.0 / 60.0))?;
if let Some(endpoint) = endpoint {
    runtime.wait_for_timeout(&endpoint, Some(std::time::Duration::from_secs(30)))?;
}
assert_eq!(runtime.progress().completed_ticks, target_tick);
```

A count of zero submits no work. `advance_ticks_async` and `wait_for_async` are
available for async hosts, including WebGPU in the browser. Neither completion
wait executes application ticks, rendering, or input handling.

For an image, construct an `OffscreenTarget`, borrow its color/MSAA/depth views in
a `RenderTarget`, and call `runtime.render(target)`. This executes render
preparation and rendering once with zero additional ticks. The returned token can
be waited on independently. See `tests/runtime.rs` for executable GPU examples.

`HeadlessHarness` packages these operations when a test needs both simulation and
capture. Its async constructor accepts a GPU descriptor, target configuration,
size, camera, and the same three-argument app factory as the window runner. The
offscreen textures are allocated only when `render_once` first runs. Named buffer
capture can therefore exercise a compute application without creating any render
target.

`capture_named("scene")` copies the most recently rendered scene and waits for the
copy endpoint. It does not render another frame. Other names come from
`App::capture_targets`; `window` is unavailable without a window compositor.
Results include tightly packed bytes, format/size information for textures, and
submission/tick/frame metadata. The scene retains its original rendered tick id
even when further simulation ticks precede the capture. Async capture variants
yield to the host's executor. Captures use one reusable slot with a 64 MiB limit.

Run the complete native example without any window/GUI feature:

```sh
cargo run --no-default-features --example headless
```

## Progress and timing

Tick ids, rendered frame ids, and submission ids use separate monotonic sequences.
`RuntimeProgress` reports encoded, submitted, and completed progress for ticks and
frames, plus submitted/completed submission counts. A render or diagnostic token's
`tick_id` identifies the latest simulation tick covered by that queue endpoint;
it does not imply that the submission executes another tick.

`presented_frames` counts host calls to `Queue::present`, recorded through
`mark_presented`. It does not measure GPU completion or physical display scanout.
Offscreen renders do not increment it. `elapsed` is simulation time and is
independent of presentation wall time.

Use completion endpoints for GPU throughput measurements. CPU encoding cadence,
submission cadence, and completed GPU cadence measure different things. A later
screenshot cannot retroactively turn an earlier CPU measurement into GPU timing.
App-owned direct queue submissions remain outside runtime counter attribution.

## Nonblocking hosts and completion

`try_tick`, `render_with`, and `submit_command` return `RuntimeError::WouldBlock`
before calling the app when submission capacity is exhausted. The default limit
is eight in-flight submissions; configure it with `set_config`.

A host should drain `poll_completions`, check available capacity, and yield when
full. Install a `RuntimeWaker` to wake a reactive event loop after completion or
device failure. Completion workers and wgpu callbacks never mutate the app;
`after_complete` runs only when its owner drains notifications.

On native platforms, the completion worker waits for each endpoint with short,
cancellable poll intervals. Browser completion uses queue callbacks. Runtime
construction installs the GPU's device-lost callback; preserve that callback when
using the public device handle. Device failure becomes a `RuntimeError` rather
than a successful completion notification. Native `run` and `run_with_agent`
return `Result<(), RunError>` after exiting and shutting down. Browser `run`
returns after startup; later failures stop the event loop and display an error
overlay rather than returning through the already completed call.

### Browser completion progress

Some Firefox WebGPU implementations can delay `onSubmittedWorkDone` delivery
until a slow browser poll after queue submissions stop. A full submission queue
then stalls fixed ticks; the single-frame VSync gate can also stall rendering
even with free submission slots. See [Issue #21](https://github.com/necocen/mikage/issues/21)
and [Mozilla's scheduling report](https://bugzilla.mozilla.org/show_bug.cgi?id=2008103).
`Device::poll` is a no-op on browser WebGPU, so polling it cannot resolve this wait.

`AppRuntime` automatically arms one cancellable browser timer while WebGPU
completion endpoints are outstanding. Only after at least 2 ms without an ordinary
submission or advancing completion does it call `queue.submit([])` to encourage
notification delivery. A real submission or completion postpones the retry;
all endpoints completing, device loss, shutdown or Drop cancels it. There is no
timer in reactive idle after work completes. The global timer also works for
async/offscreen hosts in Web Workers, without a window or host polling loop.
Native completion workers and the WebGL backend keep their existing behavior.

The retry does not encode application work, register another completion callback,
create a token, change tick/frame counters, or free submission capacity. Only an
actual completion notification releases pending work and triggers owner-thread
`after_complete` hooks. Fixed wall-time limits, bounded backlog and display-paced
presentation still apply. This is a best-effort browser workaround: background
timer throttling or genuinely slow GPU work can still delay progress. The 2 ms
value is a nominal inactivity threshold, not a guaranteed wake interval or
completion deadline. The [HTML timer specification](https://html.spec.whatwg.org/multipage/timers-and-user-prompts.html#timers)
clamps sufficiently nested short timers to at least 4 ms and does not guarantee
exact delivery. Load, power management and visibility can extend waits further.
Neither simulation correctness nor submission limits require timer precision.

Completion bursts coalesce into one host wake; each wake drains all queued
endpoints. Registering a host waker also wakes it for work already queued.
For browser VSync presentation, one rAF may be reserved while a frame or submission
capacity is still pending. The redraw callback rechecks GPU completion and capacity
before camera/GUI/render preparation, retaining one prepared frame in flight.
This avoids waiting until completion to reserve the next display opportunity.

The [browser regression fixture](../tests/fixtures/web-completion/README.md)
holds real completed-work notifications until a probe arrives and separately
checks a 100 ms delay that probes cannot shorten, async capacity waits, ordered
hooks, and timer cancellation on shutdown, Drop and device loss.

Validation on 2026-11-07 passed all seven fixture cases in both the Window and
Web Worker environments of Firefox 159 and Chrome 156 (28 cases). Chrome's
prompt native completion case required no probes. Firefox's device-lost event
itself arrived up to about 100 ms after `destroy`; retries stopped when that
event was delivered. CPU tests cover inactivity and late/out-of-order endpoints.

The final runtime was measured with an optimized temporary copy of
granular-clock `65f9c60`, seeded with `StdRng::seed_from_u64(21)` so each run
simulated the same particle trajectory. The original consumer repository was
not changed. Each browser ran separately in a focused foreground popup on
macOS 27/Apple GPU, with a 1280×720 physical canvas at DPR1, 1600 particles,
0.4 ms Fixed ticks, max 4 ticks/wake, 1.6 ms backlog, 8 submission slots and
Reactive + Vsync. After 5,000 warmup ticks, the next 10,000 ticks were measured.
Two final-runtime runs in each browser, interleaved with timer prototypes,
produced these ranges:

| Browser | Simulation speed | Render submits/s | Completion mean (ms) | Completion p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Chrome 156 | 0.548–0.559x | 141.9–142.4 | 4.88–4.98 | 5.44–5.51 |
| Firefox 159 | 0.486–0.488x | 133.0–133.6 | 5.56–5.59 | 7.18–7.36 |

These are host submission/notification timings, including actual GPU work,
not GPU timestamps. All measured runs kept at most eight real submissions
outstanding, prepared no frame before the previous frame completed, and
reported no JavaScript errors. Both browsers used non-fallback adapters.
Chrome's popup viewport was one CSS pixel shorter than Firefox's; physical
canvas dimensions matched. Other foreground runs reached about 0.57x in Chrome,
and Firefox throughput varied substantially between measurement sessions.
These observations do not establish a portable speedup or Chrome parity.

Shorter effective timer intervals and fewer completion callbacks were also
tested in temporary runtime copies. Neither restored Firefox's simulation
throughput to Chrome's, so those additional mechanisms were not adopted.
The injected-delay fixture supplies the deterministic progress/backpressure
regression check; elapsed milliseconds and FPS are environment-dependent
observations. Physical refresh-rate switching and background-tab throughput
were not measured.

The change also passed 161 CPU tests, four Metal runtime integration tests, six
doctests, strict native library Clippy, formatting and all 21 native/WASM feature
configurations (including the new fixture). The pinned size fixture produced
3,524,277 optimized WASM bytes and 1,443,048 gzip bytes, within its existing budgets.
The native window smoke test subsequently passed in a user-run desktop session
with VSync, 0.4 ms Fixed ticks and two submission slots: 892 ticks and 60 frames
over 0.5064 s, advancing 0.3568 s of simulation within the wall-time bound.
Pause/resume, reactive idle/redraw, exact scene/window captures and GPU value
checks all passed. The earlier HTTP 503 from suspended window presentation did
not recur in that run.

## Ordered composition and diagnostics

`render_with` calls `prepare_render` and `render`, then invokes the supplied
composition closure in the same encoder. A window driver can build GUI first,
copy the scene after rendering, encode the GUI overlay, and then copy the composed
window. `RenderContext::extra_command_buffers` supports auxiliary egui buffer
commands; those are submitted before the main encoder.

`submit_command` supplies a separate runtime-owned encoder at the current
simulation boundary. It works without a surface and never advances simulation.
With the agent feature, `encode_agent_command` produces a provisional JSON value;
`complete_agent_command` runs after its endpoint completes and after completion
hooks have been drained. Apps can read their completed diagnostic buffers there
and replace the provisional response.

Call `shutdown` to stop the application and completion worker. It is idempotent
and is also called by `Drop`. Shutdown does not wait for the GPU queue to drain;
request an explicit completion fence first when pending results are needed.
