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
