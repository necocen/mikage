# Mikage 0.6 runtime architecture

Mikage separates GPU ownership, rendering destinations, simulation, presentation,
and GPU completion. The same `App` runs through a window driver or a headless
harness. External UI hosts may use the GPU and surface layer directly.

```text
winit Window/EventLoop + input/GUI     External UI host       Headless harness
             |                            |                       |
             +------ AppRuntime ----------+-----------------------+
             |                 (optional for external hosts)       |
       SurfaceContext              SurfaceContext           OffscreenTarget
             +-------------------------+---------------------------+
                                GpuContext
                          Instance/Adapter/Device/Queue
```

## GPU and targets

`GpuDescriptor` owns backend, display-handle and adapter preferences, and
`GpuRequirements`. Required features/limits are strict. Optional features are
intersected with adapter support. Preferred limits are clamped in the correct
limit direction and then combined with the required contract. Omitted limits use
the previous downlevel baseline (WebGL2 baseline for GL). Factories can inspect
`GpuCapabilities`, including actual enabled limits and timestamp period, before
building pipelines.

`GpuContext` deliberately has no color format or sample count. Each
`SurfaceContext`/`OffscreenTarget` provides `RenderTargetConfig`; renderer
constructors receive it explicitly. This permits multiple targets on one logical
GPU without pretending their pipeline formats are interchangeable.

An owned display handle is supplied to the window path before Instance creation.
On WASM with `webgl`, WebGPU availability is probed before creating the canvas
surface: a canvas cannot switch graphics-context kinds after creation.

A zero-sized surface is suspended, not configured at a synthetic size. The host
must release acquired frames before resize, reconfiguration or replacement.
Outdated surfaces are reconfigured; Lost surfaces are recreated through the
retained Instance and reattached to the same Adapter/Device. Device loss terminates
the runtime: arbitrary App GPU resources cannot be reconstructed by the framework.
The native runner returns `RunError` after shutdown. Browser startup returns a
Result, while asynchronous failures stop the app and display an error overlay.
The old Surface is released before replacement, including on DX12.

Unsafe surface constructors do not retain the caller's native objects. The caller
must satisfy the target variant's safety requirements throughout async creation
and the lifetime of the surface and any acquired frame. No ScreenSaver/AppKit
ownership or product FFI exists in the library.

## Simulation and presentation

`AppRuntime` owns the GPU, App, camera, simulation time and submission tracking.
It never creates a Window, EventLoop, surface or GUI. Its target is a borrowed
`RenderTarget` for each render.

- `tick`: CPU simulation, uploads and GPU compute, then one immediate submit.
- `prepare_render`: view-dependent uploads, without simulation advancement.
- `render`: records scene rendering.
- `after_submit` / `after_complete`: submission identity and completion hooks.
- `shutdown`: called once before runtime resource teardown, including Drop.

Each tick submits separately. Repeated `queue.write_buffer` calls to the same
uniform before one batched submit would otherwise make all dispatches see the
last value. The initial API favors correct ordinary uploads over implicit batching.
Native exact stepping may wait for queue capacity; final completion is a separate
fence. WASM exposes asynchronous stepping/fences and never blocks the browser.

The window path builds GUI before render preparation, records scene then overlays,
submits all egui callback command buffers before the main encoder, frees GUI
textures after submission, and finally presents. Visual GUI changes affect that
render; simulation changes are consumed by the next tick. Headless rendering
omits GUI unless the caller explicitly composes an overlay.

## Scheduling and input

The default is one variable-duration tick per ordinary redraw, bounded to 250 ms
after a long pause. Capture-only redraws do not tick. `Fixed` mode is driven by a
timer rather than `RedrawRequested`: at most eight ticks per wake, up to 250 ms of
backlog. Discarded wall time is reported. `Manual` advances only by explicit calls.

The default limit is eight in-flight framework submissions; the window driver
reserves one slot for rendering. Exhaustion yields to the event loop. Completion
notifications wake Reactive mode even when no redraw is pending. Automatic
simulation pauses for zero-size, occlusion or host suspension and resets its time
anchor on restoration. Explicit headless/agent stepping does not use wall-time
catch-up limits.

Input remains in the optional window adapter. It delivers filtered input once per
dirty batch through `on_input`, clearing transient edges immediately afterwards.
Apps keep held controls and pending actions in their state and consume actions in
`tick`. This prevents multi-tick replay and preserves input across zero-tick
renders. Focus loss and GUI capture deliver cleared controls to prevent stuck keys.

## Progress, completion and capture

Tick IDs, render-frame IDs and submission IDs are independent. Progress reports
encoded/submitted/completed ticks and frames, submitted/completed submissions,
and presented frames. `presented` means the host called `Queue::present`, not
observed display scanout. Opaque wgpu SubmissionIndex values are returned only in
Rust tokens; HTTP uses runtime IDs.

A native completion worker polls explicit submission endpoints. GPU callbacks
only signal readiness; App callbacks run when the runtime owner drains messages.
A device failure wakes and fails the owner. Raw submissions through cloned Queue
handles are permitted but do not create framework progress IDs or a second queue.

`ReadbackRing` owns bounded staging slots. Mapping is registered on the encoder;
callbacks only signal. The consumer reads/unpads/unmaps outside the callback.
The native capture worker performs that work and PNG encoding off the render
thread. Named resources are re-registered at each checkpoint to avoid stale
textures after resize. GPU copies retain the exact state even if simulation resumes.

Agent commands execute on the runtime owner. Deferred commands encode at their
tick boundary, then complete after their token's fence; Apps may produce the final
JSON from a readback in `complete_agent_command`. `run_until_completed` reaches an
absolute target tick and leaves automatic simulation paused; `runtime.resume`
resumes it. Captures never advance simulation. Exact captures hold advancement
until their copy submission completes. Future tick capture requests require an
active scheduler or an explicit run-until request; historical resources are not
implicitly retained.

Scene/window capture renders into a temporary copyable texture, copies at the
appropriate side of GUI composition, and blits to the surface. It therefore works
without surface COPY_SRC and still presents. HTTP jobs have capacity, retained-byte
and TTL limits; `/status` reads a snapshot without waiting for GPU work.

## Features and dependencies

- No default features: GPU, targets, renderers, App runtime, readback, headless.
- `window`: winit host and input.
- `gui`: egui types/renderer, usable without winit.
- `window-gui` (default): both plus the egui-winit adapter.
- `agent`: native HTTP/capture services, also usable by custom headless hosts.
- `webgl`: WASM WebGL2 fallback; applications requiring compute must still declare
  that requirement and handle unsupported adapters rather than silently lose work.

wgpu 30.0.1 and egui 0.36.1 require API migration; the package MSRV is 1.95.
`vendor/egui-winit/MIKAGE-PATCH.md` records the temporary upstream WASM fix and its
removal criteria. This repository's existing untracked Cargo.lock policy remains.
