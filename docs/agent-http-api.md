# Agent HTTP API

The native `agent` feature exposes bounded HTTP jobs, captures, and application
commands. The default runner binds `127.0.0.1:3939` when started with
`run_with_agent(factory, RunConfig::default(), AgentConfig::default())`.

For an ephemeral port, use
`AgentConfig::default().with_bind_addr("127.0.0.1:0".parse()?).with_connection_file(".mikage-agent.json")`.
The connection file contains `addr` and an authentication hint, never a token.
`with_auth_token("secret")` enables `Authorization: Bearer secret` or
`X-Mikage-Token: secret` on all endpoints.

## Progress and control

`GET /status` returns readiness, window size, camera and app state, capture target
names, structured GPU capabilities, discarded wall time, and runtime progress.
Supported hardware features/limits and enabled device features/limits are
separate fields. Encoded, submitted, and completed simulation ticks are separate
from rendered frames and submissions. `presented_frames` counts calls to present;
it is not a GPU completion fence or a display scanout counter.

`POST /command` accepts JSON. Immediate commands return HTTP 200 JSON:

```json
{"op":"camera.drag","dx":-80,"dy":20,"button":"left"}
{"op":"camera.zoom","delta":1.0}
{"op":"camera.set_enabled","enabled":false}
{"op":"camera.set_orbit","yaw":0.6,"pitch":0.4,"distance":3.0}
{"op":"camera.set_2d","position":[0.0,0.0],"zoom":2.0}
{"op":"runtime.pause"}
{"op":"runtime.resume"}
{"op":"redraw"}
{"op":"shutdown"}
{"op":"app.command","payload":{"set_seed":42}}
```

Camera variants are supported by the corresponding built-in camera.
`app.command` calls `App::on_agent_command` for immediate CPU operations.
GPU diagnostics use a deferred command and return HTTP 202:

```json
{"op":"app.gpu_command","payload":{"reset":42},"at_tick":120}
{"op":"run_until_completed","target_tick":120,"dt":0.016666668}
```

`at_tick` is optional and identifies a simulation boundary, not a rendered frame.
`App::encode_agent_command` records ordered GPU work into the runtime encoder.
After its submission completes, the runtime calls `App::complete_agent_command`
on the host thread to finalize the response. Neither hook runs inside a GPU
callback. `run_until_completed` advances to an absolute target tick, waits for
GPU completion, and leaves automatic simulation paused. Use `runtime.resume`
to resume the configured simulation policy. Past checkpoints cannot be recreated
by rewinding the runtime.

## Capture jobs

`POST /captures` starts a job immediately:

```json
{"target":"window","format":"png"}
{"target":"scene","format":"png","exact":true,"at_tick":120}
{"target":"particles","format":"raw","exact":true}
```

Defaults are `target: "window"`, `format: "png"`, `exact: false`, and no `at_tick`.
`scene` is captured before GUI composition; `window` includes GUI. App-owned
textures and buffer regions are registered by `App::capture_targets` through
`CaptureRegistry::register_texture` and `register_buffer`. The framework reserves
`scene` and `window`. Rebuild the registry at each checkpoint so replaced buffers
and resized textures cannot leave stale references.

Exact captures coordinate the simulation boundary with the copy. Mapping and
image encoding remain asynchronous. A successful response includes the captured
tick, frame, and submission identifiers when applicable. Texture metadata also
includes dimensions, original texture format, and the tightly packed row size;
buffer metadata includes its source offset.

PNG supports RGBA8/BGRA8 textures, including their sRGB variants. BGRA is converted
to RGBA before PNG encoding. Raw captures preserve the original GPU bytes and
channel order. Texture rows are tightly packed with GPU row padding removed.
Sources require `COPY_SRC`; texture sources must be single-sampled,
uncompressed 2D color textures. Buffer regions must be nonempty and four-byte
aligned. Use raw for float textures and storage buffers.

The creation response is HTTP 202 JSON:

```json
{"id":1,"state":"pending","status_url":"/jobs/1","result_url":"/jobs/1/result"}
```

Poll `GET /jobs/1`. It reports `pending`, `completed`, or `failed`; completed
captures include `metadata` and `bytes`, while failures include an HTTP-style
`status` and `error`. Fetch `GET /jobs/1/result` after completion: PNG returns
`image/png`, raw returns `application/octet-stream`, and command results return
`application/json`. A result requested before completion returns HTTP 202 JSON.
Unknown or expired jobs return HTTP 404. Always check status and content type
before writing a response to an image file.

`GET /screenshot` and `POST /screenshot` are synchronous compatibility wrappers
around a window PNG job. Only the HTTP handler waits; rendering, mapping and PNG
encoding continue independently. A timeout returns HTTP 504 and identifies the
job, which can still be inspected while retained.

## Limits and lifecycle

Defaults: 3 readback staging slots, 64 MiB per copy, 64 retained jobs, 64 MiB of
completed result payloads, a 60-second job lifetime, and 32 concurrent HTTP
connections. Job, result, lifetime and connection limits are configurable through
`AgentConfig`. The request body limit is 1 MiB. Capacity exhaustion returns HTTP
429; clients should wait for outstanding work or retained jobs to expire and
retry with backoff. Fetching a result does not remove the retained job. Failed
requests never silently overwrite a pending capture.

Map callbacks only notify readiness. The native capture worker polls without
blocking the rendering thread, copies mapped bytes, strips row padding, unmaps,
and encodes PNG. Shutdown and device loss complete pending work with errors.
The server stops accepting connections when its bridge is dropped.

## Client helper

Run the example and capture a frame:

```sh
cargo run --example agent_capture --features agent
python3 scripts/capture.py /tmp/mikage.png
python3 scripts/capture.py /tmp/values.bin --target values --format raw --exact
```

The example also accepts `--port 0 --connection-file /tmp/mikage-agent.json`
for an isolated endpoint, `--fixed` for a 10 ms simulation step, or `--manual`
to advance only through control commands.

The Python helper uses only the standard library, follows jobs, verifies HTTP
status and content type, and atomically replaces the output after success.
It accepts `--connection-file`, `--token` (or `MIKAGE_TOKEN`), `--at-tick`, and
`--timeout`.

## Hosts without winit

`--no-default-features --features agent` provides the same native control-plane
building blocks. Start `agent::AgentBridge::start(config, wake_host)` and keep it
alive. The host drains `AgentRequest`s on its own application thread, executes
commands through `AppRuntime`, updates `AgentSnapshot`, and sends one
`AgentResponse` through `request.respond_to`. Both synchronous requests and jobs
use this response path. `job_id` lets the host discard expired work through
`bridge.is_job_pending(id)`. On shutdown or device loss call `bridge.fail_all`.

`agent::AgentCaptureWorker` accepts a registered resource, capture request, and
`ReadbackMetadata` into a caller-owned encoder. The host submits that encoder;
the worker sends the final response. If the encoder will not be submitted, call
`cancel_unsubmitted` for the returned readback ID and drop the encoder. The
host owns simulation scheduling, exact checkpoint policy, and GUI composition.
