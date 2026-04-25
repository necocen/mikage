# Agent HTTP API

mikage can expose a small localhost HTTP API for LLM agents and debugging tools.
It is native-only and gated behind the `agent` feature.

## Enable

Use `run_with_agent` instead of `run`:

```rust
fn main() {
    mikage::run_with_agent(
        MyApp::new,
        mikage::RunConfig::default(),
        mikage::AgentConfig::default(),
    );
}
```

By default, the API listens on `127.0.0.1:3939`.

```rust
let agent = mikage::AgentConfig::default()
    .with_bind_addr("127.0.0.1:0".parse().unwrap())
    .with_connection_file(".mikage-agent.json");
```

## Endpoints

### `GET /status`

Returns readiness, window size, frame timing, screenshot support, camera state,
and application-specific status from `App::agent_status()`.

```sh
curl -s http://127.0.0.1:3939/status
```

### `GET /screenshot`

Returns a PNG of the current frame, including egui.

```sh
curl -s http://127.0.0.1:3939/screenshot -o /tmp/mikage.png
```

### `POST /command`

Sends a JSON command to the running app.

```sh
curl -s -X POST http://127.0.0.1:3939/command \
  -H 'content-type: application/json' \
  -d '{"op":"camera.drag","dx":-80,"dy":20,"button":"left"}'
```

Supported built-in commands:

```json
{"op":"camera.drag","dx":-80,"dy":20,"button":"left"}
{"op":"camera.zoom","delta":1.0}
{"op":"camera.set_enabled","enabled":false}
{"op":"camera.set_orbit","yaw":0.6,"pitch":0.4,"distance":3.0}
{"op":"camera.set_2d","position":[0.0,0.0],"zoom":2.0}
{"op":"redraw"}
{"op":"shutdown"}
{"op":"app.command","payload":{"reset":true,"seed":42}}
```

`camera.drag`, `camera.zoom`, and `camera.set_enabled` work with any
`InteractiveCamera`. Absolute `camera.set_orbit` and `camera.set_2d` are
implemented by mikage's built-in `OrbitCamera` and `Camera2d`.

`shutdown` returns `{"ok":true}` and then exits the winit event loop. This is
useful for smoke-test scripts that start an example, capture evidence, and then
cleanly stop it.

`app.command` calls `App::on_agent_command(payload)` so simulations can expose
their own reset, seed, parameter, or inspection hooks without adding framework
commands.

## Optional Auth

For token protection:

```rust
let agent = mikage::AgentConfig::default().with_auth_token("secret");
```

Clients can send either header:

```sh
curl -H 'authorization: Bearer secret' http://127.0.0.1:3939/status
curl -H 'x-mikage-token: secret' http://127.0.0.1:3939/status
```
