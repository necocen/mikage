# 0.7.0 agent HTTP verification

The HTTP transport change addresses [issue #1](https://github.com/necocen/mikage/issues/1).
Validation runs on macOS arm64 with Rust 1.95.0. Initial validation was GPU-free
because another task occupied the GPU. On 2026-10-03, permission was extended
to light GPU use, allowing the controlled capture checks recorded below.

## Isolating the original socket failure

A standalone program using only `std` reproduces the old socket sequence:
nonblocking listener, `accept`, a 5-second write timeout, `write_all` of a 2 MiB
payload, then drop the stream. The client waits 100 ms before reading.

- Without resetting the accepted stream to blocking mode, `write_all` returns
  `WouldBlock` / macOS error 35 after approximately 0.16 ms. The client receives
  327,212 of 2,097,152 bytes.
- With `set_nonblocking(false)` before the same write, the write succeeds after
  approximately 107 ms and the client receives all 2,097,152 bytes.

This confirms the accepted-socket behavior on this host and a failure mechanism
that occurs before the configured timeout. The exact cutoff is timing/platform
dependent. The original downstream GPU workload has not been repeated; the
follow-up below exercises capture using a controlled workload.

## GPU-free regression command

```sh
cargo test --no-default-features --features agent --lib agent:: -- --skip gpu_capture_worker --test-threads=1
```

The filter explicitly excludes the agent test that creates a GPU device. The
tests in `agent::http::tests` use real localhost TCP connections and synthetic
completed results. They cover byte equality for a result larger than 2 MiB with
a constrained receive buffer and slow reader, repeated downloads, fragmented
authenticated requests, request-body limits, connection admission during a
blocked transfer, disconnect recovery, application and I/O timeouts, relay
capacity/cancellation, and bounded shutdown including delivery of its response.
The write-timeout test also captures and checks the transport error log.

Existing agent unit tests cover job admission, TTL, retained-result limits,
authentication, commands, and CPU-only PNG conversion.

Result: all 18 selected agent tests passed.

## Compile-only portability checks

`scripts/check-features.sh` compiles the native/WASM feature combinations and
checks dependency isolation without executing GPU or browser code. Its checks
include exclusion of axum/Hyper/Tokio from native builds without `agent` and from
WASM even when `agent` is enabled.

Result: all 20 feature configurations and the dependency-isolation checks passed.

Formatting and `cargo clippy --no-default-features --features agent --lib -- -D warnings`
pass. Strict Clippy for all targets reports pre-existing
`field_reassign_with_default` warnings in camera tests and `too_many_arguments`
in `tests/gpu.rs`; those files are unchanged by this fix.

## Light GPU capture follow-up

The light GPU checks used the axum implementation at commit
`5846c62f31655d25d6400a8e9bf2241004b477a5` from a separate source snapshot on
Apple M1 Max / Metal, before the release was assigned version 0.7.0. Built with
`--no-default-features --features window,agent`; GUI was disabled.

- Ran `agent_capture --manual --port 0 --connection-file <temporary-path>`.
  The 1280 x 720 window capture decoded to the expected uniform RGBA color
  `[44, 69, 85, 255]`. Both curl and urllib retrieved the completed capture job
  and the synchronous `/screenshot` response: four matching 19,562-byte PNGs.
- A temporary headless host used the public `AgentBridge` and
  `AgentCaptureWorker` APIs. It uploaded a CPU-generated, deterministic
  1024 x 1024 RGBA noise image to a GPU texture, then captured it as PNG.
  GPU work consisted only of upload and readback; there was no compute pass.
  Each PNG was 4,195,716 bytes, large enough to exercise the response-transfer
  failure covered by issue #1.
- Downloaded that completed job three times with each client, including a
  curl download limited to 2 MiB/s. Also fetched `/screenshot` with both
  clients. All eight downloads had HTTP 200, `image/png`, a body matching
  `Content-Length`, and identical bytes. Every decoded pixel matched the
  CPU-generated source. PNG SHA-256:
  `4a968c1dbcd9be2c15a96eb40fc40457453a0a3428bd08661e401b6f631353e4`.
- The window app's encoded, submitted, and completed simulation tick counts
  remained zero. Both hosts acknowledged shutdown and exited successfully.

No performance measurements or simulation compute workloads were run. This
confirms complete GPU capture downloads for the tested paths while leaving
full downstream workload and other GPU backend validation outside this run.

## Release split validation

On 2026-10-03, after assigning the egui dependency change to 0.6.1 and the
HTTP transport change to 0.7.0, the combined 0.7.0 checkout passed all 18
GPU-free agent regression tests and all 20 native/WASM feature configurations,
including core and native HTTP dependency-isolation checks. The egui family
uses the same pinned source as 0.6.1.

`src`, `tests`, `examples`, and `scripts` are identical to the previously
validated axum implementation and its verification commit `605fa7e`.
The version split does not change that implementation. The published main
history and the egui-only 0.6.1 commit are both ancestors of 0.7.0.
