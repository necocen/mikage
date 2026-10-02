# 0.6.1 agent HTTP verification

The HTTP transport change addresses [issue #1](https://github.com/necocen/mikage/issues/1).
Validation runs on macOS arm64 with Rust 1.95.0. No application, GPU test,
window, or capture/render workload is executed for this change because the GPU
is occupied by another task.

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
dependent. The original GPU capture workload has not been repeated.

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

GPU end-to-end capture validation remains deferred. After the GPU becomes
available, repeat the high-detail PNG capture and verify complete, decodable
downloads with both curl and urllib.
