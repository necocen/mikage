# Mikage 0.6 verification

Validation was performed on macOS / Apple Metal with Rust 1.95. The commands below
are intended to be rerun from the repository root. GPU tests must fail on an
initialization bug; `MIKAGE_REQUIRE_GPU=1` also rejects genuine adapter absence.

## Build and dependency isolation

```sh
scripts/check-features.sh
cargo clippy --lib --all-features -- -D warnings
cargo fmt --all -- --check
```

The feature script passed 20 native/WASM configurations. Native configurations
include every combination of the independent window, GUI and agent features,
and check all test/example/binary targets. WASM checks the portable library and
WebGPU/WebGL demo/examples, including builds with the native-only agent feature
selected. Three core dependency trees (native, WASM, WASM WebGL) contain no winit
or egui-family crates.

Strict clippy passes for the library, binaries and examples. The optional
`--all-targets -- -D warnings` invocation still reports 12 pre-existing test-style
warnings (camera setup field assignments and a GPU test helper argument count).

## GPU, lifecycle and image regression

```sh
MIKAGE_REQUIRE_GPU=1 cargo test --features agent -- --test-threads=1
cargo run --no-default-features --example headless
cargo run --manifest-path tests/fixtures/external-surface/Cargo.toml
```

The 119 library tests, 24 GPU/image tests and four runtime integration tests pass on
Metal. The tests cover required/optional GPU negotiation, preferred storage-buffer
limits and reverse-direction alignment limits, real pipeline creation, existing
image snapshots, MSAA resolution, zero/one/multiple ticks, per-tick uniform upload
ordering, render-only execution, completed checkpoint identity, named readback,
odd-width row padding, ring reuse/cancellation, device destruction and shutdown.
Runner tests also check invalid configuration and preservation of terminal errors.
Executable rustdoc tests and the external-surface lifetime compile-fail example
also pass. Four existing renderer/helper tutorial snippets remain marked ignored.

The headless example completes exactly 120 ticks and reads a 64×64 scene. The
standalone external consumer creates an owned CAMetalLayer with no Window or
EventLoop, renders and presents with MSAA 4, suspends at zero size, resizes,
releases/recreates the Surface on the same GPU, and tears down before releasing
the layer. Its dependency tree contains no winit or egui.

## Agent and capture end to end

```sh
cargo build --example agent_capture --features agent
python3 tests/smoke_agent.py --output-dir target/agent-smoke-reactive
python3 tests/smoke_agent.py --fixed --msaa4 --output-dir target/agent-smoke-msaa4
```

Both modes use Reactive presentation and an ephemeral localhost port. The script
asserts exact run-until completion and persistent pause, scene/window PNG captures
at the same tick with different GUI content, named raw buffer values, no extra
tick from capture or `/screenshot`, GPU commands at future tick boundaries, exact
capture between later ticks, fixed-timer resume/pause, and normal HTTP shutdown.
The capture path always uses a copyable intermediate texture and render blit;
it does not require Surface COPY_SRC or COPY_DST.

HTTP unit tests additionally cover capacity, retained byte limits, expiration,
shutdown interruption, authentication, malformed dt and rejected queue admission
without leaking an unreachable job reservation. The Python save helper verifies
status/content type and replaces the destination only after a successful result.

## Browser execution

```sh
trunk build --release false --dist target/wasm-webgpu
trunk build --release false --features webgl --dist target/wasm-webgl
```

Both bundles were served from localhost and exercised in Chrome. WebGPU selected
an Apple `metal-3` hardware adapter. The fallback test disabled `navigator.gpu`
before page initialization and verified that the application requested `webgl2`.
Both displayed the sphere, egui panel and advancing time, and continued rendering
after resize and camera interaction. No application exception or GPU validation
error occurred. The sRGB framebuffer preference warning from egui is expected;
mikage intentionally keeps sRGB presentation.

For another browser fallback run, install an init script before navigation:

```javascript
Object.defineProperty(Navigator.prototype, "gpu", { get: () => undefined });
```

The temporary validation browsers and HTTP servers were closed afterwards.

## Platform coverage and release constraint

Metal and Chrome WebGPU/WebGL were executed on hardware. DX12/Vulkan and actual
OS-triggered Surface loss were not exercised on this host; Surface reattachment
is covered by the external consumer. The runner releases the previous Surface
before recreating one, as required for a DX12 HWND swapchain.

This checkout includes the minimal egui-winit 0.36.1 WASM patch as a direct path
dependency. Git/path consumers use it transitively. Before publishing mikage to
crates.io, replace it with an upstream fixed release or a published patched crate;
see [the patch record](../vendor/egui-winit/MIKAGE-PATCH.md). No downstream
sand-picture application or ScreenSaver product was modified.
