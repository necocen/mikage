# Continuous integration

The `CI` GitHub Actions workflow runs on pull requests, pushes to `main`, and
manual dispatch. All jobs use Rust 1.95.0, matching the crate's minimum supported
Rust version. Action dependencies are pinned to commits and tracked by Renovate.

## Checks

| Job | Coverage |
| --- | --- |
| Format, lint and CPU tests | Formatting, strict library Clippy, library tests with `agent` enabled (excluding GPU tests), and doctests |
| Features (Linux, Windows, macOS) | All 20 native/WASM configurations and GUI/HTTP dependency isolation in `scripts/check-features.sh`; macOS also compiles the external CAMetalLayer consumer |
| GPU tests (Vulkan / Lavapipe) | GPU readback/capture library tests, offscreen rendering and image comparisons, compute results, and headless runtime/capture lifecycle |

The GPU job uses Mesa Lavapipe on an ordinary Ubuntu 24.04 runner. It selects
only the Lavapipe Vulkan driver with `VK_DRIVER_FILES` and records the adapter
using `vulkaninfo`. `MIKAGE_REQUIRE_GPU=1` makes unavailable adapters fail instead
of silently skipping tests. Other GPU initialization errors always fail.
Tests run serially to limit software-rendering CPU and memory use.

Lavapipe executes the graphics and compute work in software. These checks cover
correctness, not hardware performance, vendor-specific driver behavior, window
presentation, or browser execution. Windows and macOS jobs compile the native
backends; they do not execute GPU work. Physical GPU, Metal presentation, and
browser smoke tests remain separate checks.

On failure, the GPU job uploads `gpu-failure-diagnostics` for seven days. It
contains adapter information, test logs when available, reference PNGs, and
generated `_actual.png` / `_diff.png` files. Missing reference images fail and
save the actual rendering for inspection.

## Local checks

```sh
rustup target add wasm32-unknown-unknown
cargo fmt --all -- --check
cargo clippy --lib --features agent -- -D warnings
cargo test --lib --features agent -- --skip ::gpu_
cargo test --doc --features agent
bash scripts/check-features.sh

# Requires a working hardware or software adapter.
MIKAGE_REQUIRE_GPU=1 cargo test --lib --features agent ::gpu_ -- --test-threads=1
MIKAGE_REQUIRE_GPU=1 cargo test --features agent --test gpu --test runtime -- --test-threads=1
```

To use Lavapipe on Ubuntu, install `libvulkan1`, `mesa-vulkan-drivers`,
`vulkan-tools`, and `ripgrep`, then select its driver before running the GPU
commands:

```sh
export VK_DRIVER_FILES=$(dpkg -L mesa-vulkan-drivers | rg '/lvp_icd[^/]*\.json$')
export XDG_RUNTIME_DIR=$(mktemp -d)
vulkaninfo --summary
```

Reference images are compared with the existing per-channel and pixel-match
tolerances. Review any rendering differences across implementations before
changing references or tolerances. To intentionally create or replace references
locally, run:

```sh
MIKAGE_REQUIRE_GPU=1 MIKAGE_UPDATE_SNAPSHOTS=1 cargo test --test gpu snapshot_ -- --test-threads=1
git diff -- tests/snapshots
```

Snapshot updates are rejected when the `CI` environment variable is present.
Review the generated PNGs before committing them.
