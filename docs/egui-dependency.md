# egui dependencies

Mikage uses matching **0.37.0** releases of `egui`, `egui-wgpu`, and `egui-winit`
from crates.io. These releases include the
[egui-winit WASM fix from PR #8516](https://github.com/emilk/egui/pull/8516),
so the temporary Git dependency is no longer required. Rust 1.95+ and wgpu
30.0.1 remain supported.

## Keep the dependency family together

Applications should use `mikage::egui` or select `egui = "0.37.0"` when using
egui APIs directly. Any direct `egui-wgpu` or `egui-winit` dependencies should
also use 0.37.0 from crates.io. Mixing Git and registry packages produces distinct
Rust types even when their version numbers match.

[egui 0.37 no longer enables `default_fonts` by default](https://docs.rs/crate/egui/0.37.0/source/Cargo.toml.orig).
Mikage explicitly enables it so existing GUI applications keep their bundled
fonts and render text without installing a custom font provider.

Native `egui-winit` keeps its default features. On WASM, default features remain
disabled and `links` stays enabled. The WASM fix ignores winit's path-only
dropped-file event on browsers; it does not add browser file-drop support.

This library does not track `Cargo.lock`; downstream applications control their
own lockfiles. No vendor copy or `[patch.crates-io]` entry is needed.

## Previous workaround

Mikage 0.6.1 through 0.7.2 pinned the egui family to upstream commit
`1b4a68921f5ee67a529ea29948036d5e7d034952`. The 0.36.2 registry release lacked
the WASM fix, even though it had already been merged into upstream `main`.
The 0.37.0 update replaces all four dependency declarations together, including
both target-specific declarations of `egui-winit`.

For future updates, run `scripts/check-features.sh`, inspect the dependency tree
for duplicate egui-family sources, and exercise GUI capture.

## Validation (2026-10-25)

Validation used macOS with Rust 1.95.0:

```sh
cargo fmt --all -- --check
cargo test --offline --lib --features agent -- --skip readback::tests::gpu_
cargo test --offline --doc --features agent
cargo clippy --offline --lib --features agent -- -D warnings
bash scripts/check-features.sh
```

All 143 CPU-only library tests passed; the two existing GPU readback tests were
excluded. Doctests passed two executable examples and one compile-fail example;
four existing examples remain ignored. Strict library Clippy passed. The HTTP
tests used ordinary host permissions because the sandbox blocks local TCP binds.

All 20 native/WASM feature configurations passed, including WebGPU/WebGL demos
and examples. Core GUI dependency isolation and native HTTP dependency isolation
also passed.

The resolved lockfile contains exactly one registry version, 0.37.0, of each of
`egui`, `egui-wgpu`, `egui-winit`, `emath`, `ecolor`, `epaint`, and
`epaint_default_fonts`, with no Git dependencies.

A light native smoke check launched the existing `agent_capture` example in
manual simulation mode. Scene and GUI-composited window PNGs were both
1280 × 720, and visual inspection confirmed readable GUI text with bundled
fonts. Five frames completed with zero simulation ticks, and the application
shut down cleanly. Browser interaction and other native backends were not run.
