# Temporary egui Git dependency

Mikage temporarily uses [egui PR #8516](https://github.com/emilk/egui/pull/8516),
which fixes [egui-winit WASM compilation](https://github.com/emilk/egui/issues/8436).
The PR was squash-merged into `main` on 2026-09-07 as
`da7169ed1af4ab6c5487d53c6a83a5d47835a163`. At the 2026-09-16 check, no registry release contained it:
egui 0.36.2 (published 2026-09-08) was cut from the release branch without this
fix, and `egui-winit = "=0.36.2"` from crates.io still fails to compile for
`wasm32-unknown-unknown` with the same `DroppedFile` trait mismatch (checked
2026-09-16). The next release expected to include it is 0.37.0. Cargo dependencies
use the upstream repository `https://github.com/emilk/egui`, pinned to the PR head
commit `1b4a68921f5ee67a529ea29948036d5e7d034952` rather than a moving branch or
PR ref.

The upstream fix compiles `NativeFile` and its dropped-file event handler only
on native targets. On WASM it ignores the path-only dropped-file event, which
winit's web backend does not emit. It does not add browser file-drop support.
It also adds a direct WASM compilation check for egui-winit upstream.

## Keep the dependency family together

The Git workspace uses path dependencies between egui crates. Therefore mikage
pins `egui`, `egui-wgpu` and both target-specific `egui-winit` dependencies to the
same Git source and revision. Their emath/ecolor/epaint dependencies resolve from
that same checkout. Mixing Git egui-winit with registry egui/egui-wgpu would
produce distinct Rust types even when every package says version 0.36.1.

This revision reports egui 0.36.1 and Rust 1.95, but includes main-branch changes
beyond the WASM fix, including vello 0.2 and smallvec 1.16 requirements. All egui-family package
sources are pinned by Cargo.toml even though this library does not track Cargo.lock.
When updating an older local lockfile, `cargo update -p smallvec --precise 1.16.0`
resolves its previous 1.15.x selection without broadly updating other packages.

Git/path consumers of mikage inherit these direct Git dependencies; they do not
need a root-only `[patch.crates-io]` entry. Applications should use
`mikage::egui` or match the exact Git dependency when using egui / egui-wgpu
APIs directly. Native and WASM egui-winit feature selections are unchanged.
The former local vendor copy and custom patch are no longer needed.

## Return to registry releases

Once an upstream release contains the fix, switch all three egui dependencies
together to matching registry versions. Rerun `scripts/check-features.sh`, inspect
the dependency tree for duplicate egui-family sources, and exercise GUI capture.
A crates.io release of mikage still requires registry-compatible dependencies;
using a Git PR does not by itself remove that publication constraint.
