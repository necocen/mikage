# 0.6.1 egui dependency verification

This release changes the egui dependency source and removes the local
egui-winit vendor copy. The HTTP transport changes are released as 0.7.0.

The package version is 0.6.1. All four direct egui-family dependencies use
`https://github.com/emilk/egui` at revision
`1b4a68921f5ee67a529ea29948036d5e7d034952`. Native and WASM feature selections
are preserved. See [the dependency record](egui-dependency.md) for why the
family must use one source and for registry-publication constraints.

## Validation

On 2026-10-03, all 20 native/WASM feature configurations in
`scripts/check-features.sh` passed, together with core dependency-isolation
checks. No GPU or browser workloads were executed for this release split.

Cargo metadata confirms that all seven egui-family packages resolve from the
same pinned Git source, with no registry or vendor copy. It contains no
axum/Hyper packages. The native agent feature and its dependencies are
unchanged from v0.6.0.

Source comparison confirms that `src`, `tests`, `examples`, and `scripts`
match v0.6.0 exactly. The release contains only the egui dependency change,
vendor removal, version bump, and their documentation.
