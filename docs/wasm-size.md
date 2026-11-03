# WASM size budget

CI measures a dedicated, fixed application using WebGPU at runtime in
`tests/fixtures/wasm-size`. It draws one lit cube with `SolidRenderer`, uses an
`OrbitCamera`, and displays an egui label and slider. This covers ordinary window,
shader, mesh, camera and GUI use without changing when the showcase demo changes.
It does not exercise every optional renderer or represent a standalone size for
the mikage library.

## Measurement contract

- Target: `wasm32-unknown-unknown`, default window/GUI integration and dependency
  features. The fixture selects WebGPU at runtime.
- Library manifests and dependency feature defaults are left unchanged. Transitive
  dependencies can also compile WebGL support; the resolved wgpu features are
  recorded in the report. There is no compiled-backend exclusion requirement.
- Rust 1.95.0, wasm-bindgen crate/CLI 0.2.128, Binaryen wasm-opt 116.
- A separate, tracked fixture `Cargo.lock`; builds use `--locked`.
- Remap checkout, Cargo cache, build output and Rust toolchain paths to stable
  prefixes so panic locations do not depend on the installation directory.
  The measurement sets its own compiler flags rather than inheriting `RUSTFLAGS`.
- Release profile: `opt-level = "s"`, LTO, one codegen unit, aborting panics,
  stripped debug information. Then wasm-bindgen `--target web` and wasm-opt
  `-Oz --strip-debug --strip-producers --enable-bulk-memory --enable-reference-types`.
- Measure the final `*_bg.wasm`, plus gzip level 9 with no filename or timestamp.
  JavaScript glue size is reported separately, outside the WASM budget.

`budget.json` holds absolute limits for both raw WASM and gzip bytes. Either limit
being exceeded fails CI. Initial limits are the measured baseline plus 5%, rounded
up to whole KiB: **5,765,120 raw bytes** and **2,317,312 gzip bytes**. The initial
measurement was **5,490,318 raw bytes / 2,206,681 gzip bytes** on macOS arm64.
Small differences between build hosts or compression libraries fit within that
margin. The first Linux CI report should be checked against this baseline.
The tool versions, hashes and measured baseline are recorded in
[`wasm-size-baseline.json`](wasm-size-baseline.json).

The build, optimization and size checks live directly in the `wasm-size` job in
[the CI workflow](../.github/workflows/ci.yml). Historical measurement helpers are
not maintained as repository scripts.

The CI job publishes a byte-count table in the job summary and uploads the report,
resolved lockfile, and runnable application as the `wasm-size` artifact, including
on budget failures. Current artifacts live in `target/wasm-size/current/` locally.

Keep the fixture's workload stable. Review intentional workload, dependency, tool,
or optimization changes together with the resulting size change and budget.
A library dependency update may require an explicit fixture lockfile update; the
size job never silently updates its dependency resolution.

## Local measurement

Use the tool versions above. Install the Rust target and matching wasm-bindgen
CLI:

```sh
rustup toolchain install 1.95.0 --profile minimal --target wasm32-unknown-unknown
cargo +1.95.0 install wasm-bindgen-cli --version 0.2.128 --locked
```

Install Binaryen 116 from its
[official release](https://github.com/WebAssembly/binaryen/releases/tag/version_116)
and put its `bin` directory on `PATH`. The CI workflow contains pinned Linux
release downloads and their SHA-256 checksums.

The workflow's build step includes path normalization and the final budget check.
For a quick local measurement with the same optimization settings:

```sh
export RUSTUP_TOOLCHAIN=1.95.0
export CARGO_TARGET_DIR="$PWD/target/wasm-size/build"
fixture=tests/fixtures/wasm-size
output=target/wasm-size/current
cargo build --manifest-path "$fixture/Cargo.toml" --locked --release --target wasm32-unknown-unknown
mkdir -p "$output/dist"
wasm-bindgen "$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/mikage-wasm-size.wasm" --target web --out-dir "$output/dist" --out-name mikage_wasm_size
wasm-opt "$output/dist/mikage_wasm_size_bg.wasm" -Oz --strip-debug --strip-producers --enable-bulk-memory --enable-reference-types -o "$output/dist/optimized.wasm"
mv "$output/dist/optimized.wasm" "$output/dist/mikage_wasm_size_bg.wasm"
gzip -9nc "$output/dist/mikage_wasm_size_bg.wasm" > "$output/dist/mikage_wasm_size_bg.wasm.gz"
wc -c "$output/dist/mikage_wasm_size_bg.wasm" "$output/dist/mikage_wasm_size_bg.wasm.gz"
```

Local installation paths and gzip implementations can cause small differences.
Use the workflow's commands for the path-normalized CI measurement. To preview
the application, copy the fixture's `index.html` into the output directory and
serve it:

```sh
cp tests/fixtures/wasm-size/index.html target/wasm-size/current/dist/index.html
python3 -m http.server 8093 --directory target/wasm-size/current/dist
```

For a dependency update, refresh the separate fixture lockfile explicitly and
review its diff together with the size report:

```sh
cargo +1.95.0 update --manifest-path tests/fixtures/wasm-size/Cargo.toml
```

## Historical reference measurements

Each 0.x series uses its latest patch tag. These are reference builds made with
the same new cube/camera/GUI workload and current measurement tools, **not archived
release artifacts**. For this one-time comparison, each tag was exported to a
temporary checkout and only the new fixture was adapted to the older application
API. Library manifests and dependency feature selections were not patched. Older
releases used their default WebGPU runner; newer fixtures explicitly select WebGPU.

Historical lockfiles started from the current fixture lockfile and resolved
additional or incompatible versions as needed. Results include dependency
changes, rather than isolating changes to mikage's own source. The recorded
[reference data](wasm-size-history.json) includes each tag, exact commit, tools,
enabled wgpu features, and lockfile digest. Future CI measures the current fixture;
it does not rebuild these historical references.

Measured on 2026-11-03, macOS arm64:

| Release | Final WASM bytes | gzip bytes | WebGL compiled by dependencies |
| --- | ---: | ---: | --- |
| v0.1.0 | 2,731,954 | 1,332,248 | No |
| v0.2.0 | 2,735,859 | 1,333,131 | No |
| v0.3.4 | 2,737,421 | 1,333,630 | No |
| v0.4.0 | 2,737,439 | 1,333,641 | No |
| v0.5.0 | 2,742,443 | 1,336,123 | No |
| v0.6.1 | 5,429,471 | 2,179,799 | Yes |
| v0.7.3 | 5,487,314 | 2,205,486 | Yes |
| v0.8.1 | 5,490,363 | 2,206,652 | Yes |

All fixtures use WebGPU at runtime. Starting with the measured 0.6 series,
egui-wgpu defaults additionally compile WebGL support. These standard-feature
measurements include that backend code; the controlled font experiment below
excluded it.

## Controlled investigation of the egui text-stack increase

An earlier controlled experiment excluded WebGL from dependency features. Its
full-fixture values below describe that experiment, not the default-feature
baselines above. The standard dependency graph can include additional WebGL
backend code through egui-wgpu; this also contributes to the normal binary size.

The experiment isolates the egui text stack update. Mikage 0.6 moved from egui
0.33 to 0.36, crossing egui's switch from `ab_glyph` to `skrifa` + `vello_cpu`
in [0.34](https://github.com/emilk/egui/releases/tag/0.34.0), followed by
`harfrust` shaping in [0.35](https://github.com/emilk/egui/releases/tag/0.35.0).
Upstream also tracks this [WASM size regression](https://github.com/emilk/egui/issues/8328).

An isolated program constructs the same egui window, label and slider and
tessellates them, without linking mikage or winit. It uses the same compiler,
release profile, wasm-bindgen and wasm-opt settings as the main measurement,
with an isolated dependency graph:

| Isolated egui version | Final WASM bytes | gzip bytes |
| --- | ---: | ---: |
| 0.33.3 | 2,153,349 | 1,093,506 |
| 0.36.1, same pinned Git revision as mikage 0.6.1 | 2,812,347 | 1,148,630 |

That reproduces **658,998 raw bytes / 55,124 gzip bytes** of growth without
mikage, compared with **713,923 raw bytes / 79,326 gzip bytes** for the controlled
WebGPU-only 0.5.0-to-0.6.1 fixture. The isolated and full programs are different workloads;
their deltas are corroborating evidence, not an exact additive decomposition.

Function-name profiles of the controlled fixture have code/data section sizes
identical to its stripped artifacts. Code grows from 1,088,132 to 2,314,754 bytes,
while data shrinks from 1,621,550 to 1,104,205 bytes. Newly included font-stack
crates (`skrifa`, `read_fonts`, `harfrust`, `vello_cpu`, `vello_common`,
`fearless_simd`, `kurbo`) account for 928,470 function-body bytes by their retained
function names. This is shallow attribution after optimization: inlined or shared
code may be attributed to callers.

See [`wasm-size-analysis.json`](wasm-size-analysis.json) for the measurements.
This investigation does not change the fixture or its CI budget.
