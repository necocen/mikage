#!/usr/bin/env bash
# Compile the native/WASM feature matrix and verify core dependency isolation.
# Run from any directory. This script does not execute GPU or browser tests.
set -euo pipefail

mikage_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$mikage_root"

wasm_target=wasm32-unknown-unknown
wasm_libdir=$(rustc --print target-libdir --target "$wasm_target")
if [[ ! -d "$wasm_libdir" ]]; then
    printf 'Missing Rust target %s. Install it with: rustup target add %s\n' \
        "$wasm_target" "$wasm_target" >&2
    exit 1
fi

check() {
    local label=$1
    shift
    printf '\nChecking %s\n' "$label"
    cargo check "$@"
}

assert_core_isolation() {
    local label=$1
    shift
    local dependency_tree package found=0
    printf '\nChecking %s core dependency isolation\n' "$label"
    # Exclude test-only dependencies, but include the complete build graph.
    # Capture cargo's result before inspecting it so resolution failures fail
    # this check too. --prefix none gives one package name at each line's start.
    dependency_tree=$(cargo tree --no-default-features --edges normal,build \
        --prefix none --format '{p}' "$@")
    while IFS= read -r package; do
        case "$package" in
            winit\ *|egui\ *|egui-*|egui_*|epaint\ *|emath\ *|ecolor\ *)
                printf 'Unexpected GUI dependency: %s\n' "$package" >&2
                found=1
                ;;
        esac
    done <<< "$dependency_tree"
    if [[ "$found" -ne 0 ]]; then
        return 1
    fi
    printf '%s core contains no winit or egui-family dependencies.\n' "$label"
}

# --all-targets checks the library, native test code, examples and demo binary.
# Cargo honors each example/binary's required-features declaration.
check 'native default' --all-targets
check 'native core' --no-default-features --all-targets
check 'native window' --no-default-features --features window --all-targets
check 'native GUI without window' --no-default-features --features gui --all-targets
check 'native agent without window/GUI' --no-default-features --features agent --all-targets
check 'native window + GUI without their adapter' --no-default-features --features window,gui --all-targets
check 'native window + agent' --no-default-features --features window,agent --all-targets
check 'native GUI + agent without window' --no-default-features --features gui,agent --all-targets
check 'native window + GUI + agent without their adapter' --no-default-features --features window,gui,agent --all-targets
check 'native default + agent' --features agent --all-targets
check 'native all features' --all-features --all-targets
assert_core_isolation native

# Native integration tests use pollster/threads; do not cross-compile that test
# harness for browsers. Compile portable libs plus browser binaries/examples.
check 'WASM core' --target "$wasm_target" --no-default-features --lib
check 'WASM window' --target "$wasm_target" --no-default-features --features window --lib
check 'WASM GUI without window' --target "$wasm_target" --no-default-features --features gui --lib
check 'WASM agent feature without native HTTP' --target "$wasm_target" --no-default-features --features agent --lib
check 'WASM WebGPU demo/examples' --target "$wasm_target" --lib --bins --examples
check 'WASM WebGPU demo/examples + agent feature' --target "$wasm_target" --features agent --lib --bins --examples
check 'WASM WebGL core' --target "$wasm_target" --no-default-features --features webgl --lib
check 'WASM WebGL demo/examples' --target "$wasm_target" --features webgl --lib --bins --examples
check 'WASM WebGL demo/examples + agent feature' --target "$wasm_target" --features webgl,agent --lib --bins --examples
assert_core_isolation WASM --target "$wasm_target"
assert_core_isolation 'WASM WebGL' --target "$wasm_target" --features webgl

printf '\nFeature matrix and core dependency isolation passed.\n'
