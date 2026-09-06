# External CAMetalLayer consumer fixture

On macOS, run:

```sh
cargo run --manifest-path tests/fixtures/external-surface/Cargo.toml
```

The standalone consumer enables no mikage default features. It owns a retained
CAMetalLayer, creates the GPU through the raw target API, renders with MSAA,
presents through wgpu 30's queue, suspends at zero size, resizes, recreates the
surface on the same device, and releases GPU surface state before the native
layer. The process exits unsuccessfully if the GPU or drawable is unavailable.

Check dependency isolation with:

```sh
cargo tree --manifest-path tests/fixtures/external-surface/Cargo.toml
```

The tree must contain neither winit nor egui. Cocoa bindings belong only to this
consumer fixture; mikage itself does not implement ScreenSaverView or FFI.
