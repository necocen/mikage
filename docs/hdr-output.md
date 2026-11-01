# HDR output and exposure

`ToneMapper` is an optional final pass over an app-owned linear HDR texture.
The ordinary direct-to-unorm rendering path, its clamping and default appearance
stay unchanged. This is relative scene radiance, without physical light units
or a hidden compatibility scale.

## Pipeline

1. Create an `OffscreenTarget` with `color_format: HDR_FORMAT`
   (`Rgba16Float`), the scene's depth format, size and MSAA count.
2. Create the scene renderers against that target's `RenderTargetConfig`.
   Render opaque and transparent geometry into the same linear HDR target,
   retaining normal depth, alpha blending and draw ordering. Clear to an opaque
   background for a conventional scene image.
3. Resolve MSAA into the target's single-sample texture. Its `view()` is the
   input for `ToneMapper::source(gpu, view)`.
4. Create `ToneMapper::new(gpu, final_config)` and call
   `mapper.encode(encoder, &source, final_target)`. It writes the full image
   in a color-only pass, including final-target MSAA resolve if requested.
5. Render GUI/other display-space overlays afterward. Capture the final image.

For applications that own the final pass themselves, `render(pass, &source)`
draws the full-screen triangle. Use a color-only pass matching the mapper's
destination format and sample count; do not attach scene depth to that pass.
`encode` deliberately omits the supplied target's depth view. The HDR geometry
pass uses the normal depth attachment.

The input view must be a single-sample, non-sRGB floating-point texture with
`TEXTURE_BINDING` usage. No texture filtering sampler is required. Recreate
the HDR target and `ToneMapSource` when size/source changes. A source binding
belongs to the mapper that created it. Recreate the mapper if the final target
format or sample count changes.

`Rgba16Float` preserves linear values above one (finite range up to 65504),
and supports blending/MSAA on standard native and Browser WebGPU devices.
Construction validates the requested target configuration and reports unsupported
formats/sample counts. WebGL/older devices can retain the direct path when the
HDR target cannot be created. Other floating-point formats require their own
rendering/blending capability checks. The helper does not request optional
device features.

## Exposure and mapping

`ToneMapSettings::default()` uses zero EV and `ToneMapOperator::Reinhard`.

```rust,ignore
mapper.update(&gpu.queue, ToneMapSettings {
    exposure_ev: 1.0,
    operator: ToneMapOperator::Reinhard,
});
// Change exposure without resetting the operator or output encoding:
mapper.update_exposure(&gpu.queue, -1.0);
```

Exposure multiplies scene radiance by `2^EV`: 0 is unity, +1 doubles, −1 halves.
It is independent of ambient/directional scalar gains and RGB colors. Non-finite
EV becomes zero and finite values are bounded to `[-32, 32]`.

Reinhard maps each exposed linear channel `x` to `x / (1 + x)`, evaluated as
`1 - 1 / (1 + x)` to retain finite output for overflowing highlights. It keeps
values such as 1, 2, 4 and 8 distinct instead of clamping them all to white.
`ToneMapOperator::None` performs exposure only; unorm destinations still clamp.
No automatic exposure, white-balance or additional filmic curve is applied.

Input RGB is linear premultiplied/composited RGBA, as produced by the standard
alpha blend over transparent black or an opaque scene background. The mapper
unpremultiplies, exposes/maps, then premultiplies again. Alpha is retained within
`[0, 1]`; zero-alpha pixels become zero. Negative RGB is clamped to zero.
Blend transparency **before** mapping so bright transparent objects contribute
to scene radiance before compression. For subsequent linear compositing, select
linear output explicitly.

## Output color space

`ToneMapper::new` selects display sRGB output. Encoding occurs exactly once:

| Destination view | `ToneMapOutput::Srgb` | `ToneMapOutput::Linear` |
| --- | --- | --- |
| sRGB format | Hardware encodes mapped linear RGB | Rejected |
| Non-sRGB float/unorm | WGSL applies the sRGB transfer function | Mapped linear RGB |
| Integer format | Rejected | Rejected |

Select linear output with `ToneMapper::with_output(gpu, config,
ToneMapOutput::Linear)` for downstream processing. Alpha is never gamma-encoded.
Do not add another sRGB conversion after the default mapper.

## Paused rendering and example

Exposure writes, camera/light updates, shadow updates and rendering do not advance
simulation ticks. The application keeps these states separately and updates them
in `prepare_render`; the camera-only scene update retains lighting and colors.
Headless/agent captures must occur after the map pass to include its result.

`cargo run --example lighting_pipeline` demonstrates this complete pipeline with
4× MSAA, depth, transparent geometry, GPU-written instances, colored lights and
directional shadows. Its GUI provides independent EV, ambient/directional gain
and shadow controls while paused.

```sh
cargo run --example lighting_pipeline -- --headless /tmp/hdr.png
cargo run --example lighting_pipeline -- --headless /tmp/hdr-minus1.png --exposure -1
```

Both captures encode zero simulation ticks. See [shadows](shadows.md) and
[lighting](lighting.md) for the scene-side APIs.
