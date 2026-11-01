# Directional shadows

`DirectionalShadow` shares a single directional-light depth map between
`SolidRenderer` and `InstanceRenderer<InstanceData>`. Both the white-light and
colored-light constructors are supported. Existing constructors keep their
unshadowed behavior.

## Setup and pass order

Create a `DirectionalShadow` with an explicit light view-projection matrix and
the renderers with `with_shadows` or `with_light_colors_and_shadows`. For each
frame, after any compute writes to instance data:

```rust,ignore
if let Some(mut pass) = shadow.begin_pass(encoder) {
    solid.render_shadow(&mut pass);
    instances.render_shadow(&mut pass);
}
// End the depth pass before sampling the map.
// Start the normal scene pass, bind the application scene at group 0,
// and call solid.render()/instances.render() as usual.
```

The shadow pass uses each renderer's existing mesh/model buffers and the
**current** instance vertex buffer, including GPU-computed data and buffer
growth through `prepare_compute`. No CPU copy of instance positions is needed.
Rebuild compute bind groups when `prepare_compute().reallocated` is true,
as with ordinary instanced rendering.

The map is single-sample `Depth32Float`, independently of the main target's
MSAA count. It needs no color attachment. Depth compares use `LessEqual`.

## Light space and quality

`DirectionalShadowConfig::light_view_projection` maps world space to light
clip space: X/Y in `[-1, 1]`, Z in `[0, 1]`. Supply an orthographic projection
and a light view looking toward the scene, with the eye on the side indicated
by `SceneLighting::direction`. Choose bounds/near/far that contain the intended
casters and receivers; positions outside the light volume are unshadowed.
The example uses `glam::perspective::directx::orthographic` and
`glam::view::look_at_mat4`. Texture Y conversion is handled by the receiver.

- `resolution` defaults to 1024. Zero and values above device limits are errors.
- `depth_bias` and `slope_bias` configure raster depth bias (defaults 2 and 2.0).
  Adjust them for the chosen light volume to control acne and detached shadows.
- `ShadowFilter::Hard` performs one nearest comparison. `Pcf`, the default,
  uses hardware bilinear comparison filtering over a 2×2 footprint.

Resolution, bias and filter are fixed at construction. Recreate the helper and
associated renderers to change them. `update_matrix(queue, matrix)` updates
light space separately from the application's camera. `set_enabled(queue, bool)`
toggles receiving and whether `begin_pass` returns a pass, without rebuilding
pipelines. Enable a disabled map before encoding its first depth pass.

## Materials and controls

`ShadowOptions { cast, receive }` defaults to both enabled. Apply it per object
with `SolidRenderer::set_shadow_options(id, options)` or to an instance batch
with `InstanceRenderer::set_shadow_options(options)`. Separate instance batches
can use different policies.

Opaque objects cast. Alpha below one excludes a solid or individual instance
from the depth map; instance alpha is tested on the GPU. Transparent surfaces
can still receive directional shadows with their usual alpha blending. This is
opaque occlusion, without colored transmission or alpha-texture cutouts.
Unlit materials remain unlit. Shadow visibility multiplies only the directional
term, preserving ambient illumination and both independent RGB light colors.

## Capabilities and custom shaders

Metal, Vulkan, DX12 and compliant Browser WebGPU use baseline depth comparison
sampling without optional device features. The helper checks format support,
comparison sampling and resource limits. WebGL and insufficient devices return
a valid helper with `is_supported() == false`; standard shadow constructors then
use the ordinary unshadowed shaders, and `begin_pass` returns `None`.
Invalid settings still return `ShadowError`. Check `is_supported()` to expose
availability in an application UI.

For custom receivers, `receiver_layout()` and `receiver_bind_group()` return the
optional receiver resources: binding 0 is an 80-byte matrix/flags uniform,
binding 1 a depth texture, and binding 2 a comparison sampler.
`#import mikage::shadows` (also exported as `SHADOWS_WGSL`) provides
`directional_shadow_visibility(world_position, params, map, comparison)`.

| Standard pipeline | Group 0 | Group 1 | Group 2 |
| --- | --- | --- | --- |
| Solid main | Application scene | Object model/color | Shadow receiver |
| Instance main | Application scene | Shadow receiver | — |
| Solid depth | Helper light scene | Same object model/color | — |
| Instance depth | Helper light scene | — | — |

The helper binds its own light scene in the depth pass, leaving the application
camera and lighting buffers intact. End that pass and bind the application scene
again in the main pass. A renderer remains associated with the shadow helper
used to create it. Cascades, point lights and custom shadow policies can use
app-owned passes as described in [lighting](lighting.md).

## Combined example

`cargo run --example lighting_pipeline` combines both renderer types, RGB
lighting, shadows, GPU-generated instances and [HDR/exposure](hdr-output.md).
GUI controls change lighting, shadows and EV while simulation is paused.

```sh
cargo run --example lighting_pipeline -- --headless /tmp/lighting.png
cargo run --example lighting_pipeline -- --headless /tmp/unshadowed.png --no-shadows
cargo run --example lighting_pipeline -- --headless /tmp/darker.png --exposure -1
```

Each headless command captures the completed output with zero simulation ticks.
