# Lighting and shadow extension points

The built-in lit fragments in `SolidRenderer` and `InstanceRenderer` support
one directional Lambert light plus uniform ambient illumination. Unlit fragments
pass material RGB through directly. Material colors and lighting calculations
are linear; an sRGB attachment handles output encoding.

## Independent light intensity

Use `SceneLighting` with `SceneUniform::with_lighting` / `set_lighting`, or upload
it directly with `SceneBinding::update_lighting`:

```rust
use mikage::{SceneLighting, SceneUniform, glam::{Mat4, Vec3}};

let lighting = SceneLighting {
    direction: Vec3::new(1.0, 2.0, 1.0), // toward the light, normalized on upload
    ambient_intensity: 0.2,
    directional_intensity: 0.8,
};
let uniform = SceneUniform::with_lighting(Mat4::IDENTITY, Vec3::ZERO, lighting);
```

The linear output before alpha blending is:

```text
material_rgb * (ambient_intensity
    + directional_intensity * max(dot(normal, normalized_light_direction), 0))
```

Both intensities are independent, unitless, finite nonnegative gains. Zero
disables that contribution; values above one are supported. Negative and
non-finite inputs are uploaded as zero. A zero or invalid direction uses +Y;
disable the light with `directional_intensity: 0.0`. Scaling both intensities
scales brightness, so `(0.2, 0.8)` and `(0.4, 1.6)` produce different results.
The default `(0.3, 0.7)` reproduces the default legacy light.

There is no separate exposure, tone mapping, gamma adjustment, or physical lux
unit. For a global exposure multiplier, multiply both gains by the same factor.
Unorm attachments clamp out-of-range output; use an app-owned floating-point
target and tone-mapping pass when HDR is needed. Lighting never changes alpha.

## Shading and transparency

New solid objects use `SolidShading::Auto`: opaque (`alpha >= 1`) is lit,
transparent (`alpha < 1`) is unlit. After registration:

```rust,ignore
solid.set_shading(walls, mikage::SolidShading::Lit);
solid.update_object(gpu, walls, model, mikage::glam::Vec4::new(0.4, 0.4, 0.4, 0.5));
```

`Lit` and `Unlit` persist across transform, color, and alpha updates. `Auto`
restores alpha-based shading. Alpha alone still chooses rendering behavior:

| Alpha | Depth write | Culling | Blending |
| --- | --- | --- | --- |
| `>= 1` | Enabled | Back faces | Alpha blending (opaque at alpha 1) |
| `< 1` | Disabled | None | Straight alpha blending |

Opaque objects draw first; transparent objects keep registration order even
when their shading modes differ. The renderer does not sort transparent meshes
or implement order-independent transparency. Lit transparent back faces use
their original mesh normals, without flipping toward the viewer. This is a
simple Lambert material, without refraction or light transmission.

## Camera updates without light resets

Upload lighting once or whenever it changes with `scene.update_lighting`.
Each frame, call `scene.update_camera(queue, camera, aspect)` to write only the
view-projection matrix and camera position. Lighting uploaded with either the
new API or an older `SceneUniform::with_light` survives camera movement.
Conversely, `update_lighting` preserves the current camera fields.

`scene.update(queue, &uniform)` still replaces the whole uniform.
`scene.update_from_camera` still constructs default lighting for compatibility;
switch to `update_camera` when retaining configured lights is required.

## Uniform and custom shader compatibility

`SceneUniform` remains 112 bytes, with unchanged field names, offsets, and
binding (`@group(0) @binding(0)`). `light_dir.xyz` is the normalized direction;
`light_dir.w` remains unused. The `ambient` vec4 encodes:

| Component | Meaning |
| --- | --- |
| x | Ambient intensity |
| y | Independent directional intensity |
| z | 0 for legacy contrast mode; 1 for independent gains |
| w | Reserved, zero |

`SceneUniform::new` and `with_light` preserve their original bytes and formula
`ambient + diffuse * (1 - ambient)`. The new API uses formerly unused components
to opt into independent gains. Existing custom shaders and the imported
`lambert(color, normal, direction, ambient)` function remain compatible, but
must opt into `scene_lambert(color, normal, scene.light_dir.xyz, scene.ambient)`
from `mikage::lighting` to consume the independent directional gain.

## Custom shadow passes

Built-in renderers do not allocate shadow maps or cast/receive shadows. The
supported extension is an application-owned wgpu depth pass and receiver
pipeline, encoded through the existing `App` hooks:

1. Create a **single-sample** depth texture with `RENDER_ATTACHMENT |
   TEXTURE_BINDING`, a comparison sampler, a light view-projection uniform, and
   app-owned caster/receiver mesh and instance buffers. `MeshBuffers` and
   `POSITION_NORMAL_LAYOUT` can supply the same base geometry layout as the
   standard renderers. Keep light uniforms separate from the camera uniform.
2. In `prepare_render`, update camera data with `SceneBinding::update_camera`
   and upload the light matrix. Prepare light-space data without simulation
   steps; a paused camera can still need a new shadow pass.
3. At the beginning of `render`, encode a depth-only render pass on
   `ctx.encoder`: no color attachments, sample count 1, and the shadow depth
   view. Render casters using the light matrix. End that pass before sampling
   the depth texture.
4. Encode the receiver pass on the same encoder using `ctx.color_attachment`
   and `ctx.target.depth_view`. Create its pipeline with the target's color,
   depth, and MSAA configuration. Bind `SceneBinding` at group 0 and your
   shadow texture/sampler/light matrix at an additional group.
5. Transform world positions to light clip space, divide by w, convert x/y
   from NDC to texture UV (including the Y flip), and compare WebGPU's `[0, 1]`
   depth using `textureSampleCompare`. Treat positions outside the light
   frustum as unshadowed. Apply visibility to the **directional** contribution;
   leave ambient unchanged. Configure depth/slope bias and filtering for the
   scene's scale to limit acne and detached shadows.

`SolidRenderer` owns its fixed shaders, layouts, and object buffers; it cannot
be given an extra shadow bind group. `InstanceRenderer::with_shader` supports
custom WGSL but its pipeline layout contains only the supplied group 0 layout.
For a conventional extra-group shadow pipeline, construct the wgpu pipeline
directly and retain app-owned geometry. A custom instancing shader can instead
use an app-defined group 0 layout containing the scene and shadow resources.

Transparent casters require an application policy (exclude them, alpha-test,
or a transmission technique); ordinary depth maps do not model fractional
transmission. Cascades, point-light shadows, shadow caching, and a built-in
shadow material API are outside the current helper scope. This extension path
addresses the shadow-scope requirement in Issue #10 without promising those
features.
