# Migrating from mikage 0.7 to 0.8

Mikage 0.8.0 includes the lighting improvements from Issue #10 and updates its
public math dependency from glam 0.29 to 0.34. The lighting additions preserve
existing defaults; the public glam change is why this release uses 0.8.0
instead of 0.7.4.

## Align glam types

Camera traits, scene uniforms, and renderer methods expose glam types.
Types from glam 0.29 and 0.34 are distinct: a consumer using both versions
cannot pass a 0.29 `Vec3` or `Mat4` to a mikage method expecting the 0.34 type.
Prefer `use mikage::glam` or update direct dependencies to:

```toml
glam = "0.34"
```

Custom cameras and application-side matrix construction must use glam 0.34's
camera functions. For the conventions used by mikage:

| Previous constructor | Replacement |
| --- | --- |
| `Mat4::look_at_rh(eye, target, up)` | `glam::camera::rh::view::look_at_mat4(eye, target, up)` |
| `Mat4::perspective_rh(fov_y, aspect, near, far)` | `glam::camera::rh::proj::directx::perspective(fov_y, aspect, near, far)` |
| `Mat4::orthographic_rh(left, right, bottom, top, near, far)` | `glam::camera::rh::proj::directx::orthographic(left, right, bottom, top, near, far)` |

The built-in cameras already use these replacements. Their right-handed view,
Y-up convention, and WebGPU `[0, 1]` depth range are preserved.

## Adopt lighting controls when needed

Existing lighting constructors and alpha-based solid shading keep their behavior.
Use `SolidShading::Lit` for lit transparency, `SceneLighting` for independent
ambient/directional gains, and `SceneBinding::update_camera` to move the camera
without resetting uploaded lighting. See [lighting and custom shadow passes](lighting.md).

The Rust 1.95 minimum, wgpu 30.0.1, egui 0.37.0, and feature selections are
unchanged. The native pollster 1.0 update is internal to mikage.
