// Basic lighting utilities.

/// Lambert diffuse with ambient. Returns a lit color.
/// `normal` should be normalized. `light_dir` is the direction TO the light (normalized).
/// `ambient` is the minimum brightness [0, 1].
fn lambert(color: vec3<f32>, normal: vec3<f32>, light_dir: vec3<f32>, ambient: f32) -> vec3<f32> {
    let diffuse = max(dot(normal, light_dir), 0.0);
    let lit = ambient + diffuse * (1.0 - ambient);
    return color * lit;
}
