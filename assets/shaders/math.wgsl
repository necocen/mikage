// Math constants and utilities.

const PI: f32 = 3.14159265;
const TAU: f32 = 6.28318530;

/// Rotate a 2D point by `angle` radians (counter-clockwise).
fn rotate2d(p: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

/// Clamp a vec3 so its magnitude does not exceed `max_mag`.
/// Returns the original vector if already within bounds.
fn clamp_magnitude3(v: vec3<f32>, max_mag: f32) -> vec3<f32> {
    let mag = length(v);
    if mag > max_mag {
        return v * (max_mag / mag);
    }
    return v;
}

/// Clamp a vec2 so its magnitude does not exceed `max_mag`.
fn clamp_magnitude2(v: vec2<f32>, max_mag: f32) -> vec2<f32> {
    let mag = length(v);
    if mag > max_mag {
        return v * (max_mag / mag);
    }
    return v;
}
