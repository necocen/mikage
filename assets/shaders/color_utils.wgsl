// Color space conversion utilities.
//
// All functions operate on linear RGB (the GPU's working color space).
// Hue values are in radians [0, 2π).

#import mikage::math

/// Convert HSV to linear RGB.
/// h: hue in radians [0, 2π), s: saturation [0, 1], v: value [0, 1].
fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let hp = h * (3.0 / PI);  // h / (π/3) = h * 3/π
    let x = c * (1.0 - abs(hp % 2.0 - 1.0));
    let m = v - c;
    var rgb: vec3<f32>;
    if hp < 1.0 {
        rgb = vec3<f32>(c, x, 0.0);
    } else if hp < 2.0 {
        rgb = vec3<f32>(x, c, 0.0);
    } else if hp < 3.0 {
        rgb = vec3<f32>(0.0, c, x);
    } else if hp < 4.0 {
        rgb = vec3<f32>(0.0, x, c);
    } else if hp < 5.0 {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }
    return rgb + vec3<f32>(m, m, m);
}

/// Convert HSL to linear RGB.
/// h: hue in radians [0, 2π), s: saturation [0, 1], l: lightness [0, 1].
fn hsl2rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let hp = h * (3.0 / PI);
    let x = c * (1.0 - abs(hp % 2.0 - 1.0));
    let m = l - c * 0.5;
    var rgb: vec3<f32>;
    if hp < 1.0 {
        rgb = vec3<f32>(c, x, 0.0);
    } else if hp < 2.0 {
        rgb = vec3<f32>(x, c, 0.0);
    } else if hp < 3.0 {
        rgb = vec3<f32>(0.0, c, x);
    } else if hp < 4.0 {
        rgb = vec3<f32>(0.0, x, c);
    } else if hp < 5.0 {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }
    return rgb + vec3<f32>(m, m, m);
}

/// Convert a single sRGB component to linear.
fn srgb_component_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

/// Convert sRGB [0, 1] to linear RGB.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_component_to_linear(c.x),
        srgb_component_to_linear(c.y),
        srgb_component_to_linear(c.z),
    );
}

/// Convert a single linear component to sRGB.
fn linear_component_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        return c * 12.92;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

/// Convert linear RGB to sRGB [0, 1].
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        linear_component_to_srgb(c.x),
        linear_component_to_srgb(c.y),
        linear_component_to_srgb(c.z),
    );
}
