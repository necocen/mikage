@group(0) @binding(0) var hdr: texture_2d<f32>;
// x = EV, y = operator (0 Reinhard, 1 exposure only), z = manual sRGB encode.
@group(0) @binding(1) var<uniform> settings: vec4<f32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@vertex
fn vertex(@builtin(vertex_index) index: u32) -> VertexOut {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: VertexOut;
    out.position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}
fn srgb_component(c: f32) -> f32 {
    if c <= 0.0031308 { return 12.92 * c; }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}
@fragment
fn fragment(in: VertexOut) -> @location(0) vec4<f32> {
    let size = textureDimensions(hdr);
    let pixel = clamp(vec2<i32>(in.uv * vec2<f32>(size)), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
    let value = textureLoad(hdr, pixel, 0);
    let alpha = clamp(value.a, 0.0, 1.0);
    if alpha == 0.0 { return vec4<f32>(0.0); }
    let exposed = max(value.rgb / alpha, vec3<f32>(0.0)) * exp2(settings.x);
    var mapped = exposed;
    if settings.y == 0.0 { mapped = vec3<f32>(1.0) - vec3<f32>(1.0) / (vec3<f32>(1.0) + exposed); }
    var rgb = mapped * alpha;
    if settings.z != 0.0 { rgb = vec3<f32>(srgb_component(rgb.r), srgb_component(rgb.g), srgb_component(rgb.b)); }
    return vec4<f32>(rgb, alpha);
}
