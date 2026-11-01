#import mikage::shadows
@group(1) @binding(0) var<uniform> shadow: ShadowUniform;
@group(1) @binding(1) var shadow_map: texture_depth_2d;
@group(1) @binding(2) var shadow_sampler: sampler_comparison;

struct ShadowVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
};

@vertex
fn vertex_shadowed(v: Vertex) -> ShadowVertexOutput {
    let world_pos = v.position * v.i_pos_scale.w + v.i_pos_scale.xyz;

    var out: ShadowVertexOutput;
    out.clip_position = scene.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = v.i_color;
    out.world_normal = v.normal;
    out.world_position = world_pos;
    return out;
}

@fragment
fn fragment_shadow_depth(in: VertexOut) {
    if !(in.color.a >= 1.0) { discard; }
}
@fragment
fn fragment_shadowed(in: ShadowVertexOutput) -> @location(0) vec4<f32> {
    let visibility = directional_shadow_visibility(in.world_position, shadow, shadow_map, shadow_sampler);
    let rgb = shadowed_scene_lambert(in.color.rgb, normalize(in.world_normal), scene.light_dir.xyz,
        scene.ambient, vec3<f32>(1.0), vec3<f32>(1.0), visibility);
    return vec4<f32>(rgb, in.color.a);
}
@fragment
fn fragment_shadowed_colored(in: ShadowVertexOutput) -> @location(0) vec4<f32> {
    let visibility = directional_shadow_visibility(in.world_position, shadow, shadow_map, shadow_sampler);
    let rgb = shadowed_scene_lambert(in.color.rgb, normalize(in.world_normal), scene.light_dir.xyz,
        scene.ambient, light_colors.ambient.rgb, light_colors.directional.rgb, visibility);
    return vec4<f32>(rgb, in.color.a);
}
