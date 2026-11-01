#import mikage::shadows
@group(2) @binding(0) var<uniform> shadow: ShadowUniform;
@group(2) @binding(1) var shadow_map: texture_depth_2d;
@group(2) @binding(2) var shadow_sampler: sampler_comparison;

struct ShadowVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
};

@vertex
fn vertex_shadowed(v: VertexInput) -> ShadowVertexOutput {
    let world_pos = (model_data.model * vec4<f32>(v.position, 1.0)).xyz;
    let world_normal = normalize((model_data.model * vec4<f32>(v.normal, 0.0)).xyz);

    var out: ShadowVertexOutput;
    out.clip_position = scene.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_normal = world_normal;
    out.world_position = world_pos;
    return out;
}

@fragment
fn fragment_shadowed(in: ShadowVertexOutput) -> @location(0) vec4<f32> {
    let visibility = directional_shadow_visibility(in.world_position, shadow, shadow_map, shadow_sampler);
    let rgb = shadowed_scene_lambert(model_data.color.rgb, normalize(in.world_normal), scene.light_dir.xyz,
        scene.ambient, vec3<f32>(1.0), vec3<f32>(1.0), visibility);
    return vec4<f32>(rgb, model_data.color.a);
}
@fragment
fn fragment_shadowed_colored(in: ShadowVertexOutput) -> @location(0) vec4<f32> {
    let visibility = directional_shadow_visibility(in.world_position, shadow, shadow_map, shadow_sampler);
    let rgb = shadowed_scene_lambert(model_data.color.rgb, normalize(in.world_normal), scene.light_dir.xyz,
        scene.ambient, light_colors.ambient.rgb, light_colors.directional.rgb, visibility);
    return vec4<f32>(rgb, model_data.color.a);
}
