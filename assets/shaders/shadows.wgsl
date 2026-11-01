struct ShadowUniform { matrix: mat4x4<f32>, flags: vec4<f32> };
fn directional_shadow_visibility(world_position: vec3<f32>, params: ShadowUniform,
    map: texture_depth_2d, comparison: sampler_comparison) -> f32 {
    if params.flags.x == 0.0 { return 1.0; }
    let clip = params.matrix * vec4<f32>(world_position, 1.0);
    if clip.w <= 0.0 { return 1.0; }
    let ndc = clip.xyz / clip.w;
    if any(abs(ndc.xy) > vec2<f32>(1.0)) || ndc.z < 0.0 || ndc.z > 1.0 { return 1.0; }
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    return textureSampleCompareLevel(map, comparison, uv, ndc.z);
}
