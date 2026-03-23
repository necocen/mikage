// Solid-colored mesh rendering shader.
// Opaque objects use Lambert diffuse + ambient from SceneUniform.
// Transparent objects are unlit (color passed through directly).

#import mikage::scene_types
#import mikage::lighting
@group(0) @binding(0) var<uniform> scene: SceneUniform;

struct ModelUniform {
    model: mat4x4<f32>,
    color: vec4<f32>,
};
@group(1) @binding(0) var<uniform> model_data: ModelUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};

@vertex
fn vertex(v: VertexInput) -> VertexOutput {
    let world_pos = (model_data.model * vec4<f32>(v.position, 1.0)).xyz;
    let world_normal = normalize((model_data.model * vec4<f32>(v.normal, 0.0)).xyz);

    var out: VertexOutput;
    out.clip_position = scene.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_normal = world_normal;
    return out;
}

// Lit fragment for opaque objects
@fragment
fn fragment_lit(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let lit_color = lambert(model_data.color.rgb, n, scene.light_dir.xyz, scene.ambient.x);
    return vec4<f32>(lit_color, model_data.color.a);
}

// Unlit fragment for transparent objects
@fragment
fn fragment_unlit(in: VertexOutput) -> @location(0) vec4<f32> {
    return model_data.color;
}
