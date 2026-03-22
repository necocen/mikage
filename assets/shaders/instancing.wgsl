// Generic instanced rendering shader.
//
// Vertex buffer 0: base mesh (position, normal)
// Vertex buffer 1: instance data (pos_scale, color) - per instance
//
// Two fragment entry points:
// - fragment_lit: Lambert diffuse + ambient (for 3D)
// - fragment_unlit: direct color pass-through (for 2D)

#import mikage::scene_types
@group(0) @binding(0) var<uniform> scene: SceneUniform;

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,

    // Per-instance attributes (from instance buffer)
    @location(2) i_pos_scale: vec4<f32>,  // xyz = world position, w = uniform scale
    @location(3) i_color: vec4<f32>,      // RGBA
};

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
};

@vertex
fn vertex(v: Vertex) -> VertexOut {
    let world_pos = v.position * v.i_pos_scale.w + v.i_pos_scale.xyz;

    var out: VertexOut;
    out.clip_position = scene.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = v.i_color;
    out.world_normal = v.normal;
    return out;
}

@fragment
fn fragment_lit(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let diffuse = max(dot(n, scene.light_dir.xyz), 0.0);
    let lit = scene.ambient.x + diffuse * (1.0 - scene.ambient.x);

    return vec4<f32>(in.color.rgb * lit, in.color.a);
}

@fragment
fn fragment_unlit(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}
