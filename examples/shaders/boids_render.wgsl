// Boids instanced rendering shader.
// Per-instance: vec4<f32> (xy=position, z=angle, w=scale)
// Color derived from velocity direction (angle).

#import mikage::scene_types
@group(0) @binding(0) var<uniform> scene: SceneUniform;

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) i_pos_angle_scale: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vertex(v: Vertex) -> VertexOut {
    let angle = v.i_pos_angle_scale.z;
    let scale = v.i_pos_angle_scale.w;

    // 2D rotation
    let c = cos(angle);
    let s = sin(angle);
    let rotated = vec2<f32>(
        v.position.x * c - v.position.y * s,
        v.position.x * s + v.position.y * c,
    ) * scale;

    let world_pos = vec3<f32>(
        rotated.x + v.i_pos_angle_scale.x,
        rotated.y + v.i_pos_angle_scale.y,
        0.0,
    );

    // Color from velocity direction
    let hue = angle;
    let color = vec3<f32>(
        sin(hue) * 0.3 + 0.7,
        sin(hue + 2.094) * 0.3 + 0.7,
        sin(hue + 4.189) * 0.3 + 0.7,
    );

    var out: VertexOut;
    out.clip_position = scene.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fragment(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
