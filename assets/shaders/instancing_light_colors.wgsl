#import mikage::light_colors
@group(0) @binding(1) var<uniform> light_colors: LightColors;

@fragment
fn fragment_colored(in: VertexOut) -> @location(0) vec4<f32> {
    let lit_color = colored_scene_lambert(in.color.rgb, normalize(in.world_normal),
        scene.light_dir.xyz, scene.ambient, light_colors.ambient.rgb, light_colors.directional.rgb);
    return vec4<f32>(lit_color, in.color.a);
}
