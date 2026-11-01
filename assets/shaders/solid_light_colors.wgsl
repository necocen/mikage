#import mikage::light_colors
@group(0) @binding(1) var<uniform> light_colors: LightColors;

@fragment
fn fragment_colored(in: VertexOutput) -> @location(0) vec4<f32> {
    let lit_color = colored_scene_lambert(model_data.color.rgb, normalize(in.world_normal),
        scene.light_dir.xyz, scene.ambient, light_colors.ambient.rgb, light_colors.directional.rgb);
    return vec4<f32>(lit_color, model_data.color.a);
}
