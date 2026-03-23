// Boids compute shader.
// Reads boid state, applies separation/alignment/cohesion,
// writes updated state + instance data for rendering.

struct BoidState {
    pos: vec2<f32>,
    vel: vec2<f32>,
};

struct BoidParams {
    num_boids: u32,
    dt: f32,
    separation_radius: f32,
    alignment_radius: f32,
    cohesion_radius: f32,
    separation_strength: f32,
    alignment_strength: f32,
    cohesion_strength: f32,
    max_speed: f32,
    min_speed: f32,
    world_size: f32,
    boid_scale: f32,
    fov_cosine: f32, // cos(half_angle); -1.0 = 360°, 0.0 = 180°
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

struct RotatedInstance {
    pos_angle_scale: vec4<f32>,
};

@group(0) @binding(0) var<storage, read>       boids_in:  array<BoidState>;
@group(0) @binding(1) var<storage, read_write> boids_out: array<BoidState>;
@group(0) @binding(2) var<storage, read_write> instances: array<RotatedInstance>;
@group(0) @binding(3) var<uniform>             params:    BoidParams;

// Wrap-around shortest distance
fn wrap_diff(d: f32, size: f32) -> f32 {
    let s2 = size * 2.0;
    return d - round(d / s2) * s2;
}

fn wrap_pos(p: f32, size: f32) -> f32 {
    let s2 = size * 2.0;
    return p - floor((p + size) / s2) * s2;
}

@compute @workgroup_size(64)
fn update_boids(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.num_boids) {
        return;
    }

    let self_state = boids_in[id];

    var sep_force = vec2<f32>(0.0, 0.0);
    var align_vel = vec2<f32>(0.0, 0.0);
    var cohesion_center = vec2<f32>(0.0, 0.0);
    var sep_count = 0u;
    var align_count = 0u;
    var cohesion_count = 0u;

    // Normalized velocity for FOV check
    let self_speed = length(self_state.vel);
    let self_dir = select(
        vec2<f32>(1.0, 0.0),
        self_state.vel / self_speed,
        self_speed > 0.001,
    );

    for (var j = 0u; j < params.num_boids; j++) {
        if (j == id) {
            continue;
        }
        let other = boids_in[j];

        // Wrap-around difference (self - other)
        let dx = wrap_diff(self_state.pos.x - other.pos.x, params.world_size);
        let dy = wrap_diff(self_state.pos.y - other.pos.y, params.world_size);
        let diff = vec2<f32>(dx, dy);
        let dist = length(diff);

        // Separation (always applies regardless of FOV)
        if (dist < params.separation_radius && dist > 0.001) {
            sep_force += normalize(diff) / dist;
            sep_count++;
        }

        // FOV check for alignment and cohesion:
        // Is the neighbor in front of us?
        let in_fov = dist < 0.001 || dot(-normalize(diff), self_dir) >= params.fov_cosine;

        // Alignment
        if (in_fov && dist < params.alignment_radius) {
            align_vel += other.vel;
            align_count++;
        }
        // Cohesion
        if (in_fov && dist < params.cohesion_radius) {
            cohesion_center += self_state.pos - diff;
            cohesion_count++;
        }
    }

    var accel = vec2<f32>(0.0, 0.0);

    if (sep_count > 0u) {
        accel += normalize(sep_force) * params.separation_strength;
    }
    if (align_count > 0u) {
        let avg_vel = align_vel / f32(align_count);
        accel += (avg_vel - self_state.vel) * params.alignment_strength;
    }
    if (cohesion_count > 0u) {
        let center = cohesion_center / f32(cohesion_count);
        var to_center = vec2<f32>(
            wrap_diff(center.x - self_state.pos.x, params.world_size),
            wrap_diff(center.y - self_state.pos.y, params.world_size),
        );
        accel += to_center * params.cohesion_strength;
    }

    // Integrate velocity
    var new_vel = self_state.vel + accel * params.dt;

    // Clamp speed
    let speed = length(new_vel);
    if (speed > params.max_speed) {
        new_vel = normalize(new_vel) * params.max_speed;
    } else if (speed < params.min_speed && speed > 0.001) {
        new_vel = normalize(new_vel) * params.min_speed;
    }

    // Integrate position with wrap-around
    var new_pos = vec2<f32>(
        wrap_pos(self_state.pos.x + new_vel.x * params.dt, params.world_size),
        wrap_pos(self_state.pos.y + new_vel.y * params.dt, params.world_size),
    );

    // Write simulation state
    boids_out[id].pos = new_pos;
    boids_out[id].vel = new_vel;

    // Write instance data for rendering
    let angle = atan2(new_vel.y, new_vel.x);
    instances[id].pos_angle_scale = vec4<f32>(new_pos.x, new_pos.y, angle, params.boid_scale);
}
