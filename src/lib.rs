//! GPU compute and rendering for windows, external surfaces, and headless applications.
//!
//! [`GpuContext`] owns a logical GPU. [`SurfaceContext`] and [`OffscreenTarget`]
//! describe independent rendering destinations. [`AppRuntime`] separates simulation
//! ticks from rendering and tracks submission and completion explicitly.
//!
//! Default features provide the winit runner and egui. Disable default features
//! for GPU-only or headless use; enable `agent` for native HTTP diagnostics.

#[cfg(all(feature = "agent", not(target_family = "wasm")))]
pub mod agent;
pub mod app;
#[cfg(all(feature = "window", feature = "agent", not(target_family = "wasm")))]
mod blit;
pub mod camera;
pub mod capture;
pub mod context;
#[cfg(feature = "window")]
pub(crate) mod egui_integration;
pub mod headless;
pub mod helpers;
#[cfg(feature = "window")]
pub mod input;
pub mod instance_renderer;
#[cfg(feature = "window")]
mod logging;
pub mod readback;
#[cfg(feature = "window")]
pub mod runner;
pub mod runtime;
pub mod shader_processor;
pub mod solid_renderer;

#[cfg(all(feature = "agent", not(target_family = "wasm")))]
pub use agent::{AgentCommand, AgentConfig, AgentMouseButton, CameraSnapshot};
pub use app::*;
pub use camera::{Camera, Camera2d, InteractiveCamera, OrbitCamera};
pub use capture::*;
pub use context::*;
pub use headless::*;
pub use helpers::{
    CubeMesh, DEPTH_FORMAT, IcoSphereMesh, MeshBuffers, POSITION_NORMAL_LAYOUT, PlaneMesh,
    QuadMesh2d, RegularPolygonMesh, SceneBinding, SceneUniform, UniformBuffer,
    create_compute_pipeline, create_depth_texture, create_storage_buffer_init,
    storage_buffer_entry, uniform_buffer_entry,
};
#[cfg(feature = "window")]
pub use input::InputState;
pub use instance_renderer::{
    ComputeBufferState, InstanceData, InstanceRenderer, InstanceRendererConfig, InstanceVertex,
};
pub use readback::*;
#[cfg(all(feature = "window", feature = "agent", not(target_family = "wasm")))]
pub use runner::run_with_agent;
#[cfg(feature = "window")]
pub use runner::{RedrawPolicy, RunConfig, RunError, SimulationPolicy, run};
pub use runtime::*;
pub use shader_processor::{
    COLOR_UTILS_WGSL, LIGHTING_WGSL, MATH_WGSL, SCENE_TYPES_WGSL, ShaderError, ShaderProcessor,
};
pub use solid_renderer::{ModelUniform, SolidObjectId, SolidRenderer};

pub use dpi;
/// Re-exported for building UI in [`App::gui`].
#[cfg(feature = "gui")]
pub use egui;
/// Re-exported for vector and matrix math.
pub use glam;
/// Re-exported for direct access to wgpu types in application code.
pub use wgpu;
/// Re-exported for access to winit types like `PhysicalSize`.
#[cfg(feature = "window")]
pub use winit;
