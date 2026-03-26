//! # mikage
//!
//! A lightweight application framework built on wgpu + winit.
//! Provides GPU rendering, compute shaders, and egui UI integration
//! for both Native and WASM (WebGPU) targets.
//!
//! ## Usage
//!
//! Implement the [`App`] trait and launch with [`run`]:
//!
//! ```no_run
//! use mikage::{App, FrameContext, OrbitCamera, RunConfig, UpdateContext};
//!
//! struct MyApp;
//!
//! impl App for MyApp {
//!     type Camera = OrbitCamera;
//!     fn update(&mut self, _ctx: &mut UpdateContext<OrbitCamera>) {}
//!     fn encode(&mut self, _ctx: &mut FrameContext<OrbitCamera>) {}
//! }
//!
//! mikage::run(|_ctx, _size| MyApp, RunConfig::new("My App"));
//! ```
//!
//! ## Frame Loop
//!
//! Each frame executes in this order:
//!
//! 1. Process winit events, update [`InputState`]
//! 2. Forward events to egui (automatic input lock)
//! 3. Forward mouse/scroll to camera
//! 4. [`App::update`] — logic, data uploads
//! 5. [`App::encode`] — GPU compute and render passes
//! 6. [`App::gui`] — egui UI construction and rendering
//! 7. Submit command buffer and present
//!
//! ## Features
//!
//! - **Raw wgpu access**: The framework manages the surface; you create your own pipelines and buffers.
//! - **Compute shaders**: Encode compute passes in [`App::encode`], before render passes.
//! - **egui integration**: Build UI in [`App::gui`]. Input lock between egui and camera is automatic.
//! - **Camera system**: [`Camera`] trait with a built-in [`OrbitCamera`] implementation.
//! - **Multi-platform**: Native (Metal/Vulkan/DX12) and WASM (WebGPU).
//! - **Helpers**: [`SceneBinding`], [`SceneUniform`], [`UniformBuffer`], [`MeshBuffers`], [`create_storage_buffer_init`], [`storage_buffer_entry`], [`uniform_buffer_entry`], [`create_depth_texture`], and mesh generators.
//! - **Shader imports**: [`ShaderProcessor`] resolves `#import` directives in WGSL shaders.
//! - **Instanced rendering**: [`InstanceRenderer`] with generic [`InstanceVertex`] support for custom per-instance data layouts.
//!
//! ## Examples
//!
//! | Example | Description |
//! |---------|-------------|
//! | `clear` | Minimal app: animated clear color |
//! | `egui_demo` | egui integration: windows, sliders, buttons |
//! | `orbit_camera` | 3D orbit camera with a lit sphere |
//! | `instancing_2d` | 2D hex grid with [`InstanceData`] (pan/zoom) |
//! | `instancing_3d` | 3D sphere grid with wave animation |
//! | `custom_instance` | Custom [`InstanceVertex`] with per-instance 2D rotation |
//! | `boids` | GPU compute flocking (10k boids) with compute-to-instance pipeline |
//!
//! Run with `cargo run -p mikage --example <name>`.

pub mod app;
pub mod camera;
pub mod context;
pub(crate) mod egui_integration;
pub mod helpers;
pub mod input;
pub mod instance_renderer;
mod logging;
pub mod runner;
pub mod shader_processor;
pub mod solid_renderer;
mod time;

pub use app::{App, FrameContext, UpdateContext};
pub use camera::{Camera, Camera2d, InteractiveCamera, OrbitCamera};
pub use context::GpuContext;
pub use helpers::{
    CubeMesh, DEPTH_FORMAT, IcoSphereMesh, MeshBuffers, POSITION_NORMAL_LAYOUT, PlaneMesh,
    QuadMesh2d, RegularPolygonMesh, SceneBinding, SceneUniform, UniformBuffer,
    create_compute_pipeline, create_depth_texture, create_storage_buffer_init,
    storage_buffer_entry, uniform_buffer_entry,
};
pub use input::InputState;
pub use instance_renderer::{
    InstanceData, InstanceRenderer, InstanceRendererConfig, InstanceVertex,
};
pub use runner::{RunConfig, run};
pub use shader_processor::{
    COLOR_UTILS_WGSL, LIGHTING_WGSL, MATH_WGSL, SCENE_TYPES_WGSL, ShaderError, ShaderProcessor,
};
pub use solid_renderer::{ModelUniform, SolidObjectId, SolidRenderer};

/// Re-exported for building UI in [`App::gui`].
pub use egui;
/// Re-exported for vector and matrix math.
pub use glam;
/// Re-exported for direct access to wgpu types in application code.
pub use wgpu;
/// Re-exported for access to winit types like `PhysicalSize`.
pub use winit;
