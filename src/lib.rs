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
//! use mikage::{App, GpuContext, RenderContext, RunConfig, UpdateContext};
//! use winit::dpi::PhysicalSize;
//!
//! struct MyApp;
//!
//! impl App for MyApp {
//!     fn init(&mut self, _ctx: &GpuContext, _size: PhysicalSize<u32>) {}
//!     fn update(&mut self, _ctx: &mut UpdateContext) {}
//!     fn render(&mut self, _ctx: &mut RenderContext) {}
//!     fn resize(&mut self, _ctx: &GpuContext, _new_size: PhysicalSize<u32>) {}
//! }
//!
//! mikage::run(MyApp, RunConfig::default());
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
//! 5. [`App::compute`] — GPU compute passes
//! 6. [`App::render`] — GPU render passes
//! 7. [`App::gui`] — egui UI construction and rendering
//! 8. Submit command buffer and present
//!
//! ## Features
//!
//! - **Raw wgpu access**: The framework manages the surface; you create your own pipelines and buffers.
//! - **Compute shaders**: Encode compute passes in [`App::compute`], which runs before rendering.
//! - **egui integration**: Build UI in [`App::gui`]. Input lock between egui and camera is automatic.
//! - **Camera system**: [`Camera`] trait with a built-in [`OrbitCamera`] implementation.
//! - **Multi-platform**: Native (Metal/Vulkan/DX12) and WASM (WebGPU).
//! - **Helpers**: [`SceneBinding`], [`SceneUniform`], [`IcoSphereMesh`], [`CubeMesh`], [`create_depth_texture`], and more.
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

pub use app::{App, ComputeContext, RenderContext, UpdateContext};
pub use camera::{Camera, Camera2d, CameraController, OrbitCamera};
pub use context::GpuContext;
pub use helpers::{
    CubeMesh, DEPTH_FORMAT, IcoSphereMesh, PlaneMesh, QuadMesh2d, RegularPolygonMesh,
    SceneBinding, SceneUniform, create_depth_texture,
};
pub use input::InputState;
pub use instance_renderer::{
    InstanceData, InstanceRenderer, InstanceRendererConfig, InstanceVertex,
};
pub use runner::{RunConfig, run};
pub use shader_processor::{SCENE_TYPES_WGSL, ShaderError, ShaderProcessor};
pub use solid_renderer::{ModelUniform, SolidObjectId, SolidRenderer};

/// Re-exported for building UI in [`App::gui`].
pub use egui;
/// Re-exported for vector and matrix math.
pub use glam;
/// Re-exported for direct access to wgpu types in application code.
pub use wgpu;
/// Re-exported for access to winit types like `PhysicalSize`.
pub use winit;
