use std::marker::PhantomData;

use bytemuck::Pod;
use wgpu::util::DeviceExt;

/// Creates a `BindGroupLayoutEntry` for a storage buffer.
///
/// Sets `has_dynamic_offset: false` and `min_binding_size: None`.
/// For entries that need dynamic offset or min_binding_size, use raw wgpu.
pub fn storage_buffer_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Creates a `BindGroupLayoutEntry` for a uniform buffer.
///
/// Sets `has_dynamic_offset: false` and `min_binding_size: None`.
/// For entries that need dynamic offset or min_binding_size, use raw wgpu.
pub fn uniform_buffer_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A typed wrapper around a wgpu uniform buffer.
///
/// Provides `new` (create + init) and `write` (queue upload) without
/// manual `bytemuck::bytes_of` calls.
///
/// **WGSL std140 layout is the caller's responsibility** — ensure the
/// Rust struct matches the shader's expected alignment and padding.
pub struct UniformBuffer<T: Pod> {
    buffer: wgpu::Buffer,
    _marker: PhantomData<T>,
}

impl<T: Pod> UniformBuffer<T> {
    /// Creates a new uniform buffer initialized with `initial`.
    ///
    /// Usage flags: `UNIFORM | COPY_DST`.
    pub fn new(device: &wgpu::Device, label: &str, initial: &T) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(initial),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            buffer,
            _marker: PhantomData,
        }
    }

    /// Writes `data` to the buffer via `queue.write_buffer`.
    pub fn write(&self, queue: &wgpu::Queue, data: &T) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(data));
    }

    /// Returns a reference to the underlying `wgpu::Buffer`.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

/// Creates a storage buffer initialized with `data`.
///
/// Usage flags: `STORAGE | COPY_DST`.
/// For uninitialised buffers or additional usage flags, use raw wgpu.
pub fn create_storage_buffer_init<T: Pod>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

/// Creates a compute pipeline from WGSL source in one call.
///
/// Bundles shader module creation, pipeline layout, and pipeline construction.
/// Uses default `compilation_options` and no pipeline cache.
///
/// For pipelines that share a shader module across multiple entry points,
/// or need non-default compilation options, use raw wgpu.
pub fn create_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    wgsl_source: &str,
    bind_group_layouts: &[&wgpu::BindGroupLayout],
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts,
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}
