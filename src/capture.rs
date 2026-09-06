//! Named GPU resources available to diagnostics and checkpoint capture.

use std::collections::BTreeMap;

/// GPU resource retained by a capture registry.
#[derive(Clone, Debug)]
pub enum CaptureTarget {
    Texture(wgpu::Texture),
    Buffer {
        buffer: wgpu::Buffer,
        offset: u64,
        size: u64,
    },
}

/// Named resources exposed by [`crate::App::capture_targets`].
///
/// Rebuild this registry for each checkpoint so replaced or resized resources
/// cannot remain registered accidentally. The framework registers `scene` and
/// `window` after the application's hook; those two names are reserved.
#[derive(Clone, Debug, Default)]
pub struct CaptureRegistry {
    targets: BTreeMap<String, CaptureTarget>,
}

impl CaptureRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_texture(&mut self, name: impl Into<String>, texture: &wgpu::Texture) {
        self.targets
            .insert(name.into(), CaptureTarget::Texture(texture.clone()));
    }

    pub fn register_buffer(
        &mut self,
        name: impl Into<String>,
        buffer: &wgpu::Buffer,
        offset: u64,
        size: u64,
    ) {
        self.targets.insert(
            name.into(),
            CaptureTarget::Buffer {
                buffer: buffer.clone(),
                offset,
                size,
            },
        );
    }

    pub fn get(&self, name: &str) -> Option<&CaptureTarget> {
        self.targets.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.targets.keys().map(String::as_str)
    }
}
