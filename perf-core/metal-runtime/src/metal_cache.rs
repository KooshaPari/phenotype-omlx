//! Per-thread cache for verified Metal execution objects.

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::cell::RefCell;
use std::collections::HashMap;

use crate::native_catalog::spec_for_tag;
use crate::MetallibArtifact;

struct MetalCache {
    device: Option<metal::Device>,
    queue: Option<metal::CommandQueue>,
    pipelines: HashMap<String, metal::ComputePipelineState>,
}

impl MetalCache {
    fn new() -> Self {
        Self {
            device: None,
            queue: None,
            pipelines: HashMap::new(),
        }
    }
}

thread_local! {
    static CACHE: RefCell<MetalCache> = RefCell::new(MetalCache::new());
}

/// Reuse the device, command queue, and pipeline for a verified artifact.
///
/// Metal objects stay thread-local because command encoding is intentionally
/// serialized per execution lane; callers can create independent lanes when
/// they need concurrent command submission.
pub(crate) fn with_pipeline<T>(
    artifact: &MetallibArtifact,
    function_name: &str,
    operation: impl FnOnce(
        &metal::Device,
        &metal::CommandQueue,
        &metal::ComputePipelineState,
    ) -> Result<T, String>,
) -> Result<T, String> {
    CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if cache.device.is_none() {
            cache.device = metal::Device::system_default();
        }
        let device = cache
            .device
            .as_ref()
            .cloned()
            .ok_or_else(|| "no system Metal device".to_owned())?;
        if cache.queue.is_none() {
            cache.queue = Some(device.new_command_queue());
        }
        let queue = cache.queue.as_ref().cloned().expect("queue initialized");
        let key = format!(
            "{}:{:x?}:{function_name}",
            artifact.name(),
            artifact.sha256()
        );
        if !cache.pipelines.contains_key(&key) {
            let library = device
                .new_library_with_data(artifact.bytes())
                .map_err(|error| error.to_owned())?;
            let function = library
                .get_function(function_name, None)
                .map_err(|error| error.to_owned())?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| error.to_owned())?;
            cache.pipelines.insert(key.clone(), pipeline);
        }
        let pipeline = cache.pipelines.get(&key).expect("pipeline initialized");
        operation(&device, &queue, pipeline)
    })
}

/// Resolve a stable selector tag to its checked-in Metal function name before
/// entering the shared pipeline cache. This keeps high-value native wrappers
/// (MoE, diffusion, Bonsai) on the same fail-closed catalog as the compiler.
pub(crate) fn with_catalogued_pipeline<T>(
    artifact: &MetallibArtifact,
    tag: &str,
    operation: impl FnOnce(
        &metal::Device,
        &metal::CommandQueue,
        &metal::ComputePipelineState,
    ) -> Result<T, String>,
) -> Result<T, String> {
    let spec = spec_for_tag(tag).ok_or_else(|| format!("unknown native kernel tag '{tag}'"))?;
    with_pipeline(artifact, spec.function, operation)
}
