//! Native Metal function catalog shared by selector and dispatch layers.
//!
//! The plan selector speaks in stable kernel tags while Metal libraries expose
//! concrete function names. Keeping the mapping here makes the boundary
//! auditable and prevents wrappers from silently drifting from checked-in MSL.

use crate::compile::shader_catalog::source_for_tag;
use crate::{ArtifactError, MetallibArtifact, MetallibLoader};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeKernelSpec {
    pub tag: &'static str,
    pub function: &'static str,
}

#[derive(Debug, Error)]
pub enum NativeKernelError {
    #[error("unknown native kernel tag '{0}'")]
    UnknownTag(String),
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
}

/// A verified artifact plus the stable tag-to-function catalog used to look
/// up its Metal entry points. No device API is touched during construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeKernelBundle {
    artifact: MetallibArtifact,
}

impl NativeKernelBundle {
    pub fn load(
        root: impl Into<std::path::PathBuf>,
        manifest: &[u8],
        artifact_name: &str,
    ) -> Result<Self, NativeKernelError> {
        let loader = MetallibLoader::from_manifest_json(root, manifest)?;
        Ok(Self {
            artifact: loader.load(artifact_name)?,
        })
    }

    pub fn artifact(&self) -> &MetallibArtifact {
        &self.artifact
    }

    pub fn resolve(&self, tag: &str) -> Result<NativeKernelBinding<'_>, NativeKernelError> {
        let spec = spec_for_tag(tag).ok_or_else(|| NativeKernelError::UnknownTag(tag.into()))?;
        Ok(NativeKernelBinding {
            artifact: &self.artifact,
            spec,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NativeKernelBinding<'a> {
    artifact: &'a MetallibArtifact,
    spec: NativeKernelSpec,
}

impl<'a> NativeKernelBinding<'a> {
    pub fn artifact(&self) -> &'a MetallibArtifact {
        self.artifact
    }

    pub fn spec(&self) -> NativeKernelSpec {
        self.spec
    }
}

const SPECS: &[NativeKernelSpec] = &[
    NativeKernelSpec {
        tag: "cca_attention",
        function: "cca_block_attend_f32",
    },
    NativeKernelSpec {
        tag: "deltanet",
        function: "deltanet_step_f32",
    },
    NativeKernelSpec {
        tag: "short_conv",
        function: "short_conv1d_step_f32",
    },
    NativeKernelSpec {
        tag: "mamba_scan",
        function: "mamba_selective_scan_f32",
    },
    NativeKernelSpec {
        tag: "retnet",
        function: "retnet_retention_step_f32",
    },
    NativeKernelSpec {
        tag: "rwkv7_time_mix",
        function: "rwkv7_time_mix_f32",
    },
    NativeKernelSpec {
        tag: "denoise",
        function: "diffusion_argmax_confidence_f32",
    },
    NativeKernelSpec {
        tag: "moe_router",
        function: "moe_topk_f32",
    },
    NativeKernelSpec {
        tag: "moe_dispatch",
        function: "moe_grouped_gemm_f32",
    },
    NativeKernelSpec {
        tag: "ternary_pack",
        function: "ternary_gemm_f32",
    },
    NativeKernelSpec {
        tag: "mla_attention",
        function: "mla_cache_attend_f32",
    },
];

#[must_use]
pub fn spec_for_tag(tag: &str) -> Option<NativeKernelSpec> {
    SPECS.iter().copied().find(|spec| spec.tag == tag)
}

#[must_use]
pub fn all_specs() -> &'static [NativeKernelSpec] {
    SPECS
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn every_native_spec_matches_checked_in_msl_function() {
        for spec in all_specs() {
            let source = source_for_tag(spec.tag).expect("spec must have catalogued source");
            assert!(
                source.contains(&format!("kernel void {}", spec.function)),
                "{} is absent from the {} shader source",
                spec.function,
                spec.tag
            );
        }
    }

    #[test]
    fn unknown_tags_fail_closed() {
        assert_eq!(spec_for_tag("unmapped_family"), None);
    }

    #[test]
    fn bundle_resolves_only_allowlisted_artifact_and_known_tag() {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "metal-native-bundle-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let bytes = b"verified-metallib";
        std::fs::write(root.join("bundle.metallib"), bytes).unwrap();
        let digest: [u8; 32] = Sha256::digest(bytes).into();
        let hex = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let manifest = serde_json::json!({
            "artifacts": [{"name": "bundle.metallib", "sha256": hex}]
        });
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let bundle = NativeKernelBundle::load(&root, &manifest_bytes, "bundle.metallib").unwrap();
        let binding = bundle.resolve("ternary_pack").unwrap();
        assert_eq!(binding.spec().function, "ternary_gemm_f32");
        assert!(matches!(
            bundle.resolve("unknown"),
            Err(NativeKernelError::UnknownTag(_))
        ));
        std::fs::remove_dir_all(root).unwrap();
    }
}
