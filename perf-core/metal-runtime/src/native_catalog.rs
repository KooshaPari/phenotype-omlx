//! Native Metal function catalog shared by selector and dispatch layers.
//!
//! The plan selector speaks in stable kernel tags while Metal libraries expose
//! concrete function names. Keeping the mapping here makes the boundary
//! auditable and prevents wrappers from silently drifting from checked-in MSL.

use crate::compile::shader_catalog::source_for_tag;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeKernelSpec {
    pub tag: &'static str,
    pub function: &'static str,
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
}
