//! Catalog of checked-in Metal kernels used by the runtime compiler.
//!
//! This is deliberately a source catalog, not a claim that a device compiled or
//! executed the source. Native compilation and dispatch remain explicit gates.

pub(crate) fn source_for_tag(tag: &str) -> Option<&'static str> {
    Some(match tag {
        "cca_attention" => include_str!("../../shaders/cca_block_attend.metal"),
        "deltanet" | "deltanet_batched" => include_str!("../../shaders/deltanet_step.metal"),
        "short_conv" => include_str!("../../shaders/short_conv1d.metal"),
        "mamba_scan" | "mamba_selective_scan" => {
            include_str!("../../shaders/mamba_selective_scan.metal")
        }
        "retnet" => include_str!("../../shaders/retnet_retention_step.metal"),
        "rwkv7_time_mix" => include_str!("../../shaders/rwkv7_time_mix.metal"),
        "denoise" => include_str!("../../shaders/diffusion_argmax_confidence.metal"),
        "active_compact" => include_str!("../../shaders/diffusion_active_compact.metal"),
        "remask" => include_str!("../../shaders/diffusion_remask.metal"),
        "moe_router" => include_str!("../../shaders/moe_topk.metal"),
        "moe_dispatch" | "moe_reduce" | "moe_shared" => {
            include_str!("../../shaders/moe_grouped_gemm.metal")
        }
        "ternary_pack" => include_str!("../../shaders/ternary_gemm.metal"),
        "mla_attention" => include_str!("../../shaders/mla_cache_attend.metal"),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::source_for_tag;

    #[test]
    fn mapped_sources_are_non_empty_and_kernel_shaped() {
        for tag in [
            "cca_attention",
            "deltanet",
            "short_conv",
            "mamba_selective_scan",
            "retnet",
            "rwkv7_time_mix",
            "denoise",
            "active_compact",
            "remask",
            "moe_router",
            "moe_dispatch",
            "ternary_pack",
            "mla_attention",
        ] {
            let source = source_for_tag(tag).expect("catalogued source");
            assert!(!source.trim().is_empty(), "empty source for {tag}");
            assert!(
                source.contains("kernel void"),
                "not a kernel source for {tag}"
            );
        }
    }
}
