#!/usr/bin/env bash
set -euo pipefail

# Build the production MoE shader artifacts consumed by metal-runtime/tests/moe.rs.
# Usage: OUT_DIR=/tmp/omlx-moe-metallib ./scripts/build_moe_metallibs.sh

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${OUT_DIR:-$(mktemp -d /tmp/omlx-moe-metallib.XXXXXX)}"
mkdir -p "$out_dir"

metal_bin="$(xcrun --sdk macosx --find metal)"
metallib_bin="$(xcrun --sdk macosx --find metallib)"
shader_dir="$repo_root/perf-core/metal-runtime/shaders"

for name in adaln_rms flow_cfg_step joint_attention rope_3d temporal_window_attention moe_topk moe_grouped_gemm ternary_gemm diffusion_argmax_confidence short_conv1d rwkv7_time_mix mamba_selective_step mamba_selective_scan deltanet_step cca_block_attend mla_cache_attend retnet_retention_step; do
  "$metal_bin" -c "$shader_dir/$name.metal" -o "$out_dir/$name.air"
  "$metallib_bin" "$out_dir/$name.air" -o "$out_dir/$name.metallib"
done

echo "MOE_TOPK_METALLIB=$out_dir/moe_topk.metallib"
echo "ADALN_METALLIB=$out_dir/adaln_rms.metallib"
echo "FLOW_STEP_METALLIB=$out_dir/flow_cfg_step.metallib"
echo "JOINT_ATTENTION_METALLIB=$out_dir/joint_attention.metallib"
echo "ROPE_3D_METALLIB=$out_dir/rope_3d.metallib"
echo "TEMPORAL_ATTN_METALLIB=$out_dir/temporal_window_attention.metallib"
echo "MOE_GROUPED_GEMM_METALLIB=$out_dir/moe_grouped_gemm.metallib"
echo "TERNARY_GEMM_METALLIB=$out_dir/ternary_gemm.metallib"
echo "DIFFUSION_CONFIDENCE_METALLIB=$out_dir/diffusion_argmax_confidence.metallib"
echo "SHORT_CONV_METALLIB=$out_dir/short_conv1d.metallib"
echo "RWKV_METALLIB=$out_dir/rwkv7_time_mix.metallib"
echo "MAMBA_METALLIB=$out_dir/mamba_selective_step.metallib"
echo "MAMBA_SCAN_METALLIB=$out_dir/mamba_selective_scan.metallib"
echo "DELTANET_METALLIB=$out_dir/deltanet_step.metallib"
echo "CCA_METALLIB=$out_dir/cca_block_attend.metallib"
echo "MLA_CACHE_METALLIB=$out_dir/mla_cache_attend.metallib"
echo "RETNET_METALLIB=$out_dir/retnet_retention_step.metallib"
shasum -a 256 "$out_dir"/*.metallib
