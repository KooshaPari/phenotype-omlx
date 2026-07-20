//! Property 4 — encode → decode round-trip stays within one quant tier
//! of the standard affine-quantization error bound.
//!
//! Plus the fencepost fuzzers (one proptest block per `bits ∈ {2,3,4}`
//! width at three configurations: min group_size, max group_size, and
//! packed-length invariant), plus the ABI version pinning properties.

use super::{
    assert_fencepost_packed_len, assert_fencepost_round_trip, well_formed_request, V1,
};
use native_abi::{encode_v1, AbiVersion, DecodeRequest, EncodeRequest, EncodeResult, Status, ABI_VERSION_CURRENT};
use proptest::prelude::*;

proptest! {
    #[test]
    fn encode_decode_round_trip_stays_within_tolerance(
        (bits, group_size, n, data) in well_formed_request()
    ) {
        // Compute the dispatch payloads.
        let n_groups = n / group_size;
        let packed_len = (n * bits as usize).div_ceil(8);
        let mut shape_storage = vec![0usize; 1];
        let mut packed_storage = vec![0u8; packed_len];
        let mut scales_storage = vec![0.0f32; n_groups];
        let mut zeros_storage = vec![0.0f32; n_groups];
        let mut shape_ptr = shape_storage.as_mut_ptr();
        let mut packed_ptr = packed_storage.as_mut_ptr();
        let mut scales_ptr = scales_storage.as_mut_ptr();
        let mut zeros_ptr = zeros_storage.as_mut_ptr();
        let mut req = EncodeRequest::zeroed();
        req.abi = V1;
        req.data_ptr = data.as_ptr();
        req.n = n;
        req.bits = bits;
        req.group_size = group_size;
        req.out_shape = &mut shape_ptr;
        req.out_shape_capacity = shape_storage.len();
        req.out_packed = &mut packed_ptr;
        req.out_packed_capacity = packed_storage.len();
        req.out_scales = &mut scales_ptr;
        req.out_scales_capacity = scales_storage.len();
        req.out_zeros = &mut zeros_ptr;
        req.out_zeros_capacity = zeros_storage.len();
        let result: EncodeResult = unsafe { encode_v1(&req) };
        prop_assert_eq!(
            result.status,
            Status::Ok,
            "valid well-formed request must encode successfully"
        );

        // Decode through the same ABI.
        let mut decoded = vec![0.0f32; n];
        let mut dreq = DecodeRequest::zeroed();
        dreq.abi = V1;
        dreq.packed_ptr = packed_storage.as_ptr();
        dreq.packed_len = packed_storage.len();
        dreq.scales_ptr = scales_storage.as_ptr();
        dreq.zeros_ptr = zeros_storage.as_ptr();
        dreq.n = n;
        dreq.group_size = group_size;
        dreq.bits = bits;
        dreq.out_ptr = decoded.as_mut_ptr();
        let status = unsafe { native_abi::decode_v1(&dreq) };
        prop_assert_eq!(status, Status::Ok, "decode must succeed for well-formed input");

        // Per-scale tolerance: every decoded value must be within the
        // rounding tier of its group's scale. Skip groups whose scale
        // is degenerate (zero or non-finite) — the affine-quantization
        // contract deliberately returns 0 for an all-equal group, so
        // the round-trip is exact for that case.
        for (g, &scale) in scales_storage.iter().enumerate().take(n_groups) {
            if !scale.is_finite() || scale == 0.0 {
                // All-equal group: the encoder writes zero scale and
                // the decoder produces zero for every element. Verify
                // that and continue.
                for i in 0..group_size {
                    let idx = g * group_size + i;
                    if data[idx] != 0.0 {
                        // Non-zero input mapped to a zero-scale group
                        // means the encoder chose zero as the
                        // quantization level. Forbid that — the
                        // contract is: zero-scale implies zero-input.
                        prop_assert!(
                            data[idx] == 0.0,
                            "[g={g}, i={i}] non-zero input {} produced zero scale {}",
                            data[idx],
                            scale
                        );
                    }
                }
                continue;
            }
            // Affine quantization: with bits levels per element the
            // step size is exactly `scale`, so the round-trip error
            // is bounded by ±scale/2. Use that as the tolerance and
            // add a tiny epsilon for fp rounding.
            let tolerance = scale.abs() / 2.0 + 1e-5;
            for i in 0..group_size {
                let idx = g * group_size + i;
                let decoded_val = decoded[idx];
                let input_val = data[idx];
                let delta = (decoded_val - input_val).abs();
                prop_assert!(
                    delta.is_finite(),
                    "[g={g}, i={i}, bits={bits}] delta is not finite: {delta} \
                     decoded={} input={} scale={scale}",
                    decoded_val,
                    input_val,
                    bits = bits,
                    scale = scale
                );
                prop_assert!(
                    delta <= tolerance,
                    "[g={g}, i={i}, bits={bits}] delta={delta} > tolerance={tolerance}; \
                     decoded={} input={} scale={scale}",
                    decoded_val,
                    input_val,
                    bits = bits,
                    scale = scale
                );
            }
        }
    }

    // Property 5: ABI version 1 is compatible only with major = 1.
    // (Compatible regardless of minor in either direction.)
    #[test]
    fn abi_compatibility_pins_major(
        minor_a in 0u16..16,
        minor_b in 0u16..16,
    ) {
        let va = AbiVersion { major: 1, minor: minor_a };
        let vb = AbiVersion { major: 1, minor: minor_b };
        prop_assert!(native_abi::is_compatible(va, vb));
        prop_assert!(native_abi::is_compatible(vb, va));

        let v_other = AbiVersion { major: 2, minor: minor_a };
        prop_assert!(!native_abi::is_compatible(va, v_other));
    }

    // Property 6: current ABI version is always (major=1, minor=0) —
    // pinning the constant here catches a regression where the version
    // is bumped without updating the contract test.
    #[test]
    fn current_abi_version_is_pinned_to_v1(_unused in Just(())) {
        prop_assert_eq!(ABI_VERSION_CURRENT.major, 1);
        prop_assert_eq!(ABI_VERSION_CURRENT.minor, 0);
    }
}

proptest! {
    #[test]
    fn fencepost_2bit_min_encode_decodes(
        data in proptest::collection::vec(
            (-1.0e3f32..1.0e3f32).prop_filter("finite", |v| v.is_finite()),
            8,
        ),
    ) {
        // bits=2 (3 levels), group_size=1 — minimum legal group.
        assert_fencepost_round_trip(&data, 8, 1, 2);
    }
    #[test]
    fn fencepost_2bit_max_group_size(
        data in proptest::collection::vec(
            (-1.0e3f32..1.0e3f32).prop_filter("finite", |v| v.is_finite()),
            256,
        ),
    ) {
        // bits=2, group_size=256 → 1 group; catches capacity miscounts.
        assert_fencepost_round_trip(&data, 1, 256, 2);
    }
    #[test]
    fn fencepost_2bit_packed_len_matches_expected(
        n in 1usize..128,
        data in proptest::collection::vec(
            (-1.0f32..1.0f32).prop_filter("finite", |v| v.is_finite()),
            128,
        ),
    ) {
        assert_fencepost_packed_len(&data, n, 2);
    }
    #[test]
    fn fencepost_3bit_min_encode_decodes(
        data in proptest::collection::vec(
            (-1.0e3f32..1.0e3f32).prop_filter("finite", |v| v.is_finite()),
            8,
        ),
    ) {
        // bits=3 (7 levels, odd width), group_size=1.
        assert_fencepost_round_trip(&data, 8, 1, 3);
    }
    #[test]
    fn fencepost_3bit_max_group_size(
        data in proptest::collection::vec(
            (-1.0e3f32..1.0e3f32).prop_filter("finite", |v| v.is_finite()),
            256,
        ),
    ) {
        assert_fencepost_round_trip(&data, 1, 256, 3);
    }
    #[test]
    fn fencepost_3bit_packed_len_matches_expected(
        n in 1usize..128,
        data in proptest::collection::vec(
            (-1.0f32..1.0f32).prop_filter("finite", |v| v.is_finite()),
            128,
        ),
    ) {
        assert_fencepost_packed_len(&data, n, 3);
    }
    #[test]
    fn fencepost_4bit_min_encode_decodes(
        data in proptest::collection::vec(
            (-1.0e3f32..1.0e3f32).prop_filter("finite", |v| v.is_finite()),
            8,
        ),
    ) {
        // bits=4 (15 levels, max width), group_size=1.
        assert_fencepost_round_trip(&data, 8, 1, 4);
    }
    #[test]
    fn fencepost_4bit_max_group_size(
        data in proptest::collection::vec(
            (-1.0e3f32..1.0e3f32).prop_filter("finite", |v| v.is_finite()),
            256,
        ),
    ) {
        assert_fencepost_round_trip(&data, 1, 256, 4);
    }
    #[test]
    fn fencepost_4bit_packed_len_matches_expected(
        n in 1usize..128,
        data in proptest::collection::vec(
            (-1.0f32..1.0f32).prop_filter("finite", |v| v.is_finite()),
            128,
        ),
    ) {
        assert_fencepost_packed_len(&data, n, 4);
    }
}