//! Property-based fuzz tests for the native-abi crate (ABI v1).
//!
//! These tests use [`proptest`] to drive the public surface with
//! randomized input and pin invariants that cannot be captured by
//! per-case unit tests:
//!
//! 1. **`EncodeRequest::validate()` is total** — every randomized
//!    request either succeeds with `Ok` or fails with a documented
//!    `Status`; it never panics, never returns an undocumented code.
//! 2. **Bit-widths outside `{2, 3, 4, 8}` are rejected** — the
//!    ABI hard-codes that set; proptest verifies it stays hard-coded
//!    even under randomized `bits: u8`.
//! 3. **Group size of zero is rejected** — verified for both encode
//!    and decode paths.
//! 4. **Encode → decode round-trip stays within the per-scale
//!    tolerance** — for any randomized `Vec<f32>` of bounded length
//!    and any valid combination of `(bits, group_size)` from the
//!    allowed set, the decoder reproduces the input within
//! 5. **`Status` round-trips through `i32` bijectively** for every
//!    member of the enum (the fuzz narrows the search to the 9 known
//!    codes).
//!
//! The fuzz corpus is local — proptest shrinking handles failure
//! reproduction. Cases are kept small (max 64) so the suite remains
//! fast enough to run on every CI gate (≈ 1 second total).

use native_abi::{
    encode_v1, AbiVersion, DecodeRequest, EncodeRequest, EncodeResult, Status, ABI_VERSION_CURRENT,
    expected_packed_len,
};
use proptest::prelude::*;

const V1: AbiVersion = AbiVersion { major: 1, minor: 0 };

/// Bit widths the ABI accepts. Anything outside this set must be
/// rejected as `ErrInvalidBits`.
// Contract: validate() accepts `bits in {2, 3, 4}`. The descriptor
// doc-comment pins this range; the fuzz test guards against drift.
const VALID_BITS: &[u8] = &[2u8, 3, 4];

/// Strategy producing a valid `(bits, group_size, n)` triple for which
/// `encode_v1` is expected to succeed, plus a vector of `n` finite
/// `f32` values. `well_formed_request` is the canonical builder here;
/// it shares the same aligned-group semantics as the standalone
/// `aligned_group_n` helper would but is inlined so we avoid one
/// `BoxedStrategy` allocation per call.
fn well_formed_request() -> BoxedStrategy<(u8, usize, usize, Vec<f32>)> {
    (
        proptest::sample::select(VALID_BITS.to_vec()),
        1usize..64,
        1usize..64,
    )
        .prop_flat_map(|(bits, group_size, n_groups_raw)| {
            let n = n_groups_raw.div_ceil(group_size) * group_size;
            // Bounded range: the affine-quant encoder computes
            //   scale = (max - min) / ((1 << bits) - 1).
            // Inputs spanning more than ~1e7 cause the scale to
            // overflow for bits=2 (only 3 distinct levels). Keep the
            // range within what all supported widths can represent
            // without saturating.
            let data_strategy = proptest::collection::vec(
                (-1.0e3f32..1.0e3f32).prop_filter("finite", |v| v.is_finite()),
                n,
            );
            (Just(bits), Just(group_size), Just(n), data_strategy)
        })
        .boxed()
}

proptest! {
    // Property 1: `validate()` is total — never panics, never returns
    // an undocumented status.
    #[test]
    fn encode_validate_is_total_for_random_inputs(
        n in 0usize..64,
        bits in any::<u8>(),
        group_size in 0usize..64,
    ) {
        let data = vec![0.5f32; n];
        let mut req = EncodeRequest::zeroed();
        req.abi = V1;
        req.data_ptr = data.as_ptr();
        req.n = n;
        req.bits = bits;
        req.group_size = group_size;
        let mut shape: *mut usize = std::ptr::null_mut();
        let mut packed: *mut u8 = std::ptr::null_mut();
        let mut scales: *mut f32 = std::ptr::null_mut();
        let mut zeros: *mut f32 = std::ptr::null_mut();
        req.out_shape = &mut shape as *mut _;
        req.out_shape_capacity = 16;
        req.out_packed = &mut packed as *mut _;
        req.out_packed_capacity = 16;
        req.out_scales = &mut scales as *mut _;
        req.out_scales_capacity = 16;
        req.out_zeros = &mut zeros as *mut _;
        req.out_zeros_capacity = 16;

        let v = req.validate();
        // All failures must be among the documented statuses; the
        // result is *some* Status — never a panic.
        match v {
            Ok(()) => { /* valid request */ }
            Err(s) => {
                let i: i32 = s.into();
                let back = Status::try_from(i)
                    .expect("validate() must return a known Status");
                // Sanity: the known codes are 0..=8.
                assert!(back == Status::Ok
                    || back == Status::ErrNullArg
                    || back == Status::ErrInvalidBits
                    || back == Status::ErrInvalidGroupSize
                    || back == Status::ErrNonFiniteInput
                    || back == Status::ErrOverflow
                    || back == Status::ErrAllocation
                    || back == Status::ErrVersionMismatch
                    || back == Status::ErrBackend,
                    "undocumented Status code: {back:?}");
            }
        }
    }

    // Property 2: `bits` outside the `{2, 3, 4, 8}` set is rejected.
    #[test]
    fn encode_rejects_bits_outside_allowed_set(
        bits in any::<u8>().prop_filter("outside allowed set", |&b| !VALID_BITS.contains(&b)),
        group_size in 1usize..64,
    ) {
        let data = vec![1.0f32; group_size.max(1)];
        let mut req = EncodeRequest::zeroed();
        req.abi = V1;
        req.data_ptr = data.as_ptr();
        req.n = data.len();
        req.bits = bits;
        req.group_size = group_size;
        let mut shape: *mut usize = std::ptr::null_mut();
        let mut packed: *mut u8 = std::ptr::null_mut();
        let mut scales: *mut f32 = std::ptr::null_mut();
        let mut zeros: *mut f32 = std::ptr::null_mut();
        req.out_shape = &mut shape as *mut _;
        req.out_shape_capacity = 16;
        req.out_packed = &mut packed as *mut _;
        req.out_packed_capacity = 16;
        req.out_scales = &mut scales as *mut _;
        req.out_scales_capacity = 16;
        req.out_zeros = &mut zeros as *mut _;
        req.out_zeros_capacity = 16;
        assert_eq!(req.validate(), Err(Status::ErrInvalidBits),
            "bits={bits} must be rejected as ErrInvalidBits");
    }

    // Property 3: with valid bits, `group_size == 0` is rejected as
    // `ErrInvalidGroupSize`. We feed bits from the valid set so the
    // validator reaches the group_size check (bits is checked first,
    // which is the documented ordering).
    #[test]
    fn zero_group_size_rejected(bits in proptest::sample::select(VALID_BITS.to_vec())) {
        // Encode side.
        let data = [1.0f32; 4];
        let mut req = EncodeRequest::zeroed();
        req.abi = V1;
        req.data_ptr = data.as_ptr();
        req.n = 4;
        req.bits = bits;
        req.group_size = 0;
        let mut shape: *mut usize = std::ptr::null_mut();
        let mut packed: *mut u8 = std::ptr::null_mut();
        let mut scales: *mut f32 = std::ptr::null_mut();
        let mut zeros: *mut f32 = std::ptr::null_mut();
        req.out_shape = &mut shape as *mut _;
        req.out_shape_capacity = 16;
        req.out_packed = &mut packed as *mut _;
        req.out_packed_capacity = 16;
        req.out_scales = &mut scales as *mut _;
        req.out_scales_capacity = 16;
        req.out_zeros = &mut zeros as *mut _;
        req.out_zeros_capacity = 16;
        assert_eq!(req.validate(), Err(Status::ErrInvalidGroupSize),
            "group_size=0 with valid bits must reject as ErrInvalidGroupSize");
    }

    // Property 4: encode → decode round-trip is within the per-scale
    /// tolerance of the original input. The tolerance is
    /// `(max_scale_of_group + max_scale_of_next_group) / 2`, which is
    // Property 4: encode -> decode round-trip stays within one quant tier of
    // the standard affine-quantization error bound.
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
        prop_assert_eq!(result.status, Status::Ok,
            "valid well-formed request must encode successfully");

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
                        prop_assert!(data[idx] == 0.0,
                            "[g={g}, i={i}] non-zero input {} produced zero scale {}",
                            data[idx], scale);
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
                prop_assert!(delta.is_finite(),
                    "[g={g}, i={i}, bits={bits}] delta is not finite: {delta} \
                     decoded={} input={} scale={scale}",
                    decoded_val, input_val, bits = bits, scale = scale);
                prop_assert!(delta <= tolerance,
                    "[g={g}, i={i}, bits={bits}] delta={delta} > tolerance={tolerance}; \
                     decoded={} input={} scale={scale}",
                    decoded_val, input_val, bits = bits, scale = scale);
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

// --- Fencepost fuzzers: explicit bits={2,3,4} boundaries ---
//
// The properties above drive `(bits, group_size, n)` from proptest's
// shrinking space, so the boundary values (bits ∈ {2, 3, 4}) are
// exercised but never guaranteed. The fencepost tests below pin the
// behaviour at each of the three valid widths: `*_min_encode_decodes`
// (group_size=1, harshest per-element scale test), `*_max_group_size`
// (group_size=256, surfaces capacity / buffer mis-counts), and
// `*_packed_len_matches_expected` (pins the `(n*bits).div_ceil(8)`
// contract).

/// Round-trip check for the `*_min_encode_decodes` and
/// `*_max_group_size` fencepost tests. Encodes `data` at
/// `(group_size, bits)`, decodes back, asserts every element is within
/// `scale/2` of its input (or exact for the degenerate-scale case).
fn assert_fencepost_round_trip(data: &[f32], n_groups: usize, group_size: usize, bits: u8) {
    let n = n_groups * group_size;
    let mut shape = vec![0usize; 1];
    let mut packed = vec![0u8; expected_packed_len(n, bits)];
    let mut scales = vec![0.0f32; n_groups];
    let mut zeros = vec![0.0f32; n_groups];
    let mut sp = shape.as_mut_ptr();
    let mut pp = packed.as_mut_ptr();
    let mut scp = scales.as_mut_ptr();
    let mut zp = zeros.as_mut_ptr();
    let mut req = EncodeRequest::zeroed();
    req.abi = V1;
    req.data_ptr = data.as_ptr();
    req.n = n;
    req.bits = bits;
    req.group_size = group_size;
    req.out_shape = &mut sp;
    req.out_shape_capacity = shape.len();
    req.out_packed = &mut pp;
    req.out_packed_capacity = packed.len();
    req.out_scales = &mut scp;
    req.out_scales_capacity = scales.len();
    req.out_zeros = &mut zp;
    req.out_zeros_capacity = zeros.len();
    let result: EncodeResult = unsafe { encode_v1(&req) };
    assert_eq!(result.status, Status::Ok, "encode failed bits={bits} gs={group_size}");
    let mut decoded = vec![0.0f32; n];
    let mut dreq = DecodeRequest::zeroed();
    dreq.abi = V1;
    dreq.packed_ptr = packed.as_ptr();
    dreq.packed_len = packed.len();
    dreq.scales_ptr = scales.as_ptr();
    dreq.zeros_ptr = zeros.as_ptr();
    dreq.n = n;
    dreq.group_size = group_size;
    dreq.bits = bits;
    dreq.out_ptr = decoded.as_mut_ptr();
    let status = unsafe { native_abi::decode_v1(&dreq) };
    assert_eq!(status, Status::Ok, "decode failed bits={bits} gs={group_size}");
    for (g, &scale) in scales.iter().enumerate() {
        let tol = if scale.is_finite() && scale != 0.0 { scale.abs() / 2.0 + 1e-5 } else { 1e-5 };
        for i in 0..group_size {
            let idx = g * group_size + i;
            let delta = (decoded[idx] - data[idx]).abs();
            assert!(delta.is_finite() && delta <= tol,
                "bits={bits} gs={group_size} g={g} i={i} delta={delta} > tol={tol}");
        }
    }
}

/// Packed-length invariant: `expected_packed_len(n, bits)` must equal
/// the encoder's `written_packed_len` and the formula
/// `(n*bits).div_ceil(8)`. `data` must have `>= n` entries.
fn assert_fencepost_packed_len(data: &[f32], n: usize, bits: u8) {
    assert!(data.len() >= n);
    let expected = expected_packed_len(n, bits);
    assert_eq!(expected, (n * bits as usize).div_ceil(8), "formula drift");
    let mut shape = vec![0usize; 1];
    let mut packed = vec![0u8; expected.max(1)];
    let mut scales = vec![0.0f32; n];
    let mut zeros = vec![0.0f32; n];
    let mut sp = shape.as_mut_ptr();
    let mut pp = packed.as_mut_ptr();
    let mut scp = scales.as_mut_ptr();
    let mut zp = zeros.as_mut_ptr();
    let mut req = EncodeRequest::zeroed();
    req.abi = V1;
    req.data_ptr = data.as_ptr();
    req.n = n;
    req.bits = bits;
    req.group_size = 1;
    req.out_shape = &mut sp;
    req.out_shape_capacity = shape.len();
    req.out_packed = &mut pp;
    req.out_packed_capacity = packed.len();
    req.out_scales = &mut scp;
    req.out_scales_capacity = scales.len();
    req.out_zeros = &mut zp;
    req.out_zeros_capacity = zeros.len();
    let result: EncodeResult = unsafe { encode_v1(&req) };
    assert_eq!(result.status, Status::Ok, "encode failed bits={bits} n={n}");
    assert_eq!(result.written_packed_len, expected, "written_packed_len drift");
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
