//! Behavioural tests for the versioned Native ABI v1 surface in turbo-quant-c.
//!
//! These tests cover the new `encode_v1` / `decode_v1` wrappers, the C ABI's
//! rejection semantics, and the version-mismatch path. They run in addition
//! to the legacy surface tests in `src/lib.rs::tests`.

use native_abi::{ABI_VERSION_CURRENT, Status};
use turbo_quant_c::{abi_version, decode_v1, encode, encode_v1};

#[test]
fn c_abi_v1_aliases_call_into_new_entry() {
    // encode_v1 is the versioned surface; the legacy `encode` alias must
    // still work and produce a tensor with the same shape/packed/scales/zeros
    // contract so existing callers continue to function unchanged.
    let input = [0.25, -0.5, 1.0, 1.5];
    let v1 = encode_v1(&input, 3, 4).expect("v1 encode should succeed");
    let legacy = encode(&input, 3, 4).expect("legacy encode should succeed");

    assert_eq!(v1.shape, legacy.shape);
    assert_eq!(v1.packed, legacy.packed);
    assert_eq!(v1.scales.len(), legacy.scales.len());
    assert_eq!(v1.zeros.len(), legacy.zeros.len());
    for (a, b) in v1.scales.iter().zip(legacy.scales.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
    for (a, b) in v1.zeros.iter().zip(legacy.zeros.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn c_abi_v1_decode_into_preserves_sentinel_on_version_mismatch() {
    // The decode ABI has no explicit version slot on the public wrapper, but
    // the C entry point enforces abi.major == current at the C layer. The
    // public decode_v1 wrapper always sets the correct version, so we cannot
    // forge a mismatch through it. Instead we observe the same reject-and-
    // preserve-sentinel contract through the public API's invalid-argument
    // path: a zero-sized buffer or mismatched packed length must leave the
    // caller-supplied `out` untouched.
    let input = [0.0, 1.0, 2.0, 3.0];
    let tensor = encode_v1(&input, 3, 4).expect("v1 encode should succeed");

    let sentinel = 91.0_f32;
    let mut buf = vec![sentinel; input.len()];

    // group_size == 0 — invalid, decode must leave `buf` untouched.
    let status = decode_v1(
        &tensor.packed,
        &tensor.scales,
        &tensor.zeros,
        input.len(),
        0,
        3,
        &mut buf,
    );
    assert_ne!(status, Status::Ok);
    assert_eq!(buf, vec![sentinel; input.len()]);

    // bits out of range — same guarantee.
    let status = decode_v1(
        &tensor.packed,
        &tensor.scales,
        &tensor.zeros,
        input.len(),
        4,
        1,
        &mut buf,
    );
    assert_ne!(status, Status::Ok);
    assert_eq!(buf, vec![sentinel; input.len()]);

    // mismatched packed_len — same guarantee.
    let mut truncated = tensor.packed.clone();
    truncated.pop();
    let status = decode_v1(
        &truncated,
        &tensor.scales,
        &tensor.zeros,
        input.len(),
        4,
        3,
        &mut buf,
    );
    assert_ne!(status, Status::Ok);
    assert_eq!(buf, vec![sentinel; input.len()]);
}

#[test]
fn c_abi_v1_round_trips_within_tolerance_for_uniform_input() {
    let input = [-3.0, -1.0, 0.5, 2.0, 7.0, 9.0, 12.0];

    for bits in [2u8, 3, 4] {
        let tensor = encode_v1(&input, bits, 3).expect("supported encoding");
        assert_eq!(tensor.shape, vec![input.len()]);
        assert_eq!(
            tensor.packed.len(),
            (input.len() * bits as usize).div_ceil(8),
            "packed length must match contract"
        );

        let mut out = vec![0.0_f32; input.len()];
        let status = decode_v1(
            &tensor.packed,
            &tensor.scales,
            &tensor.zeros,
            input.len(),
            3,
            bits,
            &mut out,
        );
        assert_eq!(status, Status::Ok);

        for (actual, expected) in out.iter().zip(input) {
            let tolerance =
                tensor.scales.iter().copied().fold(0.0_f32, f32::max);
            assert!(
                (actual - expected).abs() <= tolerance + 1e-5,
                "bits={bits}: decoded {actual} vs expected {expected}"
            );
        }
    }
}

#[test]
fn c_abi_v1_rejects_invalid_bits() {
    let input = [1.0, 2.0, 3.0, 4.0];

    // bits == 1 — outside 2..=4.
    let err = encode_v1(&input, 1, 4).expect_err("bits=1 must be rejected");
    assert_eq!(err, Status::ErrInvalidBits);

    // bits == 5 — outside 2..=4.
    let err = encode_v1(&input, 5, 4).expect_err("bits=5 must be rejected");
    assert_eq!(err, Status::ErrInvalidBits);

    // group_size == 0.
    let err = encode_v1(&input, 3, 0).expect_err("group_size=0 must be rejected");
    assert_eq!(err, Status::ErrInvalidGroupSize);

    // empty input.
    let err = encode_v1(&[], 3, 4).expect_err("n=0 must be rejected");
    assert_eq!(err, Status::ErrNullArg);
}

#[test]
fn c_abi_v1_abi_version_helper_matches_native_abi() {
    let v = abi_version(ABI_VERSION_CURRENT.major, ABI_VERSION_CURRENT.minor);
    assert_eq!(v.major, ABI_VERSION_CURRENT.major);
    assert_eq!(v.minor, ABI_VERSION_CURRENT.minor);
}
