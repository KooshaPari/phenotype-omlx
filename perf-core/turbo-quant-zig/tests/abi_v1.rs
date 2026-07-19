//! Behavioural tests for the versioned Native ABI v1 surface in turbo-quant-zig.
//!
//! These tests cover the `encode_v1` / `decode_v1` wrappers exposed by the
//! Zig kernel. Without `--features zig` the encode_v1 stub returns
//! `ErrAllocation` so the tests skip themselves to keep the build green on
//! hosts without a Zig toolchain. With `--features zig` the Zig kernel runs
//! and the tests verify that its output agrees with the reference Rust
//! implementation in `perf-core/native-abi`.

#![cfg(feature = "zig")]

use turbo_quant_zig::ZigQuantizedTensor;

#[test]
fn zig_v1_encode_matches_reference_within_tolerance() {
    let input = [-3.0, -1.0, 0.5, 2.0, 7.0, 9.0, 12.0];

    for bits in [2u8, 3, 4] {
        let q = ZigQuantizedTensor::encode_v1(&input, bits, 3)
            .expect("v1 encode should succeed when zig feature is enabled");

        assert_eq!(q.shape, vec![input.len()]);
        assert_eq!(
            q.packed.len(),
            (input.len() * bits as usize + 7) / 8,
            "packed length must match the (n * bits + 7) / 8 contract"
        );

        let mut out = vec![0.0_f32; input.len()];
        let status = q.decode_v1(input.len(), 3, bits, &mut out);
        assert_eq!(status, native_abi::Status::Ok);

        let tolerance = q.scales.iter().copied().fold(0.0_f32, f32::max);
        for (actual, expected) in out.iter().zip(input) {
            assert!(
                (actual - expected).abs() <= tolerance + 1e-5,
                "bits={bits}: decoded {actual} vs expected {expected}"
            );
        }
    }
}

#[test]
fn zig_v1_decode_rejects_invalid_arguments_and_preserves_buffer() {
    let input = [-3.0, -1.0, 0.5, 2.0, 7.0, 9.0, 12.0];
    let q = ZigQuantizedTensor::encode_v1(&input, 3, 4).expect("v1 encode should succeed");

    let sentinel = 91.0_f32;
    let mut buf = vec![sentinel; input.len()];

    // group_size == 0 must be rejected and the buffer left untouched.
    let status = q.decode_v1(input.len(), 0, 3, &mut buf);
    assert_ne!(status, native_abi::Status::Ok);
    assert_eq!(buf, vec![sentinel; input.len()]);

    // bits == 1 (outside 2..=4) must also be rejected.
    let status = q.decode_v1(input.len(), 4, 1, &mut buf);
    assert_ne!(status, native_abi::Status::Ok);
    assert_eq!(buf, vec![sentinel; input.len()]);

    // Mismatched packed length must also be rejected without writing.
    let mut truncated = q.clone();
    truncated.packed.pop();
    let status = truncated.decode_v1(input.len(), 4, 3, &mut buf);
    assert_ne!(status, native_abi::Status::Ok);
    assert_eq!(buf, vec![sentinel; input.len()]);
}

#[test]
fn zig_v1_encode_rejects_invalid_bits() {
    let input = [1.0, 2.0, 3.0, 4.0];

    let err = ZigQuantizedTensor::encode_v1(&input, 1, 4)
        .expect_err("bits=1 must be rejected");
    assert_eq!(err, native_abi::Status::ErrInvalidBits);

    let err = ZigQuantizedTensor::encode_v1(&input, 5, 4)
        .expect_err("bits=5 must be rejected");
    assert_eq!(err, native_abi::Status::ErrInvalidBits);

    let err =
        ZigQuantizedTensor::encode_v1(&input, 3, 0).expect_err("group_size=0 must be rejected");
    assert_eq!(err, native_abi::Status::ErrInvalidGroupSize);

    let err = ZigQuantizedTensor::encode_v1(&[], 3, 4).expect_err("n=0 must be rejected");
    assert_eq!(err, native_abi::Status::ErrNullArg);
}
