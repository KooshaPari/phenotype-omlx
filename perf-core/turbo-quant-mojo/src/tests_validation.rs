// Unit tests for the pre-FFI validation helpers in `validation`.
//
// Split out of `validation.rs` so the production module stays well under
// the size budget. These exercise the helpers directly (no Mojo FFI), so
// they cover malformed outputs without any fake production hooks.

use crate::validation::{
    per_group_packed_bytes, usize_to_isize, validate_decode_inputs, validate_encode_inputs,
    validate_encode_outputs,
};

// ── validate_encode_inputs ──────────────────────────────────────

#[test]
fn validate_encode_inputs_happy_path() {
    let res = validate_encode_inputs(128, 4, 32);
    let (n, n_groups, packed_len) = res.expect("happy path");
    assert_eq!(n, 128);
    assert_eq!(n_groups, 4);
    // group_size=32, bits=4 → per_group_packed_bytes=16, packed_len=64
    assert_eq!(packed_len, 64);
}

#[test]
fn validate_encode_inputs_rejects_empty_data() {
    let err = validate_encode_inputs(0, 4, 32).expect_err("empty data");
    assert!(err.contains("non-empty"), "got: {err}");
}

#[test]
fn validate_encode_inputs_rejects_bits_below_range() {
    let err = validate_encode_inputs(32, 1, 32).expect_err("bits=1");
    assert!(err.contains("bits"), "got: {err}");
}

#[test]
fn validate_encode_inputs_rejects_bits_above_range() {
    let err = validate_encode_inputs(32, 5, 32).expect_err("bits=5");
    assert!(err.contains("bits"), "got: {err}");
}

#[test]
fn validate_encode_inputs_rejects_zero_group_size() {
    let err = validate_encode_inputs(32, 4, 0).expect_err("group_size=0");
    assert!(err.contains("group_size"), "got: {err}");
}

#[test]
fn validate_encode_inputs_rejects_non_divisor_group_size() {
    let err = validate_encode_inputs(33, 4, 32).expect_err("33 % 32 != 0");
    assert!(err.contains("divisible"), "got: {err}");
}

#[test]
fn validate_encode_inputs_rejects_n_over_isize_max() {
    let n = (isize::MAX as usize).wrapping_add(1);
    let err = validate_encode_inputs(n, 4, 1).expect_err("n > isize::MAX");
    assert!(
        err.contains("exceeds") || err.contains("isize"),
        "got: {err}"
    );
}

#[test]
fn validate_encode_inputs_rejects_group_size_over_isize_max() {
    let n = (isize::MAX as usize).wrapping_add(1);
    let err = validate_encode_inputs(n, 4, n).expect_err("group_size > isize::MAX");
    assert!(
        err.contains("exceeds") || err.contains("isize"),
        "got: {err}"
    );
}

// ── validate_decode_inputs ──────────────────────────────────────

#[test]
fn validate_decode_inputs_happy_path() {
    let shape = vec![8];
    let res = validate_decode_inputs(&shape, 4, 4, 4, 8, 2, 4);
    let (n, n_groups, packed_len) = res.expect("happy path");
    assert_eq!(n, 8);
    assert_eq!(n_groups, 4);
    assert_eq!(packed_len, 4);
}

#[test]
fn validate_decode_inputs_rejects_rank_not_one() {
    let shape = vec![2, 4];
    let err = validate_decode_inputs(&shape, 4, 2, 2, 8, 2, 4).expect_err("rank=2");
    assert!(err.contains("shape") && err.contains("rank"), "got: {err}");
}

#[test]
fn validate_decode_inputs_rejects_zero_n() {
    let shape = vec![0];
    let err = validate_decode_inputs(&shape, 0, 0, 0, 0, 2, 4).expect_err("n=0");
    assert!(err.contains("n"), "got: {err}");
}

#[test]
fn validate_decode_inputs_rejects_shape_n_mismatch() {
    let shape = vec![16];
    let err = validate_decode_inputs(&shape, 4, 4, 4, 8, 2, 4).expect_err("shape_n=16, n=8");
    assert!(err.contains("shape"), "got: {err}");
}

#[test]
fn validate_decode_inputs_rejects_wrong_packed_len() {
    let shape = vec![8];
    let err = validate_decode_inputs(&shape, 1, 4, 4, 8, 2, 4).expect_err("packed_len=1");
    assert!(err.contains("packed"), "got: {err}");
}

#[test]
fn validate_decode_inputs_rejects_wrong_scales_len() {
    let shape = vec![8];
    let err = validate_decode_inputs(&shape, 4, 2, 4, 8, 2, 4).expect_err("scales_len=2");
    assert!(err.contains("scales"), "got: {err}");
}

#[test]
fn validate_decode_inputs_rejects_wrong_zeros_len() {
    let shape = vec![8];
    let err = validate_decode_inputs(&shape, 4, 4, 2, 8, 2, 4).expect_err("zeros_len=2");
    assert!(err.contains("zeros"), "got: {err}");
}

// ── validate_encode_outputs (malformed-output validation) ───────

#[test]
fn validate_encode_outputs_accepts_matching_lengths() {
    let res = validate_encode_outputs(1, 4, 4, 4, 1, 4, 4, 4);
    assert_eq!(res, Ok((1, 4, 4, 4)));
}

#[test]
fn validate_encode_outputs_rejects_negative_shape_len() {
    let err = validate_encode_outputs(-1, 4, 4, 4, 1, 4, 4, 4).expect_err("shape_len=-1");
    assert!(err.contains("negative"), "got: {err}");
}

#[test]
fn validate_encode_outputs_rejects_negative_packed_len() {
    let err = validate_encode_outputs(1, -1, 4, 4, 1, 4, 4, 4).expect_err("packed_len=-1");
    assert!(err.contains("negative"), "got: {err}");
}

#[test]
fn validate_encode_outputs_rejects_negative_scales_len() {
    let err = validate_encode_outputs(1, 4, -1, 4, 1, 4, 4, 4).expect_err("scales_len=-1");
    assert!(err.contains("negative"), "got: {err}");
}

#[test]
fn validate_encode_outputs_rejects_negative_zeros_len() {
    let err = validate_encode_outputs(1, 4, 4, -1, 1, 4, 4, 4).expect_err("zeros_len=-1");
    assert!(err.contains("negative"), "got: {err}");
}

#[test]
fn validate_encode_outputs_rejects_shape_len_mismatch() {
    let err = validate_encode_outputs(2, 4, 4, 4, 1, 4, 4, 4).expect_err("shape_len=2 != 1");
    assert!(err.contains("shape_len"), "got: {err}");
}

#[test]
fn validate_encode_outputs_rejects_packed_len_mismatch() {
    let err = validate_encode_outputs(1, 5, 4, 4, 1, 4, 4, 4).expect_err("packed_len=5 != 4");
    assert!(err.contains("packed_len"), "got: {err}");
}

#[test]
fn validate_encode_outputs_rejects_scales_len_mismatch() {
    let err = validate_encode_outputs(1, 4, 5, 4, 1, 4, 4, 4).expect_err("scales_len=5 != 4");
    assert!(err.contains("scales_len"), "got: {err}");
}

#[test]
fn validate_encode_outputs_rejects_zeros_len_mismatch() {
    let err = validate_encode_outputs(1, 4, 4, 5, 1, 4, 4, 4).expect_err("zeros_len=5 != 4");
    assert!(err.contains("zeros_len"), "got: {err}");
}

// ── usize_to_isize (checked conversion) ─────────────────────────

#[test]
fn usize_to_isize_accepts_small_values() {
    assert_eq!(usize_to_isize("n", 0).unwrap(), 0);
    assert_eq!(usize_to_isize("n", 128).unwrap(), 128);
}

#[test]
fn usize_to_isize_accepts_isize_max() {
    let res = usize_to_isize("n", isize::MAX as usize);
    assert_eq!(res, Ok(isize::MAX));
}

#[test]
fn usize_to_isize_rejects_isize_overflow() {
    let v = (isize::MAX as usize).wrapping_add(1);
    let err = usize_to_isize("n", v).expect_err("overflow");
    assert!(
        err.contains("exceeds") || err.contains("isize"),
        "got: {err}"
    );
}

// ── per_group_packed_bytes (checked arithmetic) ─────────────────

#[test]
fn per_group_packed_bytes_happy_4_bits_32_group() {
    // (32*4+7)/8 = 16
    assert_eq!(per_group_packed_bytes(32, 4).unwrap(), 16);
}

#[test]
fn per_group_packed_bytes_carry_rounds_up() {
    // 2*4=8 bits → 1 byte (with rounding: (8+7)/8=1)
    assert_eq!(per_group_packed_bytes(2, 4).unwrap(), 1);
    // 3*4=12 bits → 2 bytes ((12+7)/8=2)
    assert_eq!(per_group_packed_bytes(3, 4).unwrap(), 2);
}

#[test]
fn per_group_packed_bytes_rejects_group_overflow() {
    let gs = (usize::MAX / 4) + 1;
    let err = per_group_packed_bytes(gs, 4).expect_err("overflow");
    assert!(err.contains("overflow"), "got: {err}");
}
