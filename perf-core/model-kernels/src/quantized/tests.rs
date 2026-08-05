//! Tests for the ternary / sub-byte pack-unpack kernels.
//!
//! See the parent module docs for the symmetric per-group quantization
//! scheme.

use super::subbyte::{subbyte_pack, subbyte_unpack};
use super::ternary::{ternary_pack, ternary_repack_for_metal, ternary_unpack, SignedTernary};
use crate::error::KernelError;

#[test]
fn ternary_pack_layout_matches_bit_table() {
    let values = vec![
        SignedTernary::Zero,
        SignedTernary::Pos,
        SignedTernary::Neg,
        SignedTernary::Pos,
        SignedTernary::Zero,
        SignedTernary::Neg,
        SignedTernary::Neg,
        SignedTernary::Pos,
    ];
    let (packed, scales, zeros) = ternary_pack(&values, 8).unwrap();
    assert_eq!(packed.len(), 2);
    assert_eq!(scales.len(), 1);
    assert_eq!(zeros.len(), 1);
    assert_eq!(packed[0], 0b01_10_01_00);
    assert_eq!(packed[1], 0b01_10_10_00);
}

#[test]
fn ternary_round_trip_inverts_pack() {
    let values = vec![
        SignedTernary::Pos,
        SignedTernary::Zero,
        SignedTernary::Neg,
        SignedTernary::Pos,
        SignedTernary::Neg,
        SignedTernary::Pos,
        SignedTernary::Zero,
        SignedTernary::Pos,
    ];
    let (packed, scales, zeros) = ternary_pack(&values, 8).unwrap();
    let mut out = vec![SignedTernary::Zero; values.len()];
    ternary_unpack(&packed, &scales, &zeros, values.len(), 8, &mut out).unwrap();
    assert_eq!(out, values);
}

#[test]
fn ternary_pack_zero_group_size_is_error() {
    let err = ternary_pack(&[], 0).unwrap_err();
    assert!(matches!(err, KernelError::ZeroDimension { .. }));
}

#[test]
fn ternary_partial_trailing_group_packs_cleanly() {
    let values = vec![SignedTernary::Pos, SignedTernary::Neg, SignedTernary::Zero];
    let (packed, scales, zeros) = ternary_pack(&values, 4).unwrap();
    // 3 values, group_size=4: a single trailing group is
    // emitted with the first three slots populated; the
    // remaining slot is Zero. Therefore packed is one byte
    // and scales/zeros have length 1.
    assert_eq!(packed.len(), 1);
    assert_eq!(scales.len(), 1);
    assert_eq!(zeros.len(), 1);
    let mut out = vec![SignedTernary::Zero; values.len()];
    ternary_unpack(&packed, &scales, &zeros, values.len(), 4, &mut out).unwrap();
    assert_eq!(out, values);
}

#[test]
fn ternary_repack_matches_metal_column_major_layout_with_k_tail() {
    let k = 5;
    let n = 3;
    let values: Vec<SignedTernary> = (0..k * n)
        .map(|i| match i % 3 {
            0 => SignedTernary::Pos,
            1 => SignedTernary::Neg,
            _ => SignedTernary::Zero,
        })
        .collect();
    let (host, _, _) = ternary_pack(&values, values.len()).unwrap();
    assert_eq!(host.len(), 4);
    let metal = ternary_repack_for_metal(&host, k, n).unwrap();
    assert_eq!(metal.len(), n * k.div_ceil(4));

    for col in 0..n {
        for row in 0..k {
            let host_index = row * n + col;
            let host_code = (host[host_index / 4] >> ((host_index % 4) * 2)) & 0b11;
            let metal_index = col * k.div_ceil(4) + row / 4;
            let metal_code = (metal[metal_index] >> ((row % 4) * 2)) & 0b11;
            assert_eq!(metal_code, host_code, "row={row}, col={col}");
        }
    }
}

#[test]
fn ternary_repack_rejects_wrong_host_length() {
    let error = ternary_repack_for_metal(&[0], 5, 3).unwrap_err();
    assert!(matches!(error, KernelError::BadBufferLength { .. }));
}

#[test]
fn ternary_unpack_rejects_missing_group_metadata() {
    let values = vec![SignedTernary::Pos, SignedTernary::Neg, SignedTernary::Zero];
    let (packed, _scales, zeros) = ternary_pack(&values, 4).unwrap();
    let mut out = vec![SignedTernary::Zero; values.len()];

    let error = ternary_unpack(&packed, &[], &zeros, values.len(), 4, &mut out).unwrap_err();
    assert!(matches!(
        error,
        KernelError::BadBufferLength {
            what: "scales",
            expected: 1,
            got: 0
        }
    ));
}

#[test]
fn subbyte_round_trip_bits_2_3_4() {
    for &bits in &[2u8, 3, 4] {
        let n = 8;
        let group_size = 8;
        let values: Vec<f32> = (0..n).map(|i| i as f32 / (n as f32)).collect();
        let (packed, scales, zeros) = subbyte_pack(&values, bits, group_size).unwrap();
        let mut out = vec![0.0f32; n];
        subbyte_unpack(&packed, &scales, &zeros, n, group_size, bits, &mut out).unwrap();
        let slack = 1.0 / (1u32 << bits) as f32;
        for (i, (&v, &r)) in values.iter().zip(out.iter()).enumerate() {
            let tol = slack + 1e-5;
            assert!(
                (v - r).abs() <= tol + 1e-4 * v.abs(),
                "bits={bits} idx={i}: got {r}, expected {v} (slack {slack})"
            );
        }
    }
}

#[test]
fn subbyte_pack_rejects_bits_outside_1_to_8() {
    let values = vec![0.0f32; 4];
    assert!(matches!(
        subbyte_pack(&values, 0, 4).unwrap_err(),
        KernelError::BitsOutOfRange { .. }
    ));
    assert!(matches!(
        subbyte_pack(&values, 9, 4).unwrap_err(),
        KernelError::BitsOutOfRange { .. }
    ));
}

#[test]
fn subbyte_pack_rejects_zero_group_size() {
    let err = subbyte_pack(&[0.0f32; 4], 4, 0).unwrap_err();
    assert!(matches!(err, KernelError::ZeroDimension { .. }));
}

#[test]
fn subbyte_handles_partial_trailing_group() {
    let values = vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5];
    let group_size = 4;
    let bits = 4;
    let (packed, scales, zeros) = subbyte_pack(&values, bits, group_size).unwrap();
    let mut out = vec![0.0f32; values.len()];
    subbyte_unpack(
        &packed,
        &scales,
        &zeros,
        values.len(),
        group_size,
        bits,
        &mut out,
    )
    .unwrap();
    let slack = 1.0 / (1u32 << bits) as f32;
    for (i, (&v, &r)) in values.iter().zip(out.iter()).enumerate() {
        let tol = slack + 1e-5;
        assert!(
            (v - r).abs() <= tol + 1e-4 * v.abs(),
            "idx {i}: got {r}, expected {v}"
        );
    }
}
