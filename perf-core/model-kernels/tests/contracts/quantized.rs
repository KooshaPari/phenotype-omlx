//! Section "Quantized" of the original contracts.rs.
//!
//! Split out of the original monolithic `model-kernels/tests/contracts.rs`
//! (1130 lines) so each topic stays under the 350-line target. Test bodies
//! are byte-identical to the source file; only the surrounding module
//! wrapper and `use super::*;` import differ.

use super::*;

#[test]
fn ternary_pack_matches_manual_packing() {
    // 8 values packed into 2 bytes (2 bits each), single group of size 8.
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
    let group_size = 8;
    let (packed, scales, zeros) = ternary_pack(&values, group_size).unwrap();
    assert_eq!(packed.len(), 2);
    assert_eq!(scales.len(), 1);
    assert_eq!(zeros.len(), 1);
    // First byte: lower 2 bits = Zero (00), then Pos (01), Neg (10), Pos (01)
    // -> 0b 01 10 01 00 = 0x64
    assert_eq!(packed[0], 0b01_10_01_00);
    // Second byte: Zero (00), Neg (10), Neg (10), Pos (01) -> 0b 01 10 10 00 = 0x68
    assert_eq!(packed[1], 0b01_10_10_00);
}

#[test]
fn ternary_unpack_inverts_pack() {
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
    let group_size = 8;
    let (packed, scales, zeros) = ternary_pack(&values, group_size).unwrap();
    let mut out = vec![SignedTernary::Zero; values.len()];
    ternary_unpack(&packed, &scales, &zeros, values.len(), group_size, &mut out).unwrap();
    assert_eq!(out, values);
}

#[test]
fn subbyte_pack_bits_2_3_4_roundtrip() {
    for &bits in &[2u8, 3, 4] {
        let n = 8;
        let group_size = 8;
        let values: Vec<f32> = (0..n).map(|i| i as f32 / (n as f32)).collect();
        let (packed, scales, zeros) = subbyte_pack(&values, bits, group_size).unwrap();
        let mut out = vec![0.0f32; n];
        subbyte_unpack(&packed, &scales, &zeros, n, group_size, bits, &mut out).unwrap();
        for (i, (&v, &r)) in values.iter().zip(out.iter()).enumerate() {
            // Allow ±1/2^bits relative slack for quantization.
            let slack = 1.0 / (1u32 << bits) as f32;
            let tol = 1e-5 + slack;
            assert!(
                approx_eq(v, r) || (v - r).abs() <= tol + 1e-4 * v.abs(),
                "bits={bits} idx={i}: got {r}, expected {v} (slack {slack})"
            );
        }
    }
}

#[test]
fn subbyte_pack_rejects_bits_outside_1_to_8() {
    let values = vec![0.0f32; 4];
    let err = subbyte_pack(&values, 0, 4).unwrap_err();
    assert!(matches!(err, KernelError::BitsOutOfRange { .. }));
    let err = subbyte_pack(&values, 9, 4).unwrap_err();
    assert!(matches!(err, KernelError::BitsOutOfRange { .. }));
}

#[test]
fn subbyte_pack_handles_partial_trailing_group() {
    // 10 values, group_size=4 -> 3 groups (4, 4, 2).
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
    for (i, (&v, &r)) in values.iter().zip(out.iter()).enumerate() {
        let slack = 1.0 / (1u32 << bits) as f32;
        assert!(
            approx_eq(v, r) || (v - r).abs() <= slack + 1e-4 * v.abs(),
            "idx {i}: got {r}, expected {v}"
        );
    }
}
