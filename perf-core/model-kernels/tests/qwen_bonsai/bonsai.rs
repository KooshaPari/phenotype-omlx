//! Bonsai fused ternary matmul parity test.

use super::*;

// ===========================================================================
// Bonsai ternary matmul parity
// ===========================================================================

#[test]
fn bonsai_ternary_matmul_matches_unpacked_reference() {
    // Activation: a is [4, 16] row-major.
    // Weight: b is [16, 32] row-major ternary. The kernel expects
    // b_packed to be the row-major flat of [16, 32] in the same
    // byte order as `ternary_pack` would produce.
    let m = 4;
    let k = 16;
    let n = 32;
    let group_size = k * n; // single Bonsai group

    // Build a deterministic ternary weight using the Lcg salt for
    // reproducibility.
    let mut rng = Lcg::new(SEED ^ 0xA11CE);
    let values: Vec<SignedTernary> = (0..k * n)
        .map(|_| match (rng.next_u64() % 3) as u8 {
            0 => SignedTernary::Pos,
            1 => SignedTernary::Neg,
            _ => SignedTernary::Zero,
        })
        .collect();
    let (packed, scales, zeros) = ternary_pack(&values, group_size).unwrap();

    let a = deterministic_vec(m * k, 0xBEEF);

    let mut out = vec![0.0f32; m * n];
    ternary_matmul(&a, &packed, &scales, &zeros, group_size, m, k, n, &mut out).unwrap();

    // Reference: unpack and run a dense per-row inner-product.
    let mut unpacked = vec![SignedTernary::Zero; values.len()];
    ternary_unpack(&packed, &scales, &zeros, values.len(), group_size, &mut unpacked).unwrap();

    let mut expected = vec![0.0f32; m * n];
    for row in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let w = match unpacked[kk * n + j] {
                    SignedTernary::Pos => 1.0,
                    SignedTernary::Neg => -1.0,
                    SignedTernary::Zero => 0.0,
                };
                acc += a[row * k + kk] * w;
            }
            expected[row * n + j] = acc;
        }
    }
    assert_buf_close(&out, &expected, 1e-5, 1e-4);
    // Bonus: every entry must be finite.
    assert!(out.iter().all(|v| v.is_finite()), "out has a non-finite entry");
}
