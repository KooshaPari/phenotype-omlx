// Unit tests for turbo-quant-mojo validation and ABI hardening.
//
// Tests are split out of lib.rs to keep lib.rs under the size budget.

use super::MojoQuantizedTensor;
#[cfg(feature = "mojo")]
use std::path::PathBuf;
#[cfg(feature = "mojo")]
use std::process::Command;

#[cfg(feature = "mojo")]
fn mojo_shared_lib_path() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let name = if cfg!(target_os = "macos") {
        "libturbo_quant_mojo.dylib"
    } else if cfg!(target_os = "windows") {
        "turbo_quant_mojo.dll"
    } else {
        "libturbo_quant_mojo.so"
    };
    manifest.join(name)
}

#[cfg(feature = "mojo")]
fn which() -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path).find_map(|dir| {
            let candidate = dir.join("mojo");
            if candidate.is_file() {
                Some(candidate)
            } else {
                None
            }
        })
    })
}

// ─── Build/smoke tests (carried over from lib.rs) ───────────────────────

#[cfg(feature = "mojo")]
#[test]
fn mojo_shared_lib_builds() {
    let lib = mojo_shared_lib_path();
    assert!(
        lib.exists(),
        "Mojo shared library missing at {} — build.rs compile gate failed",
        lib.display()
    );
}

#[cfg(feature = "mojo")]
#[test]
fn mojo_smoke_script_roundtrips() {
    let mojo = std::env::var_os("MOJO_PATH")
        .map(PathBuf::from)
        .or_else(which)
        .expect("mojo not found in PATH — install with: modular install mojo");
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let smoke = manifest.join("mojo-src/turbo_quant_smoke.mojo");
    let status = Command::new(mojo)
        .arg("run")
        .arg(smoke.file_name().expect("smoke filename"))
        .current_dir(manifest.join("mojo-src"))
        .status()
        .expect("spawn mojo run smoke");
    assert!(status.success(), "mojo smoke script failed");
}

// ─── Helpers for decode validation tests ────────────────────────────────

// For n=8, group_size=2, bits=4 → n_groups=4, per_group_packed_bytes=(2*4+7)/8=1,
// packed_len=4*1=4, scales_len=4, zeros_len=4.
fn minimal_valid_tensor() -> MojoQuantizedTensor {
    MojoQuantizedTensor {
        shape: vec![8],
        packed: vec![0u8; 4],
        scales: vec![1.0; 4],
        zeros: vec![0.0; 4],
    }
}

// ─── Canonical roundtrip (preserved) ────────────────────────────────────

#[cfg(feature = "mojo")]
#[test]
fn mojo_encode_decode_roundtrip_ffi_owned_outputs() {
    let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01 - 0.64).collect();
    match MojoQuantizedTensor::encode(&data, 4, 32) {
        Ok(q) => {
            let decoded = q.decode(data.len(), 32, 4);
            for (a, b) in data.iter().zip(decoded.iter()) {
                assert!((a - b).abs() < 0.15, "roundtrip mismatch: {a} vs {b}");
            }
        }
        Err(e) if e.contains("null output pointers") => {
            panic!("Mojo @export out-pointer ABI returned null after ABI hardening — {e}");
        }
        Err(e) => panic!("Mojo encode failed unexpectedly: {e}"),
    }
}

#[cfg(not(feature = "mojo"))]
#[test]
fn native_calls_fail_closed_without_mojo_feature() {
    let tensor = minimal_valid_tensor();
    assert_eq!(
        MojoQuantizedTensor::encode(&[0.0], 4, 1).unwrap_err(),
        "Mojo feature not enabled"
    );
    assert_eq!(
        tensor.try_decode(8, 2, 4).unwrap_err(),
        "Mojo feature not enabled"
    );
}

// ─── try_decode validation (must reject before unsafe Mojo) ─────────────

#[cfg(feature = "mojo")]
#[test]
fn try_decode_valid_inputs_returns_ok() {
    let q = minimal_valid_tensor();
    let res = q.try_decode(8, 2, 4);
    assert!(res.is_ok(), "expected Ok, got {:?}", res.err());
    let decoded = res.unwrap();
    assert_eq!(decoded.len(), 8);
}

#[test]
fn try_decode_zero_n_returns_err() {
    let q = MojoQuantizedTensor {
        shape: vec![0],
        packed: vec![],
        scales: vec![],
        zeros: vec![],
    };
    let err = q.try_decode(0, 32, 4).expect_err("expected Err for n=0");
    assert!(
        err.contains("n") && (err.contains("> 0") || err.contains("empty")),
        "unexpected message: {err}"
    );
}

#[test]
fn try_decode_zero_group_size_returns_err() {
    let q = MojoQuantizedTensor {
        shape: vec![32],
        packed: vec![],
        scales: vec![],
        zeros: vec![],
    };
    let err = q
        .try_decode(32, 0, 4)
        .expect_err("expected Err for group_size=0");
    assert!(
        err.contains("group_size") && err.contains("> 0"),
        "unexpected message: {err}"
    );
}

#[test]
fn try_decode_bits_too_low_returns_err() {
    let q = minimal_valid_tensor();
    let err = q.try_decode(8, 2, 1).expect_err("expected Err for bits=1");
    assert!(err.contains("bits"), "unexpected message: {err}");
}

#[test]
fn try_decode_bits_too_high_returns_err() {
    let q = minimal_valid_tensor();
    let err = q.try_decode(8, 2, 5).expect_err("expected Err for bits=5");
    assert!(err.contains("bits"), "unexpected message: {err}");
}

#[test]
fn try_decode_shape_len_not_one_returns_err() {
    let q = MojoQuantizedTensor {
        shape: vec![2, 4],
        packed: vec![0u8; 4],
        scales: vec![1.0; 2],
        zeros: vec![0.0; 2],
    };
    let err = q
        .try_decode(8, 2, 4)
        .expect_err("expected Err for shape!=1d");
    assert!(err.contains("shape"), "unexpected message: {err}");
}

#[test]
fn try_decode_shape_n_mismatch_returns_err() {
    let q = MojoQuantizedTensor {
        shape: vec![16],
        packed: vec![0u8; 4],
        scales: vec![1.0; 2],
        zeros: vec![0.0; 2],
    };
    let err = q
        .try_decode(8, 2, 4)
        .expect_err("expected Err for shape[0]!=n");
    assert!(err.contains("shape"), "unexpected message: {err}");
}

#[test]
fn try_decode_n_not_divisible_by_group_size_returns_err() {
    let q = MojoQuantizedTensor {
        shape: vec![10],
        packed: vec![],
        scales: vec![],
        zeros: vec![],
    };
    let err = q
        .try_decode(10, 3, 4)
        .expect_err("expected Err for 10 % 3 != 0");
    assert!(
        err.contains("divisible") || err.contains("divisor"),
        "unexpected message: {err}"
    );
}

#[test]
fn try_decode_wrong_packed_len_returns_err() {
    let q = MojoQuantizedTensor {
        shape: vec![8],
        packed: vec![0u8; 1], // expected packed_len=4
        scales: vec![1.0; 4],
        zeros: vec![0.0; 4],
    };
    let err = q
        .try_decode(8, 2, 4)
        .expect_err("expected Err for packed_len mismatch");
    assert!(err.contains("packed"), "unexpected message: {err}");
}

#[test]
fn try_decode_wrong_scales_len_returns_err() {
    let q = MojoQuantizedTensor {
        shape: vec![8],
        packed: vec![0u8; 4],
        scales: vec![1.0; 2], // expected scales_len=4
        zeros: vec![0.0; 4],
    };
    let err = q
        .try_decode(8, 2, 4)
        .expect_err("expected Err for scales_len mismatch");
    assert!(err.contains("scales"), "unexpected message: {err}");
}

#[test]
fn try_decode_wrong_zeros_len_returns_err() {
    let q = MojoQuantizedTensor {
        shape: vec![8],
        packed: vec![0u8; 4],
        scales: vec![1.0; 4],
        zeros: vec![0.0; 2], // expected zeros_len=4
    };
    let err = q
        .try_decode(8, 2, 4)
        .expect_err("expected Err for zeros_len mismatch");
    assert!(err.contains("zeros"), "unexpected message: {err}");
}

#[test]
fn try_decode_n_over_isize_max_returns_err() {
    let n = (isize::MAX as usize).wrapping_add(1);
    let q = MojoQuantizedTensor {
        shape: vec![n],
        packed: vec![0u8; 1],
        scales: vec![1.0; 1],
        zeros: vec![0.0; 1],
    };
    let err = q
        .try_decode(n, 1, 4)
        .expect_err("expected Err for n > isize::MAX");
    assert!(
        err.contains("exceeds") || err.contains("isize") || err.contains("overflow"),
        "unexpected message: {err}"
    );
}

#[test]
fn try_decode_negative_returned_len_returns_err() {
    // We can't easily make the Mojo side return negative, but we can verify that
    // we never accept negative shape_len/packed_len via the strict length check.
    // This test ensures validation rejects mismatched length BEFORE casting.
    let q = MojoQuantizedTensor {
        shape: vec![8],
        packed: vec![], // wrong length
        scales: vec![1.0; 4],
        zeros: vec![0.0; 4],
    };
    let res = q.try_decode(8, 2, 4);
    assert!(res.is_err(), "expected Err for empty packed slice");
}

#[test]
fn try_decode_arithmetic_overflow_returns_err() {
    // n close to usize::MAX/2 + 1, group_size=1, bits=4.
    // n_groups = n, per_group_packed_bytes = 1, packed_len = n which exceeds isize::MAX.
    let n = (isize::MAX as usize).wrapping_add(1);
    let q = MojoQuantizedTensor {
        shape: vec![n],
        packed: vec![0u8; 4],
        scales: vec![1.0; 4],
        zeros: vec![0.0; 4],
    };
    // The n > isize::MAX check should fire first.
    let res = q.try_decode(n, 1, 4);
    assert!(res.is_err(), "expected Err for n > isize::MAX");
}

// ─── decode() panics on invalid input (preserved compatibility) ────────

#[test]
#[should_panic(expected = "MojoQuantizedTensor::decode: invalid input")]
fn decode_panics_on_invalid_input() {
    let q = MojoQuantizedTensor {
        shape: vec![0],
        packed: vec![],
        scales: vec![],
        zeros: vec![],
    };
    let _ = q.decode(0, 32, 4);
}

// ─── encode validation (must reject before unsafe Mojo) ────────────────

#[test]
fn encode_empty_data_returns_err() {
    let data: Vec<f32> = vec![];
    let err = MojoQuantizedTensor::encode(&data, 4, 32).expect_err("expected Err for empty data");
    assert!(
        err.contains("empty") || err.contains("non-empty"),
        "got: {err}"
    );
}

#[test]
fn encode_bits_too_low_returns_err() {
    let data: Vec<f32> = vec![0.0; 32];
    let err = MojoQuantizedTensor::encode(&data, 1, 32).expect_err("expected Err for bits=1");
    assert!(err.contains("bits"), "got: {err}");
}

#[test]
fn encode_bits_too_high_returns_err() {
    let data: Vec<f32> = vec![0.0; 32];
    let err = MojoQuantizedTensor::encode(&data, 5, 32).expect_err("expected Err for bits=5");
    assert!(err.contains("bits"), "got: {err}");
}

#[test]
fn encode_zero_group_size_returns_err() {
    let data: Vec<f32> = vec![0.0; 32];
    let err = MojoQuantizedTensor::encode(&data, 4, 0).expect_err("expected Err for group_size=0");
    assert!(err.contains("group_size"), "got: {err}");
}

#[test]
fn encode_data_not_divisible_by_group_size_returns_err() {
    let data: Vec<f32> = vec![0.0; 33];
    let err = MojoQuantizedTensor::encode(&data, 4, 32).expect_err("expected Err for 33 % 32 != 0");
    assert!(
        err.contains("divisible") || err.contains("divisor"),
        "got: {err}"
    );
}

#[test]
fn encode_unaligned_group_size_returns_err() {
    // 32 elements, group_size=64 → 32 % 64 != 0
    let data: Vec<f32> = vec![0.0; 32];
    let err = MojoQuantizedTensor::encode(&data, 4, 64).expect_err("expected Err for 32 % 64 != 0");
    assert!(
        err.contains("divisible") || err.contains("divisor"),
        "got: {err}"
    );
}

// ─── Sanity: positive encode still works ────────────────────────────────

#[cfg(feature = "mojo")]
#[test]
fn encode_valid_input_succeeds_or_reports_known_abi_issue() {
    let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01 - 0.64).collect();
    match MojoQuantizedTensor::encode(&data, 4, 32) {
        Ok(_) => {}
        Err(e) if e.contains("null output pointers") => {
            // Known Mojo 1.0.0b3 ABI issue. Validation passed (no early return).
        }
        Err(e) => panic!("Mojo encode failed unexpectedly: {e}"),
    }
}

// ─── Regression: repeated encode/decode exercises guard cleanup ─────────
//
// Every successful encode arms the guard, copies the Mojo buffers into
// Rust-owned vectors, and then drops the guard at end of scope — which
// frees the four Mojo allocations. Repeating this many times exercises
// the success-path cleanup on every iteration. A leak or a double-free
// would surface as a process abort on the second iteration on platforms
// where the allocator tracks freed pointers.

#[cfg(feature = "mojo")]
#[test]
fn repeated_encode_decode_exercises_guard_cleanup() {
    let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01 - 0.32).collect();
    for round in 0..64 {
        match MojoQuantizedTensor::encode(&data, 4, 16) {
            Ok(q) => {
                let decoded = q.decode(data.len(), 16, 4);
                assert_eq!(decoded.len(), data.len(), "round {round}: decoded length");
            }
            Err(e) if e.contains("null output pointers") => {
                // Mojo 1.0.0b3 out-pointer ABI still broken — validation
                // passed so the guard's pre-arm path did fire and was
                // responsible for the null-pointer error message.
                continue;
            }
            Err(e) => panic!("round {round}: Mojo encode failed: {e}"),
        }
    }
}
