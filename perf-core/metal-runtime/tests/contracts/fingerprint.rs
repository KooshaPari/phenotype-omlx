//! §1 — Device fingerprinting contracts.
//!
//! Covers stability, distinctness across fake GPU families, hash self-
//! equivalence, deterministic software fallback, and serde round-trip.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use metal_runtime::{DeviceFingerprint, GpuFamily};

use super::common::{identity_fp, tnow_ms};

#[test]
fn fingerprint_is_stable_across_calls_on_same_machine() {
    let a = DeviceFingerprint::compute().expect("compute on host platform");
    let b = DeviceFingerprint::compute().expect("compute on host platform");
    // device_name, os, arch, gpu_family, sysctl_cached must be stable. The
    // captured_at_unix_ms is explicitly NOT compared.
    assert_eq!(a.device_name, b.device_name);
    assert_eq!(a.os, b.os);
    assert_eq!(a.arch, b.arch);
    assert_eq!(a.gpu_family, b.gpu_family);
    assert_eq!(a.simd_bit_width, b.simd_bit_width);
    assert_eq!(a.total_memory_bytes, b.total_memory_bytes);
}

#[test]
fn fingerprint_distinct_across_fake_gpu_families() {
    let mut set = HashSet::new();
    for fam in [
        GpuFamily::Software,
        GpuFamily::AppleSilicon,
        GpuFamily::DiscreteGpu,
        GpuFamily::IntegratedGpu,
    ] {
        let fp = identity_fp(fam);
        // hash() must be different across families (device_name identical).
        set.insert(fp.fingerprint_hash());
    }
    assert_eq!(set.len(), 4, "all four GpuFamily variants must hash distinctly");
}

#[test]
fn fingerprint_hash_matches_itself() {
    let fp = identity_fp(GpuFamily::AppleSilicon);
    assert_eq!(fp.fingerprint_hash(), fp.fingerprint_hash());
    // Hashing the same fingerprint twice via the Hash trait must produce
    // the same u64 — exercise this directly without BuildHasher (which is
    // not implemented for DefaultHasher itself).
    let mut h1 = DefaultHasher::new();
    fp.hash(&mut h1);
    let digest1 = h1.finish();
    let mut h2 = DefaultHasher::new();
    fp.hash(&mut h2);
    let digest2 = h2.finish();
    assert_eq!(digest1, digest2);
}

#[test]
fn fingerprint_software_fallback_on_non_macos_is_deterministic() {
    let a = DeviceFingerprint::compute_software();
    let b = DeviceFingerprint::compute_software();
    assert_eq!(a, b);
    assert_eq!(a.gpu_family, GpuFamily::Software);
    assert_eq!(a.device_name, "software-fallback");
    assert!(a.total_memory_bytes > 0);
    assert!(a.simd_bit_width >= 64);
    assert!(a.captured_at_unix_ms <= tnow_ms());
}

#[test]
fn fingerprint_serializes_and_round_trips() {
    let fp = DeviceFingerprint {
        device_name: "M2 Pro".to_string(),
        os: "macos".to_string(),
        arch: "aarch64".to_string(),
        simd_bit_width: 128,
        total_memory_bytes: 16 * 1024 * 1024 * 1024,
        gpu_family: GpuFamily::AppleSilicon,
        sysctl_cached: true,
        captured_at_unix_ms: 1_700_000_000_000,
    };
    let s = serde_json::to_string(&fp).expect("serialize");
    let back: DeviceFingerprint = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(back, fp);
}
