//! Device fingerprint: a stable, hashable description of the device the
//! pipeline is running on.
//!
//! The fingerprint is the cache-key dimension that prevents a shader built
//! for an Apple-silicon M2 Pro from being silently reused on an Intel iGPU.
//! On macOS we collect real values via `sysctl` + `mach`; on every other
//! platform (Linux CI, Windows) we return a deterministic
//! [`GpuFamily::Software`] fallback so the rest of the crate compiles,
//! links, and tests without an Apple SDK.
//!
//! Stability contract: two calls to [`DeviceFingerprint::compute`] on the
//! same machine must agree on every field except `captured_at_unix_ms`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Coarse classification of the GPU family the fingerprint describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuFamily {
    /// Software-only fallback (CPU emulation). Used on non-Apple platforms
    /// and when Metal is unavailable.
    Software,
    /// Apple-silicon integrated GPU (M1/M2/M3/... class).
    AppleSilicon,
    /// Discrete GPU (e.g. AMD Radeon on a Mac Pro, or external eGPU).
    DiscreteGpu,
    /// Generic integrated GPU (Intel Iris / UHD class).
    IntegratedGpu,
}

impl GpuFamily {
    /// Short lowercase tag for logs.
    pub fn tag(&self) -> &'static str {
        match self {
            GpuFamily::Software => "software",
            GpuFamily::AppleSilicon => "apple_silicon",
            GpuFamily::DiscreteGpu => "discrete",
            GpuFamily::IntegratedGpu => "integrated",
        }
    }
}

/// A captured snapshot of the host device.
///
/// `Eq`, `Hash`, and `Serialize`/`Deserialize` are all derived so the
/// fingerprint can be used directly as part of the [`crate::cache::PipelineCache`]
/// key and persisted to disk via [`crate::cache::PipelineCache::write_through`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceFingerprint {
    /// Human-readable device name (e.g. `"MacBook Pro (M2 Pro, 2023)"`).
    pub device_name: String,
    /// Operating-system identifier (e.g. `"macos"`, `"linux"`, `"windows"`).
    pub os: String,
    /// CPU architecture string (e.g. `"aarch64"`, `"x86_64"`).
    pub arch: String,
    /// SIMD register width in bits (`64`, `128`, `256`, `512`).
    pub simd_bit_width: u32,
    /// Total system memory in bytes. Reported as `0` on unsupported
    /// platforms; treated as informational only by the cache key.
    pub total_memory_bytes: u64,
    /// Classification of the active GPU family.
    pub gpu_family: GpuFamily,
    /// `true` when the fingerprint was assembled from a `sysctl` cache hit
    /// rather than a fresh probe. Always `false` for the software fallback.
    pub sysctl_cached: bool,
    /// Unix epoch millis at which the snapshot was captured. NOT part of
    /// the cache key; two captures of the same device at different times
    /// are considered identical.
    pub captured_at_unix_ms: u64,
}

impl DeviceFingerprint {
    /// Construct the deterministic software fallback. Used on every
    /// non-macOS platform, on macOS when Metal is unavailable, and by
    /// tests that want a reproducible fingerprint regardless of host.
    pub fn compute_software() -> Self {
        Self {
            device_name: "software-fallback".to_string(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            simd_bit_width: 128,
            total_memory_bytes: 8 * 1024 * 1024 * 1024,
            gpu_family: GpuFamily::Software,
            sysctl_cached: false,
            captured_at_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }

    /// Capture the device fingerprint for the current host.
    ///
    /// On macOS this delegates to the sysctl-backed probe; on every other
    /// platform it returns [`DeviceFingerprint::compute_software`] with no
    /// I/O.
    pub fn compute() -> Result<Self, FingerprintError> {
        #[cfg(target_os = "macos")]
        {
            macos::probe()
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(Self::compute_software())
        }
    }

    /// Stable, host-independent hash of the fingerprint. Includes every
    /// field *except* `captured_at_unix_ms` and `sysctl_cached` so that two
    /// captures of the same device at different times produce the same
    /// hash (and therefore the same cache key).
    pub fn fingerprint_hash(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.device_name.hash(&mut h);
        self.os.hash(&mut h);
        self.arch.hash(&mut h);
        self.simd_bit_width.hash(&mut h);
        self.total_memory_bytes.hash(&mut h);
        self.gpu_family.hash(&mut h);
        h.finish()
    }
}

/// Errors produced by [`DeviceFingerprint::compute`]. The only variant
/// today signals that the macOS sysctl probe could not determine a
/// required field; in that case the caller should fall back to
/// [`DeviceFingerprint::compute_software`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("device fingerprint probe failed: {0}")]
pub struct FingerprintError(pub String);

// ---------------------------------------------------------------------------
// macOS-specific sysctl probe. Lives behind a `cfg` so the rest of the
// crate compiles on every platform without an Apple SDK.
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod macos {
    use super::{DeviceFingerprint, FingerprintError, GpuFamily};
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn run_sysctl(name: &str) -> Option<String> {
        let out = Command::new("/usr/sbin/sysctl")
            .arg("-n")
            .arg(name)
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let s = String::from_utf8(out.stdout).ok()?;
        let trimmed = s.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    }

    fn sysctl_u64(name: &str) -> Option<u64> {
        run_sysctl(name).and_then(|s| s.parse::<u64>().ok())
    }

    /// Return the active GPU family inferred from sysctl probes.
    ///
    /// `hw.optional.arm64` indicates an Apple-silicon host. On Intel hosts
    /// we distinguish integrated vs. discrete by `machdep.cpu.brand_string`
    /// (which contains "Apple" on M-series, never on Intel).
    fn classify_gpu(arch: &str, brand: &str) -> GpuFamily {
        if arch == "aarch64" || brand.contains("Apple") {
            GpuFamily::AppleSilicon
        } else {
            // Without more probing we cannot reliably distinguish integrated
            // from discrete on Intel Macs, so default to IntegratedGpu.
            GpuFamily::IntegratedGpu
        }
    }

    pub fn probe() -> Result<DeviceFingerprint, FingerprintError> {
        let is_arm = run_sysctl("hw.optional.arm64").is_some();
        let arch: String = if is_arm {
            "aarch64".to_string()
        } else {
            run_sysctl("hw.machine").unwrap_or_else(|| std::env::consts::ARCH.to_string())
        };
        let brand =
            run_sysctl("machdep.cpu.brand_string").unwrap_or_else(|| "unknown-cpu".to_string());
        let model = run_sysctl("hw.model").unwrap_or_else(|| "unknown-model".to_string());

        // hw.memsize is in bytes; fall back to 0 if the call fails.
        let total_memory_bytes = sysctl_u64("hw.memsize").unwrap_or(0);

        // SIMD width: NEON on aarch64 = 128 bits, SSE2 baseline on x86 = 128.
        let simd_bit_width: u32 = 128;

        let gpu_family = classify_gpu(&arch, &brand);
        let device_name = if model.is_empty() {
            brand.clone()
        } else {
            format!("{} ({})", model, brand)
        };

        // sysctl_cached tracks whether `hw.optional.arm64` was a fresh
        // probe or a cache hit. macOS sysctl itself caches internally; we
        // treat a successful probe as "fresh" and an absent one as "cached
        // fallback". The signal is informational and not part of the
        // cache key.
        let sysctl_cached = !is_arm;

        Ok(DeviceFingerprint {
            device_name,
            os: "macos".to_string(),
            arch,
            simd_bit_width,
            total_memory_bytes,
            gpu_family,
            sysctl_cached,
            captured_at_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn software_fallback_is_self_consistent() {
        let fp = DeviceFingerprint::compute_software();
        assert_eq!(fp.gpu_family, GpuFamily::Software);
        assert!(fp.total_memory_bytes > 0);
        assert_eq!(fp.fingerprint_hash(), fp.fingerprint_hash());
    }

    #[test]
    fn fingerprint_hash_is_stable_for_same_fields() {
        let a = DeviceFingerprint {
            device_name: "x".into(),
            os: "macos".into(),
            arch: "aarch64".into(),
            simd_bit_width: 128,
            total_memory_bytes: 16,
            gpu_family: GpuFamily::AppleSilicon,
            sysctl_cached: false,
            captured_at_unix_ms: 100,
        };
        let b = DeviceFingerprint {
            captured_at_unix_ms: 200, // different time
            sysctl_cached: true,      // different cache state
            ..a.clone()
        };
        assert_eq!(a.fingerprint_hash(), b.fingerprint_hash());
    }

    #[test]
    fn gpu_family_tag_is_lowercase() {
        assert_eq!(GpuFamily::Software.tag(), "software");
        assert_eq!(GpuFamily::AppleSilicon.tag(), "apple_silicon");
        assert_eq!(GpuFamily::DiscreteGpu.tag(), "discrete");
        assert_eq!(GpuFamily::IntegratedGpu.tag(), "integrated");
    }
}
