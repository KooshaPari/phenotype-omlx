//! ABI version identification and compatibility check.
//!
//! The v1 ABI uses a `{major, minor}` pair where the **major** version is the
//! compatibility boundary. A backend compiled against v1 cannot service a host
//! requesting v2; it can service v1.x for any x. Backwards-compatible
//! additions (new fields with sentinel defaults) bump the minor version.

/// Identifies the ABI revision a request or backend was compiled against.
///
/// `PartialEq` / `Eq` are derived so callers can compare the constant
/// `ABI_VERSION_CURRENT` to other descriptors directly. `#[repr(C)]` makes
/// the layout stable across Rust versions and FFI-safe so backends declared
/// `extern "C"` can take a `*const EncodeRequest` without `improper_ctypes`
/// warnings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct AbiVersion {
    pub major: u16,
    pub minor: u16,
}

/// The ABI revision this crate compiles against.
///
/// Backends and frontends compare this against their local expectations and
/// must reject the call (with [`Status::ErrVersionMismatch`]) when the major
/// versions differ.
pub const ABI_VERSION_CURRENT: AbiVersion = AbiVersion { major: 1, minor: 0 };

/// Returns true iff the host and guest can exchange ABI v1 descriptors.
///
/// Compatibility is purely a function of the major version. The minor version
/// is informational; additive changes within a major release must remain
/// source- and binary-compatible at the descriptor boundary.
#[inline]
pub fn is_compatible(host: AbiVersion, guest: AbiVersion) -> bool {
    host.major == guest.major
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_is_major_only() {
        let a = AbiVersion { major: 1, minor: 0 };
        let b = AbiVersion { major: 1, minor: 7 };
        let c = AbiVersion { major: 2, minor: 0 };
        assert!(is_compatible(a, b));
        assert!(is_compatible(b, a));
        assert!(!is_compatible(a, c));
        assert!(!is_compatible(c, a));
    }
}
