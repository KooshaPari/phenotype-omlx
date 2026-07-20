//! `native-abi` — versioned Native ABI v1 contract for the TurboQuant family.
//!
//! This crate is the single source of truth for the cross-language descriptor
//! layout that every backend (C, Zig, Mojo, Nim, Go, ...) compiles against.
//! The Rust descriptors here are mirrored into a generated C header that is
//! checked into `include/abi_v1.h` via `build.rs` and shipped with the crate
//! for downstream polyglot consumers.
//!
//! Public surface:
//!
//! * [`version`]: [`AbiVersion`], [`ABI_VERSION_CURRENT`], [`is_compatible`];
//! * [`status`]: [`Status`] and its `i32` round-trip;
//! * [`descriptor`]: [`EncodeRequest`], [`DecodeRequest`], [`EncodeResult`];
//! * [`dispatch`]: reference [`encode_v1`] / [`decode_v1`] / [`release_v1`];
//! * [`headers`]: [`write_c_header`] and [`HEADER_C_SYMBOLS`];
//! * [`NativeAbiError`]: safe-Rust error wrapper.
//!
//! The crate is pure Rust and depends only on `thiserror`. It compiles
//! `no_std`-friendly *except* for the dispatcher (which uses the global
//! allocator); consumers that need `no_std` may still depend on
//! `descriptor`, `status`, `version`, and `headers` directly.

pub mod descriptor;
pub mod dispatch;
pub mod error;
pub mod headers;
pub mod status;
pub mod version;

// Re-export the most commonly used items at the crate root so callers don't
// have to spell out the module paths.
pub use descriptor::{
    bits_valid, expected_packed_len, group_count, DecodeRequest, EncodeRequest, EncodeResult,
};
pub use dispatch::{decode_v1, encode_v1, release_v1, ReleaseKind};
pub use error::NativeAbiError;
pub use headers::{write_c_header, HEADER_C_SYMBOLS};
pub use status::Status;
pub use version::{is_compatible, AbiVersion, ABI_VERSION_CURRENT};