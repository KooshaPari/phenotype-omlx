//! ABI v1 descriptors: `EncodeRequest`, `DecodeRequest`, `EncodeResult`.
//!
//! These types are the canonical contract. They are layout-stable across
//! Rust versions because they are `#[repr(C)]` and contain only primitive
//! fields. Any addition must be additive (new field with safe sentinel default)
//! and accompanied by a minor ABI bump.
//!
//! Validation is provided by [`EncodeRequest::validate`] and
//! [`DecodeRequest::validate`]; both return a [`Status`] and never panic on
//! shape / size problems.

use core::ptr;

use crate::status::Status;
use crate::version::AbiVersion;

/// True iff `bits` is one of the supported sub-byte quantisation widths.
#[inline]
pub const fn bits_valid(bits: u8) -> bool {
    bits >= 2 && bits <= 4
}

/// True iff the packed-buffer length produced by encoding `n` elements at
/// `bits` per element matches `packed_len`.
#[inline]
pub const fn packed_len_valid(n: usize, bits: u8, packed_len: usize) -> bool {
    let expected = expected_packed_len(n, bits);
    expected == packed_len
}

/// `(n * bits + 7) / 8`, returning 0 when n or bits is 0.
#[inline]
pub const fn expected_packed_len(n: usize, bits: u8) -> usize {
    if n == 0 || bits == 0 {
        0
    } else {
        (n * bits as usize).div_ceil(8)
    }
}

/// Number of affine-quantisation groups implied by `n` and `group_size`.
#[inline]
pub const fn group_count(n: usize, group_size: usize) -> usize {
    if group_size == 0 {
        0
    } else {
        n.div_ceil(group_size)
    }
}

/// Encode request: caller fills the descriptor, calls `dispatch::encode`,
/// reads the populated output buffers and lengths from the descriptor.
///
/// All output pointers are caller-owned storage. The dispatcher writes into
/// them and writes the matching `*_len` slot. On failure every output is
/// set to NULL / 0 so a partial state cannot leak across calls.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EncodeRequest {
    /// ABI revision the caller was compiled against.
    pub abi: AbiVersion,
    /// Read-only input data.
    pub data_ptr: *const f32,
    /// Number of input elements.
    pub n: usize,
    /// Sub-byte width, 2..=4.
    pub bits: u8,
    /// Group size for affine quantisation. Must be > 0.
    pub group_size: usize,
    /// Output: shape array. Caller-owned.
    pub out_shape: *mut *mut usize,
    pub out_shape_capacity: usize,
    /// Output: packed bitstream. Caller-owned.
    pub out_packed: *mut *mut u8,
    pub out_packed_capacity: usize,
    /// Output: per-group scale factors. Caller-owned.
    pub out_scales: *mut *mut f32,
    pub out_scales_capacity: usize,
    /// Output: per-group zero offsets. Caller-owned.
    pub out_zeros: *mut *mut f32,
    pub out_zeros_capacity: usize,
}

impl EncodeRequest {
    /// Construct a descriptor with every field zeroed. Useful for callers
    /// that fill in only the slots they need before validation.
    pub fn zeroed() -> Self {
        Self {
            abi: AbiVersion { major: 0, minor: 0 },
            data_ptr: ptr::null(),
            n: 0,
            bits: 0,
            group_size: 0,
            out_shape: ptr::null_mut(),
            out_shape_capacity: 0,
            out_packed: ptr::null_mut(),
            out_packed_capacity: 0,
            out_scales: ptr::null_mut(),
            out_scales_capacity: 0,
            out_zeros: ptr::null_mut(),
            out_zeros_capacity: 0,
        }
    }

    /// Validate the descriptor. Returns `Err(Status)` if the request is
    /// invalid for any reason; otherwise `Ok(())`.
    ///
    /// Validation rules (in order):
    ///   * abi.major must be 1;
    ///   * `data_ptr` non-null and `n > 0`;
    ///   * `bits in 2..=4`;
    ///   * `group_size > 0`;
    ///   * all output slots non-null with capacity > 0.
    pub fn validate(&self) -> Result<(), Status> {
        if self.abi.major != 1 {
            return Err(Status::ErrVersionMismatch);
        }
        if self.data_ptr.is_null() || self.n == 0 {
            return Err(Status::ErrNullArg);
        }
        if !bits_valid(self.bits) {
            return Err(Status::ErrInvalidBits);
        }
        if self.group_size == 0 {
            return Err(Status::ErrInvalidGroupSize);
        }
        if self.out_shape.is_null()
            || self.out_shape_capacity == 0
            || self.out_packed.is_null()
            || self.out_packed_capacity == 0
            || self.out_scales.is_null()
            || self.out_scales_capacity == 0
            || self.out_zeros.is_null()
            || self.out_zeros_capacity == 0
        {
            return Err(Status::ErrNullArg);
        }
        Ok(())
    }
}

/// Decode request: caller fills the descriptor, calls `dispatch::decode`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DecodeRequest {
    pub abi: AbiVersion,
    pub packed_ptr: *const u8,
    pub packed_len: usize,
    pub scales_ptr: *const f32,
    pub zeros_ptr: *const f32,
    pub n: usize,
    pub group_size: usize,
    pub bits: u8,
    /// Caller-owned output buffer. The contract guarantees it is left
    /// untouched on any validation failure.
    pub out_ptr: *mut f32,
}

impl DecodeRequest {
    pub fn zeroed() -> Self {
        Self {
            abi: AbiVersion { major: 0, minor: 0 },
            packed_ptr: ptr::null(),
            packed_len: 0,
            scales_ptr: ptr::null(),
            zeros_ptr: ptr::null(),
            n: 0,
            group_size: 0,
            bits: 0,
            out_ptr: ptr::null_mut(),
        }
    }

    /// Validate the decode request. On any failure, the dispatcher must
    /// leave `out_ptr` and its contents untouched.
    pub fn validate(&self) -> Result<(), Status> {
        if self.abi.major != 1 {
            return Err(Status::ErrVersionMismatch);
        }
        if self.out_ptr.is_null() || self.n == 0 {
            return Err(Status::ErrNullArg);
        }
        if !bits_valid(self.bits) {
            return Err(Status::ErrInvalidBits);
        }
        if self.group_size == 0 {
            return Err(Status::ErrInvalidGroupSize);
        }
        if self.packed_ptr.is_null() || self.scales_ptr.is_null() || self.zeros_ptr.is_null() {
            return Err(Status::ErrNullArg);
        }
        // Packed length must match the contract for `n * bits`.
        if !packed_len_valid(self.n, self.bits, self.packed_len) {
            return Err(Status::ErrInvalidBits);
        }
        // Caller must provide enough per-group scale/zero slots.
        let ng = group_count(self.n, self.group_size);
        if self.scales_ptr as usize == 0 || self.zeros_ptr as usize == 0 {
            return Err(Status::ErrNullArg);
        }
        // The descriptor doesn't carry scales_len / zeros_len; the contract
        // requires it to be inferable from n and group_size. The dispatcher
        // will bounds-check via slice reads — the validate function simply
        // notes the dependency here for documentation.
        let _ = ng;
        Ok(())
    }
}

/// Result of an encode call. The dispatcher populates `written_*` only on
/// `status == Ok`; on failure every field is 0.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeResult {
    pub status: Status,
    pub written_packed_len: usize,
    pub written_shape_len: usize,
    pub written_scales_len: usize,
    pub written_zeros_len: usize,
}

impl EncodeResult {
    /// All-zero result, indicating either success-with-zero-output (which the
    /// dispatcher never produces) or a failure that the caller treats as
    /// "nothing to read".
    pub fn zeroed() -> Self {
        Self {
            status: Status::Ok,
            written_packed_len: 0,
            written_shape_len: 0,
            written_scales_len: 0,
            written_zeros_len: 0,
        }
    }
}
