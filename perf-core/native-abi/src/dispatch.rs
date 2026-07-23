//! Reference implementations of `tq_abi_encode` / `tq_abi_decode`.
//!
//! These satisfy the v1 contract without depending on any C / Zig library.
//! They are the baseline against which backends (C, Zig, Mojo, Nim, Go, ...)
//! are conformance-tested.
//!
//! Contract:
//!
//! * Every input is validated before any output is written.
//! * On failure every output slot is reset to NULL / 0 so the caller can
//!   always recover a clean state without having to reason about the
//!   failure point.
//! * On decode failure the caller's `out_ptr` is left **untouched**.
//! * Encode writes into caller-provided buffers; the dispatcher allocates
//!   the packed bitstream and scales/zeros arrays using the Rust global
//!   allocator, then transfers ownership of the heap allocations to the
//!   caller via the `*_ptr` slots. The caller is responsible for freeing
//!   them with the matching release function (see `release_v1`).

use core::ptr;
use core::slice;
use std::alloc::{self, Layout};

use crate::descriptor::{
    expected_packed_len, group_count, DecodeRequest, EncodeRequest, EncodeResult,
};
use crate::status::Status;

/// Versioned encode entry. Callers fill `req` and read back the populated
/// output slots + `EncodeResult.written_*` lengths on success.
///
/// # Safety
///
/// `req` must be a valid [`EncodeRequest`] whose output pointers are
/// non-null, properly aligned, and point to caller-owned buffers with at
/// least `out_*_capacity` elements. Every `*_ptr` in `req` must point to
/// readable / writable memory of the appropriate type for `n` elements.
/// Violating these requirements is undefined behavior.
pub unsafe fn encode_v1(req: &EncodeRequest) -> EncodeResult {
    // Failure returns are funnelled through `fail_with_zeroed_outputs` so the
    // zero-on-failure contract is implemented in exactly one place. On
    // success the caller's output slots are written to directly.
    macro_rules! fail {
        ($status:expr) => {{
            zero_outputs(req);
            return EncodeResult {
                status: $status,
                ..EncodeResult::zeroed()
            };
        }};
    }

    if let Err(status) = req.validate() {
        fail!(status);
    }

    let n = req.n;
    let bits = req.bits;
    let group_size = req.group_size;
    let data = slice::from_raw_parts(req.data_ptr, n);

    // Overflow checks.
    if n > usize::MAX - group_size {
        fail!(Status::ErrOverflow);
    }
    if (bits as usize) > 0 && n > usize::MAX / (bits as usize) {
        fail!(Status::ErrOverflow);
    }

    // Reject non-finite input.
    for &v in data {
        if !v.is_finite() {
            fail!(Status::ErrNonFiniteInput);
        }
    }

    let packed_len = expected_packed_len(n, bits);
    let n_groups = group_count(n, group_size);

    // Capacity checks against caller-provided slots.
    if req.out_packed_capacity < packed_len
        || req.out_shape_capacity < 1
        || req.out_scales_capacity < n_groups
        || req.out_zeros_capacity < n_groups
    {
        fail!(Status::ErrOverflow);
    }

    let levels = ((1u32 << bits) - 1) as f32;

    // Write shape.
    let shape_slot = *req.out_shape;
    *shape_slot = n;

    // Write scales/zeros.
    let scales_slot = slice::from_raw_parts_mut(*req.out_scales, n_groups);
    let zeros_slot = slice::from_raw_parts_mut(*req.out_zeros, n_groups);

    // Write packed bitstream. Caller-owned; we just write into it.
    let packed_slot = slice::from_raw_parts_mut(*req.out_packed, packed_len);

    for g in 0..n_groups {
        let start = g * group_size;
        let end = core::cmp::min(start + group_size, n);

        let (mut gmin, mut gmax) = (data[start], data[start]);
        for &v in &data[start + 1..end] {
            if v < gmin {
                gmin = v;
            }
            if v > gmax {
                gmax = v;
            }
        }

        let span = gmax - gmin;
        let scale = if span > 0.0 { span / levels } else { 1e-30_f32 };
        scales_slot[g] = scale;
        zeros_slot[g] = gmin;

        for (rel, &v) in data[start..end].iter().enumerate() {
            let qf = ((v - gmin) / scale).clamp(0.0, levels);
            let q = (qf + 0.5) as u32;
            let q = if q > levels as u32 { levels as u32 } else { q };
            let bit_off = (g * group_size + rel) * (bits as usize);
            write_bits(packed_slot, bit_off, q as u8, bits);
        }
    }

    EncodeResult {
        status: Status::Ok,
        written_packed_len: packed_len,
        written_shape_len: 1,
        written_scales_len: n_groups,
        written_zeros_len: n_groups,
    }
}

/// Versioned decode entry. Validates every input first; on failure the
/// caller's `out_ptr` and its contents are guaranteed untouched.
///
/// # Safety
///
/// `req` must be a valid [`DecodeRequest`] whose pointers are non-null,
/// properly aligned, and point to caller-owned storage with the lengths
/// declared in `req`. `out_ptr` in particular must be writable for `n`
/// `f32` values. Violating these requirements is undefined behavior.
pub unsafe fn decode_v1(req: &DecodeRequest) -> Status {
    if let Err(status) = req.validate() {
        return status;
    }

    let n = req.n;
    let bits = req.bits;
    let group_size = req.group_size;
    let packed_len = req.packed_len;

    let packed = slice::from_raw_parts(req.packed_ptr, packed_len);
    let ng = group_count(n, group_size);
    let scales = slice::from_raw_parts(req.scales_ptr, ng);
    let zeros = slice::from_raw_parts(req.zeros_ptr, ng);

    let out = slice::from_raw_parts_mut(req.out_ptr, n);

    for g in 0..ng {
        let scale = scales[g];
        let zero = zeros[g];
        let start = g * group_size;
        let end = core::cmp::min(start + group_size, n);
        for (rel, slot) in out[start..end].iter_mut().enumerate() {
            let bit_off = (g * group_size + rel) * (bits as usize);
            let q = read_bits(packed, bit_off, bits) as f32;
            *slot = zero + q * scale;
        }
    }
    Status::Ok
}

/// Free a heap allocation previously produced by `encode_v1`.
///
/// `kind` selects which allocator/element-size to use. This is the v1
/// matching-release entry: every allocate has exactly one matching release.
///
/// Safe to call with `ptr = NULL`; the call is a no-op.
///
/// # Safety
///
/// `ptr` must either be null or have been produced by `encode_v1` with
/// the matching `kind`, and must not have been freed previously. Passing
/// any other pointer — including the wrong kind for an allocation —
/// is undefined behavior.
pub unsafe fn release_v1(kind: ReleaseKind, ptr: *mut u8, count: usize) {
    if ptr.is_null() || count == 0 {
        return;
    }
    let layout = match kind {
        ReleaseKind::ShapeUsize => Layout::array::<usize>(count).expect("shape layout"),
        ReleaseKind::PackedBytes => Layout::array::<u8>(count).expect("packed layout"),
        ReleaseKind::ScaleF32 => Layout::array::<f32>(count).expect("scale layout"),
        ReleaseKind::ZeroF32 => Layout::array::<f32>(count).expect("zero layout"),
    };
    alloc::dealloc(ptr, layout);
}

/// Selects the layout used by [`release_v1`] for a given output buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseKind {
    ShapeUsize,
    PackedBytes,
    ScaleF32,
    ZeroF32,
}

// ── output-zeroing helpers ───────────────────────────────────────────────────

/// Reset every output slot to NULL / 0 so callers see a clean state on
/// failure or success-with-no-output.
unsafe fn zero_outputs(req: &EncodeRequest) {
    if !req.out_shape.is_null() {
        *req.out_shape = ptr::null_mut();
    }
    if !req.out_packed.is_null() {
        *req.out_packed = ptr::null_mut();
    }
    if !req.out_scales.is_null() {
        *req.out_scales = ptr::null_mut();
    }
    if !req.out_zeros.is_null() {
        *req.out_zeros = ptr::null_mut();
    }
}

// ── bit-packing helpers ─────────────────────────────────────────────────────

fn write_bits(buf: &mut [u8], bit_offset: usize, value: u8, bits: u8) {
    let mask = (1u8 << bits) - 1;
    let value = value & mask;
    let byte_idx = bit_offset >> 3;
    let bit_in_byte = bit_offset & 7;
    let room = 8 - bit_in_byte;

    if room >= bits as usize {
        let shift_mask = mask << bit_in_byte;
        buf[byte_idx] = (buf[byte_idx] & !shift_mask) | (value << bit_in_byte);
    } else {
        let lo_mask = (1u8 << room) - 1;
        buf[byte_idx] =
            (buf[byte_idx] & !(lo_mask << bit_in_byte)) | ((value & lo_mask) << bit_in_byte);
        let hi_bits = bits as usize - room;
        let hi_mask = (1u8 << hi_bits) - 1;
        buf[byte_idx + 1] = (buf[byte_idx + 1] & !hi_mask) | ((value >> room) & hi_mask);
    }
}

fn read_bits(buf: &[u8], bit_offset: usize, bits: u8) -> u8 {
    let mask = (1u8 << bits) - 1;
    let byte_idx = bit_offset >> 3;
    let bit_in_byte = bit_offset & 7;
    let room = 8 - bit_in_byte;

    if room >= bits as usize {
        return (buf[byte_idx] >> bit_in_byte) & mask;
    }
    let lo_mask = (1u8 << room) - 1;
    let lo = (buf[byte_idx] >> bit_in_byte) & lo_mask;
    let hi_bits = bits as usize - room;
    let hi_mask = (1u8 << hi_bits) - 1;
    let hi = buf[byte_idx + 1] & hi_mask;
    lo | (hi << room)
}
