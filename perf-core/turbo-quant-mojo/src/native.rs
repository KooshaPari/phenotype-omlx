// Mojo FFI declarations and Rust-side wrappers.
//
// All caller-controlled dimensions crossing the ABI boundary are passed as
// `isize` (matching Mojo's signed `Int`); the public safe API takes `usize`
// and converts with checked arithmetic in `validation`.

use super::MojoQuantizedTensor;
#[cfg(all(feature = "mojo-native", mojo_native))]
use crate::validation::{
    usize_to_isize, validate_decode_inputs, validate_encode_inputs, validate_encode_outputs,
};
#[cfg(not(all(feature = "mojo-native", mojo_native)))]
use crate::validation::{validate_decode_inputs, validate_encode_inputs};
#[cfg(all(feature = "mojo-native", mojo_native))]
use std::os::raw::c_uchar;

#[cfg(all(feature = "mojo-native", mojo_native))]
extern "C" {
    fn tq_mojo_encode(
        data_addr: isize,
        n: isize,
        bits: c_uchar,
        group_size: isize,
        shape_ptr_out: *mut isize,
        out_shape_len: *mut isize,
        packed_ptr_out: *mut isize,
        out_packed_len: *mut isize,
        scales_ptr_out: *mut isize,
        out_scales_len: *mut isize,
        zeros_ptr_out: *mut isize,
        out_zeros_len: *mut isize,
    ) -> bool;

    fn tq_mojo_decode(
        packed_ptr: *const u8,
        packed_len: isize,
        scales_ptr: *const f32,
        zeros_ptr: *const f32,
        n: isize,
        group_size: isize,
        bits: c_uchar,
        out_ptr: *mut f32,
    );

    fn tq_mojo_free(address: isize);
}

#[cfg(all(feature = "mojo-ffi", feature = "mojo-native", mojo_native))]
extern "C" {
    fn tq_gemv_decode(
        weights: *const f32,
        input: *const f32,
        output: *mut f32,
        rows: usize,
        cols: usize,
    );
}

/// RAII guard that releases every strictly-positive Mojo-allocated address
/// exactly once on drop. Built around three safety invariants:
///
/// 1. The guard is armed IMMEDIATELY after `tq_mojo_encode` returns, before
///    the caller inspects `ok`. A `false` return with partial positive
///    output slots is therefore still cleaned up via Drop.
/// 2. Only addresses that are strictly positive are freed. Zero slots are
///    skipped (no allocation to release) and negative addresses are skipped
///    (corrupt-build guard — handing a negative isize to `tq_mojo_free` is
///    undefined behaviour, so we refuse to do so).
/// 3. The successful encode path does NOT disarm or take the guard; the
///    Drop runs at end of scope and frees every Mojo buffer exactly once
///    after the Rust-owned vectors have been constructed.
#[cfg(any(all(feature = "mojo-native", mojo_native), test))]
struct EncodeOutputGuard {
    shape_addr: isize,
    packed_addr: isize,
    scales_addr: isize,
    zeros_addr: isize,
}

#[cfg(any(all(feature = "mojo-native", mojo_native), test))]
impl EncodeOutputGuard {
    fn empty() -> Self {
        Self {
            shape_addr: 0,
            packed_addr: 0,
            scales_addr: 0,
            zeros_addr: 0,
        }
    }

    /// Populate the guard with the four raw addresses returned by
    /// `tq_mojo_encode`. The destructor will free every strictly-positive
    /// address exactly once and skip zero / negative entries.
    fn arm(&mut self, shape: isize, packed: isize, scales: isize, zeros: isize) {
        self.shape_addr = shape;
        self.packed_addr = packed;
        self.scales_addr = scales;
        self.zeros_addr = zeros;
    }

    /// True iff `addr` is a real Mojo allocation that must be released.
    /// Exposed at `pub(crate)` so unit tests can exercise the policy without
    /// invoking the (unsafe, side-effecting) `tq_mojo_free` entry point.
    pub(crate) const fn addr_should_free(addr: isize) -> bool {
        addr > 0
    }
}

#[cfg(any(all(feature = "mojo-native", mojo_native), test))]
impl Drop for EncodeOutputGuard {
    fn drop(&mut self) {
        // No early-return: the guard is built so that arming it is the only
        // way to populate addresses. Every drop releases the live positive
        // addresses exactly once.
        #[cfg(all(feature = "mojo-native", mojo_native))]
        unsafe {
            if Self::addr_should_free(self.shape_addr) {
                tq_mojo_free(self.shape_addr);
            }
            if Self::addr_should_free(self.packed_addr) {
                tq_mojo_free(self.packed_addr);
            }
            if Self::addr_should_free(self.scales_addr) {
                tq_mojo_free(self.scales_addr);
            }
            if Self::addr_should_free(self.zeros_addr) {
                tq_mojo_free(self.zeros_addr);
            }
        }
    }
}

#[cfg(all(feature = "mojo-native", mojo_native))]
pub(super) fn mojo_encode(
    data: &[f32],
    bits: u8,
    group_size: usize,
) -> Result<MojoQuantizedTensor, String> {
    let (n, n_groups, expected_packed_len) = validate_encode_inputs(data.len(), bits, group_size)?;
    let expected_shape_len = 1usize;
    let expected_scales_len = n_groups;
    let expected_zeros_len = n_groups;

    let n_isize = usize_to_isize("encode n", n)?;
    let group_size_isize = usize_to_isize("encode group_size", group_size)?;

    let mut shape_addr: isize = 0;
    let mut shape_len: isize = 0;
    let mut packed_addr: isize = 0;
    let mut packed_len: isize = 0;
    let mut scales_addr: isize = 0;
    let mut scales_len: isize = 0;
    let mut zeros_addr: isize = 0;
    let mut zeros_len: isize = 0;

    let mut guard = EncodeOutputGuard::empty();

    let ok = unsafe {
        tq_mojo_encode(
            data.as_ptr() as isize,
            n_isize,
            bits,
            group_size_isize,
            &mut shape_addr,
            &mut shape_len,
            &mut packed_addr,
            &mut packed_len,
            &mut scales_addr,
            &mut scales_len,
            &mut zeros_addr,
            &mut zeros_len,
        )
    };

    // Arm the guard IMMEDIATELY after the FFI call — before inspecting
    // `ok` — so that a `false` return with partial positive output slots
    // still has every positive allocation freed via Drop. Zero slots are
    // skipped by the guard's policy; nothing has been leaked.
    guard.arm(shape_addr, packed_addr, scales_addr, zeros_addr);

    if !ok {
        return Err("Mojo tq_mojo_encode returned false".to_string());
    }

    // Reject null OR negative addresses BEFORE any unsafe dereference.
    // Canonical Mojo never returns these, but a corrupted build could.
    // The guard's Drop still runs on this error path and frees every
    // strictly-positive sibling output address exactly once.
    if !EncodeOutputGuard::addr_should_free(shape_addr)
        || !EncodeOutputGuard::addr_should_free(packed_addr)
        || !EncodeOutputGuard::addr_should_free(scales_addr)
        || !EncodeOutputGuard::addr_should_free(zeros_addr)
    {
        return Err(format!(
            "encode: Mojo returned non-positive pointer address \
             (shape={shape_addr}, packed={packed_addr}, \
              scales={scales_addr}, zeros={zeros_addr})"
        ));
    }

    let (shape_len_u, packed_len_u, scales_len_u, zeros_len_u) = validate_encode_outputs(
        shape_len,
        packed_len,
        scales_len,
        zeros_len,
        expected_shape_len,
        expected_packed_len,
        expected_scales_len,
        expected_zeros_len,
    )?;

    let shape_ptr = shape_addr as *mut usize;
    let packed_ptr = packed_addr as *mut u8;
    let scales_ptr = scales_addr as *mut f32;
    let zeros_ptr = zeros_addr as *mut f32;

    // From here on, every step is total — no more `?`. A panic inside
    // `from_raw_parts` would still free on drop because `guard` is armed.
    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, shape_len_u) }.to_vec();
    let packed = unsafe { std::slice::from_raw_parts(packed_ptr, packed_len_u) }.to_vec();
    let scales = unsafe { std::slice::from_raw_parts(scales_ptr, scales_len_u) }.to_vec();
    let zeros = unsafe { std::slice::from_raw_parts(zeros_ptr, zeros_len_u) }.to_vec();

    // Guard's Drop runs at end of scope and frees every positive Mojo
    // buffer exactly once — the successful path does NOT disarm. Skipping
    // the free here would leak every Mojo allocation on every successful
    // encode.
    Ok(MojoQuantizedTensor {
        shape,
        packed,
        scales,
        zeros,
    })
}

#[cfg(all(feature = "mojo-native", mojo_native))]
pub(super) fn mojo_try_decode(
    shape: &[usize],
    packed: &[u8],
    scales: &[f32],
    zeros: &[f32],
    n: usize,
    group_size: usize,
    bits: u8,
) -> Result<Vec<f32>, String> {
    let (n_u, n_groups, expected_packed_len) = validate_decode_inputs(
        shape,
        packed.len(),
        scales.len(),
        zeros.len(),
        n,
        group_size,
        bits,
    )?;
    debug_assert_eq!(n_groups, scales.len());
    debug_assert_eq!(n_groups, zeros.len());
    debug_assert_eq!(expected_packed_len, packed.len());

    let n_isize = usize_to_isize("decode n", n_u)?;
    let group_size_isize = usize_to_isize("decode group_size", group_size)?;
    let packed_len_isize = usize_to_isize("decode packed_len", expected_packed_len)?;

    let mut out = vec![0.0f32; n_u];

    unsafe {
        tq_mojo_decode(
            packed.as_ptr(),
            packed_len_isize,
            scales.as_ptr(),
            zeros.as_ptr(),
            n_isize,
            group_size_isize,
            bits,
            out.as_mut_ptr(),
        );
    }

    Ok(out)
}

#[cfg(not(all(feature = "mojo-native", mojo_native)))]
pub(super) fn mojo_encode(
    data: &[f32],
    bits: u8,
    group_size: usize,
) -> Result<MojoQuantizedTensor, String> {
    validate_encode_inputs(data.len(), bits, group_size)?;
    Err("Mojo native library unavailable; build libturbo_quant_mojo.dylib first".to_string())
}

#[cfg(not(all(feature = "mojo-native", mojo_native)))]
pub(super) fn mojo_try_decode(
    shape: &[usize],
    packed: &[u8],
    scales: &[f32],
    zeros: &[f32],
    n: usize,
    group_size: usize,
    bits: u8,
) -> Result<Vec<f32>, String> {
    validate_decode_inputs(
        shape,
        packed.len(),
        scales.len(),
        zeros.len(),
        n,
        group_size,
        bits,
    )?;
    Err("Mojo native library unavailable; build libturbo_quant_mojo.dylib first".to_string())
}

#[cfg(all(feature = "mojo-ffi", feature = "mojo-native", mojo_native))]
pub fn mojo_gemv_decode(
    weights: &[f32],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    unsafe {
        tq_gemv_decode(
            weights.as_ptr(),
            input.as_ptr(),
            output.as_mut_ptr(),
            rows,
            cols,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::EncodeOutputGuard;

    // ── addr_should_free policy ──────────────────────────────────────
    //
    // These exercise the guard's free-or-skip policy without actually
    // calling `tq_mojo_free` (which is unsafe and would dereference the
    // fake address). The policy is the only thing the Drop relies on;
    // covering it directly pins down the safety invariants.

    #[test]
    fn addr_should_free_zero_is_false() {
        assert!(!EncodeOutputGuard::addr_should_free(0));
    }

    #[test]
    fn addr_should_free_positive_small_is_true() {
        assert!(EncodeOutputGuard::addr_should_free(1));
        assert!(EncodeOutputGuard::addr_should_free(0x1000));
    }

    #[test]
    fn addr_should_free_negative_is_false() {
        // Negative addresses must NEVER be passed to tq_mojo_free.
        assert!(!EncodeOutputGuard::addr_should_free(-1));
        assert!(!EncodeOutputGuard::addr_should_free(isize::MIN));
    }

    #[test]
    fn addr_should_free_isize_max_is_true() {
        assert!(EncodeOutputGuard::addr_should_free(isize::MAX));
    }

    #[test]
    fn addr_should_free_is_const() {
        // Compiles as a const expression so the policy can be used in
        // const contexts (static tables, etc.) — the const bindings are
        // evaluated at compile time, then we runtime-assert via shadowing
        // locals so the test itself is not a clippy::assertions_on_constants.
        const OK: bool = EncodeOutputGuard::addr_should_free(42);
        const NOT_OK: bool = EncodeOutputGuard::addr_should_free(-7);
        let ok_runtime = OK;
        let not_ok_runtime = NOT_OK;
        assert!(ok_runtime);
        assert!(!not_ok_runtime);
    }

    // ── Drop end-to-end with only zero / negative addresses ──────────
    //
    // Verifies the destructor does not panic on addresses that the policy
    // rejects. If anyone ever changes the policy to free negative or zero
    // addresses, these tests would still pass at the policy layer but the
    // Drop would dereference UB pointers — the gate is the policy, which
    // the tests above pin down.

    #[test]
    fn drop_with_only_zero_addresses_does_not_call_free() {
        let mut g = EncodeOutputGuard::empty();
        g.arm(0, 0, 0, 0);
        // Drop runs here; policy says: free nothing.
        drop(g);
    }

    #[test]
    fn drop_with_negative_addresses_does_not_call_free() {
        let mut g = EncodeOutputGuard::empty();
        // Negative entries that the policy must skip. We never arm
        // positive addresses here so Drop must not invoke tq_mojo_free.
        g.arm(-1, -2, -3, -4);
        drop(g);
    }

    #[test]
    fn drop_with_mixed_zero_and_negative_does_not_call_free() {
        let mut g = EncodeOutputGuard::empty();
        g.arm(0, -1, 0, isize::MIN);
        drop(g);
    }
}
