//! Property 1 — `EncodeRequest::validate()` is total.
//!
//! For every randomized triple `(n, bits, group_size)` plus a synthetic
//! data buffer, `validate()` must either return `Ok(())` or return an
//! `Err` whose discriminant is one of the nine documented `Status`
//! codes. It must never panic and never surface an undocumented code.

use super::V1;
use native_abi::{EncodeRequest, Status};
use proptest::prelude::*;

proptest! {
    #[test]
    fn encode_validate_is_total_for_random_inputs(
        n in 0usize..64,
        bits in any::<u8>(),
        group_size in 0usize..64,
    ) {
        let data = vec![0.5f32; n];
        let mut req = EncodeRequest::zeroed();
        req.abi = V1;
        req.data_ptr = data.as_ptr();
        req.n = n;
        req.bits = bits;
        req.group_size = group_size;
        let mut shape: *mut usize = std::ptr::null_mut();
        let mut packed: *mut u8 = std::ptr::null_mut();
        let mut scales: *mut f32 = std::ptr::null_mut();
        let mut zeros: *mut f32 = std::ptr::null_mut();
        req.out_shape = &mut shape as *mut _;
        req.out_shape_capacity = 16;
        req.out_packed = &mut packed as *mut _;
        req.out_packed_capacity = 16;
        req.out_scales = &mut scales as *mut _;
        req.out_scales_capacity = 16;
        req.out_zeros = &mut zeros as *mut _;
        req.out_zeros_capacity = 16;

        let v = req.validate();
        // All failures must be among the documented statuses; the
        // result is *some* Status — never a panic.
        match v {
            Ok(()) => { /* valid request */ }
            Err(s) => {
                let i: i32 = s.into();
                let back = Status::try_from(i)
                    .expect("validate() must return a known Status");
                // Sanity: the known codes are 0..=8.
                assert!(back == Status::Ok
                    || back == Status::ErrNullArg
                    || back == Status::ErrInvalidBits
                    || back == Status::ErrInvalidGroupSize
                    || back == Status::ErrNonFiniteInput
                    || back == Status::ErrOverflow
                    || back == Status::ErrAllocation
                    || back == Status::ErrVersionMismatch
                    || back == Status::ErrBackend,
                    "undocumented Status code: {back:?}");
            }
        }
    }
}