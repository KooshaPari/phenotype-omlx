//! Property 2 — `bits` outside the allowed set `{2, 3, 4, 8}` must be
//! rejected as `ErrInvalidBits`. Proptest drives `bits: u8` and filters
//! out the allowed set, so every iteration exercises a forbidden width.

use super::{V1, VALID_BITS};
use native_abi::{EncodeRequest, Status};
use proptest::prelude::*;

proptest! {
    #[test]
    fn encode_rejects_bits_outside_allowed_set(
        bits in any::<u8>().prop_filter("outside allowed set", |&b| !VALID_BITS.contains(&b)),
        group_size in 1usize..64,
    ) {
        let data = vec![1.0f32; group_size.max(1)];
        let mut req = EncodeRequest::zeroed();
        req.abi = V1;
        req.data_ptr = data.as_ptr();
        req.n = data.len();
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
        assert_eq!(
            req.validate(),
            Err(Status::ErrInvalidBits),
            "bits={bits} must be rejected as ErrInvalidBits"
        );
    }
}