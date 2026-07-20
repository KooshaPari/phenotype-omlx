//! Property 3 — with valid bits, `group_size == 0` must be rejected as
//! `ErrInvalidGroupSize`. We feed `bits` from the valid set so the
//! validator reaches the group_size check (bits is checked first,
//! which is the documented ordering).

use super::{V1, VALID_BITS};
use native_abi::{EncodeRequest, Status};
use proptest::prelude::*;

proptest! {
    #[test]
    fn zero_group_size_rejected(bits in proptest::sample::select(VALID_BITS.to_vec())) {
        // Encode side.
        let data = [1.0f32; 4];
        let mut req = EncodeRequest::zeroed();
        req.abi = V1;
        req.data_ptr = data.as_ptr();
        req.n = 4;
        req.bits = bits;
        req.group_size = 0;
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
            Err(Status::ErrInvalidGroupSize),
            "group_size=0 with valid bits must reject as ErrInvalidGroupSize"
        );
    }
}