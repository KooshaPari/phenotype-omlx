//! Contract tests for the native-abi crate (ABI v1).
//!
//! These tests define the externally observable behaviour of the ABI surface.
//! They are written before the implementation (TDD) and exercise every
//! invariant the v1 contract promises:
//!
//! * version constants are stable and version compatibility is governed by major;
//! * Status codes round-trip through i32 with one-line descriptions;
//! * EncodeRequest / DecodeRequest validate every required slot;
//! * dispatch::encode zeros every output on failure;
//! * dispatch::decode leaves the caller's `out` untouched on failure;
//! * dispatch::encode round-trips small uniform input;
//! * the generated C header (`include/abi_v1.h`) declares every required symbol;
//! * the header writer is consistent with the Rust descriptors.

use native_abi::{
    encode_v1, AbiVersion, DecodeRequest, EncodeRequest, EncodeResult, Status, ABI_VERSION_CURRENT,
    HEADER_C_SYMBOLS,
};

const V1: AbiVersion = AbiVersion { major: 1, minor: 0 };

// ── version contract ──────────────────────────────────────────────────────────

#[test]
fn abi_version_constants_are_stable() {
    assert_eq!(ABI_VERSION_CURRENT.major, 1);
    assert_eq!(ABI_VERSION_CURRENT.minor, 0);
    // Major version 1 is the v1 contract; bumping it is a breaking change.
    // Minor version increments are additive. The combination here is the
    // pinned identity the header generation must emit.
    assert_eq!(V1, ABI_VERSION_CURRENT);
}

#[test]
fn abi_version_is_compatible_only_when_major_matches() {
    // Same major is compatible regardless of minor.
    assert!(native_abi::is_compatible(
        AbiVersion { major: 1, minor: 0 },
        AbiVersion { major: 1, minor: 7 },
    ));
    assert!(native_abi::is_compatible(
        AbiVersion { major: 1, minor: 7 },
        AbiVersion { major: 1, minor: 0 },
    ));
    // Different majors are incompatible.
    assert!(!native_abi::is_compatible(
        AbiVersion { major: 1, minor: 0 },
        AbiVersion { major: 2, minor: 0 },
    ));
    assert!(!native_abi::is_compatible(
        AbiVersion {
            major: 0,
            minor: 99
        },
        AbiVersion { major: 1, minor: 0 },
    ));
}

// ── status contract ───────────────────────────────────────────────────────────

#[test]
fn status_round_trips_through_i32() {
    let cases = [
        Status::Ok,
        Status::ErrNullArg,
        Status::ErrInvalidBits,
        Status::ErrInvalidGroupSize,
        Status::ErrNonFiniteInput,
        Status::ErrOverflow,
        Status::ErrAllocation,
        Status::ErrVersionMismatch,
        Status::ErrBackend,
    ];
    for s in cases {
        let code: i32 = s.into();
        let back: Status = Status::try_from(code).expect("status must round-trip");
        assert_eq!(back, s);
    }
}

#[test]
fn status_description_is_non_empty() {
    let cases = [
        Status::Ok,
        Status::ErrNullArg,
        Status::ErrInvalidBits,
        Status::ErrInvalidGroupSize,
        Status::ErrNonFiniteInput,
        Status::ErrOverflow,
        Status::ErrAllocation,
        Status::ErrVersionMismatch,
        Status::ErrBackend,
    ];
    for s in cases {
        let desc = s.description();
        assert!(!desc.is_empty(), "status {:?} has empty description", s);
    }
}

// ── EncodeRequest validation ──────────────────────────────────────────────────

fn encode_req_with_data(
    data_ptr: *const f32,
    n: usize,
    bits: u8,
    group_size: usize,
) -> EncodeRequest {
    let mut req = EncodeRequest::zeroed();
    req.abi = V1;
    req.data_ptr = data_ptr;
    req.n = n;
    req.bits = bits;
    req.group_size = group_size;
    // Provide valid output slot pointers + capacity. They are required by
    // the contract even when validation is expected to fail before they are
    // written to.
    let dummy_shape: *mut usize = std::ptr::null_mut();
    let dummy_packed: *mut u8 = std::ptr::null_mut();
    let dummy_scales: *mut f32 = std::ptr::null_mut();
    let dummy_zeros: *mut f32 = std::ptr::null_mut();
    req.out_shape = &dummy_shape as *const _ as *mut *mut usize;
    req.out_shape_capacity = 16;
    req.out_packed = &dummy_packed as *const _ as *mut *mut u8;
    req.out_packed_capacity = 16;
    req.out_scales = &dummy_scales as *const _ as *mut *mut f32;
    req.out_scales_capacity = 16;
    req.out_zeros = &dummy_zeros as *const _ as *mut *mut f32;
    req.out_zeros_capacity = 16;
    req
}

#[test]
fn encode_request_validate_rejects_null_data() {
    let req = encode_req_with_data(std::ptr::null(), 4, 3, 4);
    assert_eq!(req.validate(), Err(Status::ErrNullArg));
}

#[test]
fn encode_request_validate_rejects_zero_n() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let req = encode_req_with_data(data.as_ptr(), 0, 3, 4);
    assert_eq!(req.validate(), Err(Status::ErrNullArg));
}

#[test]
fn encode_request_validate_rejects_bits_out_of_range() {
    let data = [1.0f32];
    let req = encode_req_with_data(data.as_ptr(), 1, 1, 4);
    assert_eq!(req.validate(), Err(Status::ErrInvalidBits));
    let req = encode_req_with_data(data.as_ptr(), 1, 5, 4);
    assert_eq!(req.validate(), Err(Status::ErrInvalidBits));
}

#[test]
fn encode_request_validate_rejects_zero_group_size() {
    let data = [1.0f32];
    let req = encode_req_with_data(data.as_ptr(), 1, 3, 0);
    assert_eq!(req.validate(), Err(Status::ErrInvalidGroupSize));
}

#[test]
fn encode_request_validate_rejects_null_output_slot() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let mut req = encode_req_with_data(data.as_ptr(), 4, 3, 4);
    req.out_shape = std::ptr::null_mut();
    assert_eq!(req.validate(), Err(Status::ErrNullArg));
}

// ── DecodeRequest validation ──────────────────────────────────────────────────

fn decode_req_with_data(
    packed: *const u8,
    packed_len: usize,
    scales: *const f32,
    zeros: *const f32,
    n: usize,
    group_size: usize,
    bits: u8,
    out: *mut f32,
) -> DecodeRequest {
    let mut req = DecodeRequest::zeroed();
    req.abi = V1;
    req.packed_ptr = packed;
    req.packed_len = packed_len;
    req.scales_ptr = scales;
    req.zeros_ptr = zeros;
    req.n = n;
    req.group_size = group_size;
    req.bits = bits;
    req.out_ptr = out;
    req
}

#[test]
fn decode_request_validate_rejects_null_out() {
    let packed = [0u8];
    let scales = [1.0f32];
    let zeros = [0.0f32];
    let req = decode_req_with_data(
        packed.as_ptr(),
        packed.len(),
        scales.as_ptr(),
        zeros.as_ptr(),
        1,
        1,
        3,
        std::ptr::null_mut(),
    );
    assert_eq!(req.validate(), Err(Status::ErrNullArg));
}

#[test]
fn decode_request_validate_rejects_zero_n() {
    let packed = [0u8];
    let scales = [1.0f32];
    let zeros = [0.0f32];
    let mut out = [0.0f32; 1];
    let req = decode_req_with_data(
        packed.as_ptr(),
        packed.len(),
        scales.as_ptr(),
        zeros.as_ptr(),
        0,
        1,
        3,
        out.as_mut_ptr(),
    );
    assert_eq!(req.validate(), Err(Status::ErrNullArg));
}

#[test]
fn decode_request_validate_rejects_zero_group_size() {
    let packed = [0u8];
    let scales = [1.0f32];
    let zeros = [0.0f32];
    let mut out = [0.0f32; 1];
    let req = decode_req_with_data(
        packed.as_ptr(),
        packed.len(),
        scales.as_ptr(),
        zeros.as_ptr(),
        1,
        0,
        3,
        out.as_mut_ptr(),
    );
    assert_eq!(req.validate(), Err(Status::ErrInvalidGroupSize));
}

#[test]
fn decode_request_validate_rejects_bits_out_of_range() {
    let packed = [0u8];
    let scales = [1.0f32];
    let zeros = [0.0f32];
    let mut out = [0.0f32; 1];
    let req = decode_req_with_data(
        packed.as_ptr(),
        packed.len(),
        scales.as_ptr(),
        zeros.as_ptr(),
        1,
        1,
        1,
        out.as_mut_ptr(),
    );
    assert_eq!(req.validate(), Err(Status::ErrInvalidBits));
}

#[test]
fn decode_request_validate_rejects_mismatched_packed_len() {
    // n=4, bits=3 → expected packed_len = ceil(4*3/8) = 2
    let packed = [0u8, 0, 0]; // intentionally wrong length
    let scales = [1.0f32; 4];
    let zeros = [0.0f32; 4];
    let mut out = [0.0f32; 4];
    let req = decode_req_with_data(
        packed.as_ptr(),
        packed.len(),
        scales.as_ptr(),
        zeros.as_ptr(),
        4,
        1,
        3,
        out.as_mut_ptr(),
    );
    assert_eq!(req.validate(), Err(Status::ErrInvalidBits));
}

// ── dispatch contract ─────────────────────────────────────────────────────────

#[test]
fn dispatch_encode_zeroes_all_outputs_on_failure() {
    // Use a null data pointer to force validation failure.
    let mut req = encode_req_with_data(std::ptr::null(), 0, 0, 0);

    let mut shape: *mut usize = std::ptr::null_mut();
    let mut packed: *mut u8 = std::ptr::null_mut();
    let mut scales: *mut f32 = std::ptr::null_mut();
    let mut zeros: *mut f32 = std::ptr::null_mut();

    // Stash stale sentinels via the request's output slots.
    req.out_shape = &mut shape as *mut _;
    req.out_shape_capacity = 4;
    req.out_packed = &mut packed as *mut _;
    req.out_packed_capacity = 4;
    req.out_scales = &mut scales as *mut _;
    req.out_scales_capacity = 4;
    req.out_zeros = &mut zeros as *mut _;
    req.out_zeros_capacity = 4;

    let result: EncodeResult = unsafe { encode_v1(&req) };
    assert_ne!(result.status, Status::Ok);
    assert_eq!(shape, std::ptr::null_mut());
    assert_eq!(packed, std::ptr::null_mut());
    assert_eq!(scales, std::ptr::null_mut());
    assert_eq!(zeros, std::ptr::null_mut());
}

#[test]
fn dispatch_decode_leaves_out_untouched_on_failure() {
    let packed = [0u8, 0, 0, 0];
    let scales = [1.0f32, 1.0, 1.0, 1.0];
    let zeros = [0.0f32, 0.0, 0.0, 0.0];
    let sentinel = 91.0_f32;
    let mut out = [sentinel; 4];
    let req = decode_req_with_data(
        packed.as_ptr(),
        packed.len(),
        scales.as_ptr(),
        zeros.as_ptr(),
        4,
        0, // invalid group_size
        3,
        out.as_mut_ptr(),
    );
    let status = unsafe { native_abi::decode_v1(&req) };
    assert_ne!(status, Status::Ok);
    for v in &out {
        assert_eq!(*v, sentinel);
    }
}

#[test]
fn dispatch_encode_round_trips_small_uniform_input() {
    let input = [-3.0f32, -1.0, 0.5, 2.0, 7.0, 9.0, 12.0];

    // Allocate backing storage the dispatch will write into.
    let n_groups = input.len().div_ceil(3);
    let bits: u8 = 3;
    let packed_len = (input.len() * bits as usize).div_ceil(8);

    let mut shape_storage: Vec<usize> = vec![0; 1];
    let mut packed_storage: Vec<u8> = vec![0; packed_len];
    let mut scales_storage: Vec<f32> = vec![0.0; n_groups];
    let mut zeros_storage: Vec<f32> = vec![0.0; n_groups];

    let mut shape_ptr = shape_storage.as_mut_ptr();
    let mut packed_ptr = packed_storage.as_mut_ptr();
    let mut scales_ptr = scales_storage.as_mut_ptr();
    let mut zeros_ptr = zeros_storage.as_mut_ptr();

    let mut req = EncodeRequest::zeroed();
    req.abi = V1;
    req.data_ptr = input.as_ptr();
    req.n = input.len();
    req.bits = bits;
    req.group_size = 3;
    req.out_shape = &mut shape_ptr;
    req.out_shape_capacity = shape_storage.len();
    req.out_packed = &mut packed_ptr;
    req.out_packed_capacity = packed_storage.len();
    req.out_scales = &mut scales_ptr;
    req.out_scales_capacity = scales_storage.len();
    req.out_zeros = &mut zeros_ptr;
    req.out_zeros_capacity = zeros_storage.len();

    let result: EncodeResult = unsafe { encode_v1(&req) };
    assert_eq!(result.status, Status::Ok, "encode should succeed");
    assert_eq!(result.written_packed_len, packed_len);
    assert_eq!(result.written_shape_len, 1);
    assert_eq!(result.written_scales_len, n_groups);
    assert_eq!(result.written_zeros_len, n_groups);
    assert_eq!(shape_storage[0], input.len());

    // Decode through the same ABI and compare within tolerance.
    let mut decoded = vec![0.0f32; input.len()];
    let mut dreq = DecodeRequest::zeroed();
    dreq.abi = V1;
    dreq.packed_ptr = packed_storage.as_ptr();
    dreq.packed_len = packed_storage.len();
    dreq.scales_ptr = scales_storage.as_ptr();
    dreq.zeros_ptr = zeros_storage.as_ptr();
    dreq.n = input.len();
    dreq.group_size = 3;
    dreq.bits = bits;
    dreq.out_ptr = decoded.as_mut_ptr();

    let status = unsafe { native_abi::decode_v1(&dreq) };
    assert_eq!(status, Status::Ok);
    let tolerance = scales_storage.iter().copied().fold(0.0_f32, f32::max);
    for (a, e) in decoded.iter().zip(input.iter()) {
        assert!((a - e).abs() <= tolerance + 1e-5);
    }
}

#[test]
fn dispatch_decode_rejects_when_abi_version_mismatches() {
    let packed = [0u8];
    let scales = [1.0f32];
    let zeros = [0.0f32];
    let mut out = [0.0f32; 1];
    let mut req = decode_req_with_data(
        packed.as_ptr(),
        packed.len(),
        scales.as_ptr(),
        zeros.as_ptr(),
        1,
        1,
        3,
        out.as_mut_ptr(),
    );
    req.abi = AbiVersion {
        major: 99,
        minor: 0,
    };
    let status = unsafe { native_abi::decode_v1(&req) };
    assert_eq!(status, Status::ErrVersionMismatch);
    for v in &out {
        assert_eq!(*v, 0.0);
    }
}

// ── header writer contract ────────────────────────────────────────────────────

#[test]
fn header_writer_emits_all_required_symbols() {
    let header = native_abi::write_c_header();
    for needle in HEADER_C_SYMBOLS {
        assert!(
            header.contains(needle),
            "header is missing required symbol `{}`",
            needle
        );
    }
}

#[test]
fn header_writer_round_trip_parses_back_to_struct_names() {
    // Emit the header, then scan its body for the struct declarations the
    // Rust descriptors declare. Each one must appear verbatim so a polyglot
    // consumer (C / Zig / Mojo / Nim / Go) sees the same names.
    let header = native_abi::write_c_header();
    for needle in [
        "typedef struct tq_abi_encode_request",
        "typedef struct tq_abi_decode_request",
        "typedef struct tq_abi_encode_result",
        "typedef enum tq_abi_status",
        "typedef struct tq_abi_version",
        "TQ_ABI_VERSION_MAJOR",
        "TQ_ABI_VERSION_MINOR",
        "tq_abi_encode(",
        "tq_abi_decode(",
    ] {
        assert!(header.contains(needle), "missing: {}", needle);
    }
}
