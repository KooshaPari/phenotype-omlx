// turbo-quant-nim — Rust workspace member for Nim binding.

use native_abi::{
    DecodeRequest as RustDecodeRequest, EncodeRequest as RustEncodeRequest,
    Status as RustStatus, ABI_VERSION_CURRENT,
};

#[derive(Debug, Clone)]
pub struct NimQuantizedTensor {
    pub shape: Vec<usize>,
    pub packed: Vec<u8>,
    pub scales: Vec<f32>,
    pub zeros: Vec<f32>,
}

impl NimQuantizedTensor {
    pub fn encode(data: &[f32], bits: u8, group_size: usize) -> Result<Self, String> {
        encode_v1(data, bits, group_size).map_err(|e| format!("{:?}", e))
    }

    pub fn decode(&self, n: usize, group_size: usize, bits: u8) -> Vec<f32> {
        let mut out = vec![0.0; n];
        let status = decode_v1(&self.packed, &self.scales, &self.zeros, n, group_size, bits, &mut out);
        if status != RustStatus::Ok {
            eprintln!("Nim decode failed with status {:?}", status);
        }
        out
    }
}

pub fn encode_v1(data: &[f32], bits: u8, group_size: usize) -> Result<NimQuantizedTensor, RustStatus> {
    let n_groups = native_abi::group_count(data.len(), group_size);
    let packed_len = native_abi::expected_packed_len(data.len(), bits);

    let mut shape_storage: Vec<usize> = vec![0; 1];
    let mut packed_storage: Vec<u8> = vec![0; packed_len];
    let mut scales_storage: Vec<f32> = vec![0.0; n_groups];
    let mut zeros_storage: Vec<f32> = vec![0.0; n_groups];

    let mut shape_ptr = shape_storage.as_mut_ptr();
    let mut packed_ptr = packed_storage.as_mut_ptr();
    let mut scales_ptr = scales_storage.as_mut_ptr();
    let mut zeros_ptr = zeros_storage.as_mut_ptr();

    let mut req = RustEncodeRequest::zeroed();
    req.abi = ABI_VERSION_CURRENT;
    req.data_ptr = data.as_ptr();
    req.n = data.len();
    req.bits = bits;
    req.group_size = group_size;
    req.out_shape = &mut shape_ptr;
    req.out_shape_capacity = shape_storage.len();
    req.out_packed = &mut packed_ptr;
    req.out_packed_capacity = packed_storage.len();
    req.out_scales = &mut scales_ptr;
    req.out_scales_capacity = scales_storage.len();
    req.out_zeros = &mut zeros_ptr;
    req.out_zeros_capacity = zeros_storage.len();

    let result = unsafe { native_abi::dispatch::encode_v1(&req) };
    if result.status != RustStatus::Ok {
        return Err(result.status);
    }

    Ok(NimQuantizedTensor {
        shape: shape_storage,
        packed: packed_storage,
        scales: scales_storage,
        zeros: zeros_storage,
    })
}

pub fn decode_v1(
    packed: &[u8],
    scales: &[f32],
    zeros: &[f32],
    n: usize,
    group_size: usize,
    bits: u8,
    out: &mut [f32],
) -> RustStatus {
    let mut req = RustDecodeRequest::zeroed();
    req.abi = ABI_VERSION_CURRENT;
    req.packed_ptr = packed.as_ptr();
    req.packed_len = packed.len();
    req.scales_ptr = scales.as_ptr();
    req.zeros_ptr = zeros.as_ptr();
    req.n = n;
    req.group_size = group_size;
    req.bits = bits;
    req.out_ptr = out.as_mut_ptr();

    unsafe { native_abi::dispatch::decode_v1(&req) }
}

pub fn placeholder() -> &'static str {
    "turbo-quant-nim"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nim_placeholder() {
        assert_eq!(super::placeholder(), "turbo-quant-nim");
    }

    #[test]
    fn nim_encode_decode_smoke() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = NimQuantizedTensor::encode(&data, 4, 2).unwrap();
        let decoded = tensor.decode(4, 2, 4);
        assert_eq!(decoded.len(), 4);
    }
}
