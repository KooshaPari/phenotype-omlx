use crate::{PolyglotQuantizer, QuantizedTensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolyglotBackend {
    Cpu,
    C,
    Zig,
    Nim,
    Mojo,
}

impl PolyglotBackend {
    pub fn all() -> &'static [PolyglotBackend] {
        &[
            PolyglotBackend::Cpu,
            PolyglotBackend::C,
            PolyglotBackend::Zig,
            PolyglotBackend::Nim,
            PolyglotBackend::Mojo,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            PolyglotBackend::Cpu => "Cpu",
            PolyglotBackend::C => "C",
            PolyglotBackend::Zig => "Zig",
            PolyglotBackend::Nim => "Nim",
            PolyglotBackend::Mojo => "Mojo",
        }
    }
}

impl PolyglotQuantizer for PolyglotBackend {
    fn encode(&self, data: &[f32], bits: u8, group_size: usize) -> Result<QuantizedTensor, String> {
        match self {
            PolyglotBackend::Cpu => {
                if data.is_empty() {
                    return Err("turbo-quant: data must be non-empty".to_string());
                }
                if !(2..=4).contains(&bits) {
                    return Err(format!("turbo-quant: bits must be 2..=4, got {bits}"));
                }
                if group_size == 0 {
                    return Err("turbo-quant: group_size must be > 0".to_string());
                }
                if data.iter().any(|v| !v.is_finite()) {
                    return Err("turbo-quant: data contains non-finite values".to_string());
                }
                Ok(QuantizedTensor::encode_uniform(data, bits, group_size))
            }
            PolyglotBackend::C => {
                let c_res = turbo_quant_c::encode_v1(data, bits, group_size);
                match c_res {
                    Ok(ct) => Ok(QuantizedTensor {
                        shape: ct.shape,
                        bits,
                        group_size,
                        packed: ct.packed,
                        scales: ct.scales,
                        zeros: ct.zeros,
                    }),
                    Err(st) => Err(format!("C backend encode failed with status {:?}", st)),
                }
            }
            PolyglotBackend::Zig => {
                let z_res = turbo_quant_zig::ZigQuantizedTensor::encode_v1(data, bits, group_size);
                match z_res {
                    Ok(zt) => Ok(QuantizedTensor {
                        shape: zt.shape,
                        bits,
                        group_size,
                        packed: zt.packed,
                        scales: zt.scales,
                        zeros: zt.zeros,
                    }),
                    Err(st) => Err(format!("Zig backend encode failed with status {:?}", st)),
                }
            }
            PolyglotBackend::Nim => {
                let n_res = turbo_quant_nim::NimQuantizedTensor::encode(data, bits, group_size);
                match n_res {
                    Ok(nt) => Ok(QuantizedTensor {
                        shape: nt.shape,
                        bits,
                        group_size,
                        packed: nt.packed,
                        scales: nt.scales,
                        zeros: nt.zeros,
                    }),
                    Err(err) => Err(format!("Nim backend encode failed: {err}")),
                }
            }
            PolyglotBackend::Mojo => {
                let m_res = turbo_quant_mojo::MojoQuantizedTensor::encode(data, bits, group_size);
                match m_res {
                    Ok(mt) => Ok(QuantizedTensor {
                        shape: mt.shape,
                        bits,
                        group_size,
                        packed: mt.packed,
                        scales: mt.scales,
                        zeros: mt.zeros,
                    }),
                    Err(err) => Err(format!("Mojo backend encode failed: {err}")),
                }
            }
        }
    }

    fn decode(&self, tensor: &QuantizedTensor, out: &mut [f32]) -> Result<(), String> {
        let expected_len = tensor.shape.iter().product::<usize>();
        if out.len() != expected_len {
            return Err(format!(
                "output slice length ({}) does not match tensor element count ({})",
                out.len(),
                expected_len
            ));
        }

        match self {
            PolyglotBackend::Cpu => {
                tensor.decode_uniform(out);
                Ok(())
            }
            PolyglotBackend::C => {
                let status = turbo_quant_c::decode_v1(
                    &tensor.packed,
                    &tensor.scales,
                    &tensor.zeros,
                    expected_len,
                    tensor.group_size,
                    tensor.bits,
                    out,
                );
                if status == native_abi::Status::Ok {
                    Ok(())
                } else {
                    Err(format!("C backend decode failed with status {:?}", status))
                }
            }
            PolyglotBackend::Zig => {
                let zt = turbo_quant_zig::ZigQuantizedTensor {
                    shape: tensor.shape.clone(),
                    packed: tensor.packed.clone(),
                    scales: tensor.scales.clone(),
                    zeros: tensor.zeros.clone(),
                };
                let status = zt.decode_v1(expected_len, tensor.group_size, tensor.bits, out);
                if status == native_abi::Status::Ok {
                    Ok(())
                } else {
                    Err(format!(
                        "Zig backend decode failed with status {:?}",
                        status
                    ))
                }
            }
            PolyglotBackend::Nim => {
                let status = turbo_quant_nim::decode_v1(
                    &tensor.packed,
                    &tensor.scales,
                    &tensor.zeros,
                    expected_len,
                    tensor.group_size,
                    tensor.bits,
                    out,
                );
                if status == native_abi::Status::Ok {
                    Ok(())
                } else {
                    Err(format!(
                        "Nim backend decode failed with status {:?}",
                        status
                    ))
                }
            }
            PolyglotBackend::Mojo => {
                let mt = turbo_quant_mojo::MojoQuantizedTensor {
                    shape: tensor.shape.clone(),
                    packed: tensor.packed.clone(),
                    scales: tensor.scales.clone(),
                    zeros: tensor.zeros.clone(),
                };
                match mt.try_decode(expected_len, tensor.group_size, tensor.bits) {
                    Ok(decoded) => {
                        out.copy_from_slice(&decoded);
                        Ok(())
                    }
                    Err(err) => Err(format!("Mojo backend decode failed: {err}")),
                }
            }
        }
    }
}
