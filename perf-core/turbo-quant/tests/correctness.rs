use turbo_quant::{QuantError, QuantizedTensor};

#[test]
fn encode_rejects_unsupported_bits() {
    for bits in [0, 1, 9] {
        let result = QuantizedTensor::encode_uniform(&[1.0], bits, 64);
        assert!(matches!(result, Err(QuantError::InvalidBits(actual)) if actual == bits));
    }
}

#[test]
fn encode_rejects_zero_group_size() {
    let result = QuantizedTensor::encode_uniform(&[1.0], 4, 0);
    assert!(matches!(result, Err(QuantError::InvalidGroupSize)));
}

#[test]
fn encode_rejects_non_finite_data() {
    let result = QuantizedTensor::encode_uniform(&[f32::NAN], 4, 64);
    assert!(matches!(result, Err(QuantError::NonFiniteInput)));
}

#[test]
fn constructor_rejects_inconsistent_payload() {
    let result = QuantizedTensor::try_from_parts(vec![2], 4, 64, vec![], vec![1.0], vec![0.0]);
    assert!(matches!(result, Err(QuantError::InvalidMetadata(_))));
}

#[test]
fn constructor_rejects_non_finite_metadata() {
    let result =
        QuantizedTensor::try_from_parts(vec![2], 4, 64, vec![0], vec![f32::INFINITY], vec![0.0]);
    assert!(matches!(result, Err(QuantError::NonFiniteInput)));
}

#[test]
fn decode_rejects_output_length_mismatch() {
    let tensor = QuantizedTensor::encode_uniform(&[1.0, 2.0], 4, 64).unwrap();
    let mut output = vec![0.0; 1];
    assert!(matches!(
        tensor.decode_uniform(&mut output),
        Err(QuantError::OutputLengthMismatch {
            expected: 2,
            actual: 1
        })
    ));
}

#[test]
fn round_trips_supported_bit_widths() {
    let input: Vec<f32> = (0..129).map(|value| value as f32 / 17.0).collect();
    for bits in [2, 3, 4] {
        let tensor = QuantizedTensor::encode_uniform(&input, bits, 17).unwrap();
        let mut output = vec![0.0; input.len()];
        tensor.decode_uniform(&mut output).unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
        assert_eq!(tensor.bits, bits);
        assert_eq!(tensor.group_size, 17);
    }
}
