// TurboQuant pyo3 surface — strict validation + metadata round-trip.
//
// All numeric inputs are validated up-front and bad inputs raise
// `PyValueError` with a descriptive message rather than panicking or
// silently corrupting packed bytes.

// PyO3's generated wrappers contain identity conversions which trigger this lint.
#![allow(clippy::useless_conversion)]

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::panic::{catch_unwind, AssertUnwindSafe};

use turbo_quant::{QuantizedTensor, TurboMode};

/// Central validator for encode / decode inputs. Returns
/// `PyValueError` with a human-readable message for every failure mode
/// the API contract forbids.
fn validate_dimensions(
    data_len: usize,
    n: usize,
    group_size: usize,
    bits: u8,
    scales_len: usize,
    zeros_len: usize,
    packed_len: usize,
) -> PyResult<()> {
    if n == 0 {
        return Err(PyValueError::new_err("n must be > 0"));
    }
    if data_len != n {
        return Err(PyValueError::new_err(format!(
            "data length {data_len} does not match n {n}"
        )));
    }
    validate_bits(bits)?;
    if group_size == 0 {
        return Err(PyValueError::new_err("group_size must be > 0"));
    }
    let expected_groups = n.div_ceil(group_size);
    if scales_len != expected_groups || zeros_len != expected_groups {
        return Err(PyValueError::new_err(format!(
            "scales ({scales_len}) and zeros ({zeros_len}) must each have \
             {expected_groups} entries (one per group)"
        )));
    }
    let expected_packed = n
        .checked_mul(bits as usize)
        .ok_or_else(|| PyValueError::new_err("n * bits exceeds platform limits"))?
        .div_ceil(8);
    if packed_len != expected_packed {
        return Err(PyValueError::new_err(format!(
            "packed length {packed_len} does not match expected {expected_packed} \
             (= ceil(n * bits / 8) = ceil({n} * {bits} / 8))"
        )));
    }
    Ok(())
}

fn validate_bits(bits: u8) -> PyResult<()> {
    if !(2..=4).contains(&bits) {
        return Err(PyValueError::new_err(format!(
            "bits must be 2, 3, or 4 (got {bits})"
        )));
    }
    Ok(())
}

fn contain_panic<T>(operation: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    catch_unwind(AssertUnwindSafe(operation)).unwrap_or_else(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("unknown Rust panic");
        Err(PyRuntimeError::new_err(format!(
            "turbo-quant core panicked: {message}"
        )))
    })
}

fn validate_finite(name: &str, values: &[f32]) -> PyResult<()> {
    if let Some((idx, v)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name}[{idx}] = {v} is not finite"
        )));
    }
    Ok(())
}

#[pyfunction]
pub fn turbo_quant_label_for_bits(bits: u8) -> PyResult<String> {
    validate_bits(bits)?;
    let m = match bits {
        4 => TurboMode::Asymmetric4,
        3 => TurboMode::Symmetric3,
        2 => TurboMode::Symmetric2,
        _ => unreachable!("validated above"),
    };
    Ok(m.label().to_string())
}

#[pyfunction]
pub fn turbo_quant_encode(
    py: Python<'_>,
    data: Vec<f32>,
    group_size: usize,
    bits: u8,
) -> PyResult<PyObject> {
    if data.is_empty() {
        return Err(PyValueError::new_err("data must be non-empty"));
    }
    validate_finite("data", &data)?;
    validate_bits(bits)?;
    if group_size == 0 {
        return Err(PyValueError::new_err("group_size must be > 0"));
    }
    data.len()
        .checked_mul(bits as usize)
        .ok_or_else(|| PyValueError::new_err("data length * bits exceeds platform limits"))?;
    let q = contain_panic(|| Ok(QuantizedTensor::encode_uniform(&data, bits, group_size)))?;
    let dict = PyDict::new_bound(py);
    dict.set_item("shape", q.shape.clone())?;
    dict.set_item("packed", q.packed.clone())?;
    dict.set_item("scales", q.scales.clone())?;
    dict.set_item("zeros", q.zeros.clone())?;
    dict.set_item("bits", q.bits)?;
    dict.set_item("group_size", q.group_size)?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (packed, scales, zeros, n, group_size=64, bits=4))]
pub fn turbo_quant_decode(
    py: Python<'_>,
    packed: Vec<u8>,
    scales: Vec<f32>,
    zeros: Vec<f32>,
    n: usize,
    group_size: usize,
    bits: u8,
) -> PyResult<PyObject> {
    validate_dimensions(
        n,
        n,
        group_size,
        bits,
        scales.len(),
        zeros.len(),
        packed.len(),
    )?;
    validate_finite("scales", &scales)?;
    validate_finite("zeros", &zeros)?;
    if scales.iter().any(|scale| *scale <= 0.0) {
        return Err(PyValueError::new_err("scales must be > 0"));
    }
    let q = QuantizedTensor {
        shape: vec![n],
        packed,
        scales,
        zeros,
        bits,
        group_size,
    };
    let mut buf = vec![0f32; n];
    contain_panic(|| {
        q.decode_uniform(&mut buf);
        Ok(())
    })?;
    let lst = PyList::new_bound(py, buf.iter().copied());
    Ok(lst.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyValueError;
    use pyo3::types::{PyDict, PyList};

    #[test]
    fn invalid_bits_are_rejected() {
        assert!(validate_bits(1).is_err());
        assert!(validate_bits(5).is_err());
    }

    #[test]
    fn invalid_decode_dimensions_are_rejected() {
        assert!(validate_dimensions(8, 8, 0, 4, 0, 0, 4).is_err());
        assert!(validate_dimensions(8, 8, 3, 4, 3, 3, 4).is_ok());
        assert!(validate_dimensions(8, 8, 4, 4, 1, 2, 4).is_err());
        assert!(validate_dimensions(8, 8, 4, 4, 2, 2, 3).is_err());
    }

    #[test]
    fn rust_panics_are_contained_as_python_errors() {
        let error = contain_panic(|| -> PyResult<()> { panic!("core invariant") })
            .expect_err("panic must become a Python exception");
        assert!(error.to_string().contains("core invariant"));
    }

    #[test]
    fn exported_decode_rejects_non_positive_scales_as_value_error() {
        Python::with_gil(|py| {
            for scale in [0.0, -0.25] {
                let error = turbo_quant_decode(py, vec![0], vec![scale], vec![0.0], 2, 2, 4)
                    .expect_err("non-positive scale must be rejected");
                assert!(error.is_instance_of::<PyValueError>(py));
            }
        });
    }

    #[test]
    fn exported_encode_decode_round_trip_preserves_metadata_and_values() {
        Python::with_gil(|py| {
            let encoded =
                turbo_quant_encode(py, vec![-1.0, 0.0, 1.0], 2, 4).expect("encode succeeds");
            let encoded = encoded.bind(py).downcast::<PyDict>().expect("dict payload");
            let packed = encoded
                .get_item("packed")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            let scales = encoded
                .get_item("scales")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            let zeros = encoded
                .get_item("zeros")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(
                encoded
                    .get_item("bits")
                    .unwrap()
                    .unwrap()
                    .extract::<u8>()
                    .unwrap(),
                4
            );
            assert_eq!(
                encoded
                    .get_item("group_size")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                2
            );

            let decoded =
                turbo_quant_decode(py, packed, scales, zeros, 3, 2, 4).expect("decode succeeds");
            let values = decoded
                .bind(py)
                .downcast::<PyList>()
                .unwrap()
                .extract::<Vec<f32>>()
                .unwrap();
            assert_eq!(values.len(), 3);
            for (actual, expected) in values.iter().zip([-1.0, 0.0, 1.0]) {
                assert!((actual - expected).abs() <= 0.14);
            }
        });
    }

    #[test]
    fn exported_validation_errors_use_value_error() {
        Python::with_gil(|py| {
            let encode_error = turbo_quant_encode(py, vec![], 64, 4).unwrap_err();
            assert!(encode_error.is_instance_of::<PyValueError>(py));
            let label_error = turbo_quant_label_for_bits(1).unwrap_err();
            assert!(label_error.is_instance_of::<PyValueError>(py));
        });
    }
}
