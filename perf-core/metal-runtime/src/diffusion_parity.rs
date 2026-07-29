//! Host-side parity oracle for diffusion Metal outputs.

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DiffusionParityError {
    #[error("{what} length mismatch: expected {expected}, got {got}")]
    Length {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("{what} mismatch at index {index}: expected {expected}, got {got}")]
    Value {
        what: &'static str,
        index: usize,
        expected: String,
        got: String,
    },
}

pub fn compare_u32(
    what: &'static str,
    expected: &[u32],
    actual: &[u32],
) -> Result<(), DiffusionParityError> {
    if expected.len() != actual.len() {
        return Err(DiffusionParityError::Length {
            what,
            expected: expected.len(),
            got: actual.len(),
        });
    }
    if let Some((index, (expected, got))) = expected
        .iter()
        .zip(actual)
        .enumerate()
        .find(|(_, (expected, got))| expected != got)
    {
        return Err(DiffusionParityError::Value {
            what,
            index,
            expected: expected.to_string(),
            got: got.to_string(),
        });
    }
    Ok(())
}

pub fn compare_u8(
    what: &'static str,
    expected: &[u8],
    actual: &[u8],
) -> Result<(), DiffusionParityError> {
    compare_u32(
        what,
        &expected
            .iter()
            .map(|value| *value as u32)
            .collect::<Vec<_>>(),
        &actual.iter().map(|value| *value as u32).collect::<Vec<_>>(),
    )
}

pub fn compare_f32(
    what: &'static str,
    expected: &[f32],
    actual: &[f32],
    tolerance: f32,
) -> Result<(), DiffusionParityError> {
    if expected.len() != actual.len() {
        return Err(DiffusionParityError::Length {
            what,
            expected: expected.len(),
            got: actual.len(),
        });
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(DiffusionParityError::Value {
            what,
            index: 0,
            expected: "finite non-negative tolerance".into(),
            got: tolerance.to_string(),
        });
    }
    if let Some((index, (expected, got))) = expected
        .iter()
        .zip(actual)
        .enumerate()
        .find(|(_, (expected, got))| (*expected - *got).abs() > tolerance)
    {
        return Err(DiffusionParityError::Value {
            what,
            index,
            expected: expected.to_string(),
            got: got.to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compares_native_shapes_and_values() {
        compare_u32("positions", &[0, 3], &[0, 3]).unwrap();
        compare_u8("mask", &[1, 0], &[1, 0]).unwrap();
        compare_f32("momentum", &[0.1, 0.3], &[0.100001, 0.3], 1e-4).unwrap();
    }

    #[test]
    fn rejects_shape_and_value_drift() {
        assert!(matches!(
            compare_u32("positions", &[0], &[1]),
            Err(DiffusionParityError::Value { .. })
        ));
        assert!(matches!(
            compare_f32("momentum", &[0.1], &[], 1e-4),
            Err(DiffusionParityError::Length { .. })
        ));
    }
}
