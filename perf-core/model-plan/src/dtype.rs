//! Runtime dtype for tensor storage and kernel specialization.
//!
//! [`DType`] is distinct from [`crate::precision::Precision`]: a tensor can
//! be stored as `F16` while the operator policy says "treat values as
//! `Bf16`" (mixed-precision pipelines do this routinely). Keeping the two
//! enums separate prevents silent coercion when a kernel selects against
//! the storage dtype.

use serde::{Deserialize, Serialize};

/// Runtime dtype the kernel should read and write from memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    /// 32-bit IEEE-754 float.
    F32,
    /// 16-bit IEEE-754 float.
    F16,
    /// 16-bit brain float. Renamed explicitly because
    /// `rename_all = "snake_case"` would otherwise split `BF16` as
    /// `b_f16`.
    #[serde(rename = "bf16")]
    BF16,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 8-bit integer.
    U8,
}

impl DType {
    /// Lowercase string identifier; matches the JSON representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I8 => "i8",
            DType::U8 => "u8",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_round_trips_via_json() {
        let variants = [DType::F32, DType::F16, DType::BF16, DType::I8, DType::U8];
        for d in variants {
            let s = serde_json::to_string(&d).unwrap();
            let back: DType = serde_json::from_str(&s).unwrap();
            assert_eq!(back, d);
        }
    }

    #[test]
    fn as_str_matches_json() {
        for d in [DType::F32, DType::F16, DType::BF16, DType::I8, DType::U8] {
            assert_eq!(
                serde_json::to_string(&d).unwrap(),
                format!("\"{}\"", d.as_str())
            );
        }
    }
}