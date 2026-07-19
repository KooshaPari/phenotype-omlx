//! Status codes returned across the native ABI boundary.
//!
//! Codes are stable integer values. New variants must be appended at the end
//! so that polyglot consumers (C, Zig, Mojo, Nim, Go) continue to interpret
//! older values correctly. Never reorder or repurpose an existing code.
//!
//! `Ok == 0` so callers can use a single bit to check "did it work?".

use core::convert::TryFrom;
use core::fmt;

/// All status codes the ABI v1 contract may return.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Status {
    Ok = 0,
    ErrNullArg = 1,
    ErrInvalidBits = 2,
    ErrInvalidGroupSize = 3,
    ErrNonFiniteInput = 4,
    ErrOverflow = 5,
    ErrAllocation = 6,
    ErrVersionMismatch = 7,
    ErrBackend = 8,
}

impl Status {
    /// One-line human description for logs / error messages.
    ///
    /// The string is non-empty for every variant — contract tests rely on
    /// this invariant to detect accidentally-empty `match` arms.
    pub fn description(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::ErrNullArg => "null pointer or zero length where one is required",
            Status::ErrInvalidBits => "bits must be in 2..=4",
            Status::ErrInvalidGroupSize => "group_size must be > 0",
            Status::ErrNonFiniteInput => "input contains NaN or +-Inf",
            Status::ErrOverflow => "size computation would overflow usize",
            Status::ErrAllocation => "backend failed to allocate output storage",
            Status::ErrVersionMismatch => "ABI version mismatch between host and guest",
            Status::ErrBackend => "backend reported an unspecified error",
        }
    }
}

impl From<Status> for i32 {
    #[inline]
    fn from(s: Status) -> i32 {
        s as i32
    }
}

impl TryFrom<i32> for Status {
    type Error = i32;
    fn try_from(v: i32) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Status::Ok),
            1 => Ok(Status::ErrNullArg),
            2 => Ok(Status::ErrInvalidBits),
            3 => Ok(Status::ErrInvalidGroupSize),
            4 => Ok(Status::ErrNonFiniteInput),
            5 => Ok(Status::ErrOverflow),
            6 => Ok(Status::ErrAllocation),
            7 => Ok(Status::ErrVersionMismatch),
            8 => Ok(Status::ErrBackend),
            other => Err(other),
        }
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.description())
    }
}