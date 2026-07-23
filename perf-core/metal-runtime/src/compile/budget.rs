//! [`CompileBudget`] — wall-clock and shader-byte budgets for
//! [`super::BoundedCompiler::compile`].

/// Wall-clock and shader-byte budgets for [`super::BoundedCompiler::compile`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompileBudget {
    /// Maximum wall-clock compile time in milliseconds. `0` makes the
    /// budget impossible to satisfy for any non-trivial plan (used by
    /// tests to drive the over-budget error path).
    pub max_ms: u64,
    /// Maximum total shader-source bytes the compiler may emit.
    pub max_shader_bytes: usize,
}

impl CompileBudget {
    /// Generous default used when the caller has no specific budget in
    /// mind. Mirrors the policy used by the reference interpreter's
    /// compile-time smoke tests.
    pub const DEFAULT: Self = Self {
        max_ms: 5_000,
        max_shader_bytes: 1 << 20, // 1 MiB
    };
}
