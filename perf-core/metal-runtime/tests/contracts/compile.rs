//! §3 — Bounded compiler contracts.
//!
//! Covers the happy path (budget respected), the shader-byte budget
//! violation, the millisecond budget violation, and the error-message
//! contract when both budget dimensions are violated simultaneously.

use metal_runtime::{BoundedCompiler, CompileBudget, CompileError, GpuFamily};

use super::common::{identity_fp, two_op_plan};

#[test]
fn compile_returns_ok_with_budget_respected() {
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 100,
        max_shader_bytes: 1024,
    });
    let fp = identity_fp(GpuFamily::Software);
    let plan = two_op_plan();
    let res = compiler.compile(&plan, &fp);
    assert!(res.is_ok(), "compile should succeed: {:?}", res.err());
    let cp = res.unwrap();
    assert!(cp.shader_source.len() <= 1024);
}

#[test]
fn compile_budget_exceeded_when_shader_source_exceeds_max_shader_bytes() {
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 10_000,
        max_shader_bytes: 8,
    });
    let fp = identity_fp(GpuFamily::Software);
    let plan = two_op_plan();
    let err = compiler.compile(&plan, &fp).expect_err("must exceed budget");
    match err {
        CompileError::BudgetExceeded {
            max_shader_bytes,
            shader_bytes,
            ..
        } => {
            assert!(shader_bytes > max_shader_bytes);
            assert_eq!(max_shader_bytes, 8);
        }
        other => panic!("expected BudgetExceeded, got {:?}", other),
    }
}

#[test]
fn compile_budget_exceeded_when_compile_ms_exceeds_max_ms() {
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 0,
        max_shader_bytes: 1024 * 1024,
    });
    let fp = identity_fp(GpuFamily::Software);
    let plan = two_op_plan();
    let err = compiler.compile(&plan, &fp).expect_err("must exceed budget");
    match err {
        CompileError::BudgetExceeded {
            max_ms,
            compile_ms,
            ..
        } => {
            assert!(compile_ms > max_ms);
            assert_eq!(max_ms, 0);
        }
        other => panic!("expected BudgetExceeded, got {:?}", other),
    }
}

#[test]
fn compile_error_message_includes_both_budget_dimensions_when_both_violated() {
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 0,
        max_shader_bytes: 4,
    });
    let fp = identity_fp(GpuFamily::Software);
    let plan = two_op_plan();
    let err = compiler.compile(&plan, &fp).expect_err("must exceed both");
    let msg = err.to_string();
    assert!(msg.contains("ms"), "error msg must mention ms: {}", msg);
    assert!(msg.contains("bytes"), "error msg must mention bytes: {}", msg);
}
