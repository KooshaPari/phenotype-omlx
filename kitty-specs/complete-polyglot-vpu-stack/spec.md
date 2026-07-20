# Specification: Complete the phenotype-omlx polyglot vPU performance stack

## Problem Statement
The resumed TurboQuant and vPU expansion contains incomplete evaluation, Mojo, Nim, Go, SIMD, model-evaluation, and stock harness integration work. The checked-in evaluation tests currently fail to compile.

## Target Users
Local ML inference and multi-device orchestration developers using phenotype-omlx and hwLedger-compatible runtimes.

## Functional Requirements
- **FR-1**: Complete deterministic MMLU, GPQA, terminal-bench, and perplexity evaluation primitives and loaders.
- **FR-2**: Add a tested AArch64 NEON min/max reduction path with a portable fallback.
- **FR-3**: Compile and smoke-test the Mojo TurboQuant kernel with the installed SDK.
- **FR-4**: Validate Nim and Go bindings end-to-end against the C ABI.
- **FR-5**: Run the Qwen3.5 NIAH and locally available quality evaluations, recording model-architecture limitations.
- **FR-6**: Reuse stock helios-cli, Codex, or ForgeCode wrappers without creating a custom ForgeCode-style loop.
- **FR-7**: Validate the existing vPU dashboard serving workflow.

## Non-Functional Requirements
Use measured performance, deterministic tests, pure FFI payloads, safe memory ownership, nightly Rust where useful, and global toolchains. Preserve stable fallbacks where package compatibility requires them.

## Constraints & Dependencies
macOS arm64 development host; NVIDIA devices may be unavailable locally. Qwen3.5-0.8B-OptiQ-4bit uses linear attention, so standard KV-cache compression metrics are not applicable. Do not fork ForgeCode or implement a ForgeCode-style agent loop.

## Acceptance Criteria
All available lint, typecheck, focused polyglot tests, Rust workspace tests, and release builds pass; unavailable hardware evaluations are reported explicitly rather than fabricated; changes are reviewed for secrets before the final commit.
