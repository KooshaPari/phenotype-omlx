# ADR-006: Federated synthetic monolith (omlx ⊕ hwLedger)

## Status
Proposed → Accepted for scaffolding (2026-07-22)

## Context
`bench-cockpit` README and ADR-001 already name a unified AI·ML platform that
**absorbs omlx + hwLedger**. ADR-035A keeps hwLedger as a **federated service**
with math extracted to `pheno-capacity`. We need a single **synthetic monolith**
product surface without merging git histories into one blob or requiring Metal
on Windows/Linux.

## Decision
Ship a **micro-federated synthetic monolith**:

| Seam | Mechanism | Owns |
|------|-----------|------|
| **Embed** | Cargo path/git dep in `perf-core` | `pheno-capacity` VRAM/fit/Chinchilla math; `fleet-proto` |
| **Plugin** | NanoVM `plugin.toml` + PhenoPlugins traits | Engines (mlx-metal Mac-only; vllm/trt/sglang/llamacpp Win/Linux); `hwledger-probe` heartbeats |
| **Compose** | PhenoCompose manifest v0 | Multi-process stacks (cockpit + Langfuse + portage + probe) |
| **Shell** | Thin native hosts | Mac: oMLX.app; Win/Linux: Tauri/`platform-shell` over same HTTP+FFI core |
| **Spoke service** | hwLedger repo (unchanged) | Inventory persistence, fleet API, OS GUIs |

**Parity rule:** one embedded Rust core; OS/accel specifics are plugins or thin
shells. Metal/MLX never required on Win/Linux CI. Native optimalities:
CUDA/TensorRT/WSL2 on Windows, CUDA/ROCm on Linux, Metal on Darwin.

## Consequences
- Do **not** submodule-dump hwLedger into omlx.
- Restore capacity as embedded crate under `perf-core/pheno-capacity` (pheno-capacity GH deleted).
- Cockpit + Portage + Langfuse remain the eval iteration surface (pairings.yaml).
- Control-plane `EnginePort` binds `hwledger` + `omlx` over NATS subjects.

## Related
- ADR-001, ADR-002, ADR-035A
- `platform/federation/composition.v0.yaml`
- `platform/federation/parity-matrix.md`
- `python/omlx_research/nanovm/plugins/hwledger-probe/plugin.toml`
- PhenoCompose `docs/composition-manifest-v0.md`, PhenoPlugins
