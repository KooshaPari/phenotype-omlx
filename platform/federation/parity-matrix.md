# Federated synthetic monolith — OS / accel parity

One **embedded** core (`perf-core` + `pheno-capacity` + `fleet-proto`).
**Plugins** carry engines and probes. **Shells** are thin natives.

| Capability | Windows | Linux | macOS |
|------------|---------|-------|-------|
| Shell | Tauri / `apps/platform-shell` (planned) | same | oMLX.app + optional SwiftUI (hwLedger) |
| Default backend | `llamacpp` → `vllm`/`tensorrt`/`sglang` | same | `mlx-metal` → `llamacpp` |
| Accel | CUDA (WSL2 / native), CPU | CUDA, ROCm, CPU | Metal, CPU |
| Capacity math | **embed** `pheno-capacity` | embed | embed |
| Fleet inventory | **plugin** `hwledger-probe` → NATS | same | same + hwLedger sidecar |
| Eval UI | bench-cockpit `:8090` | same | same |
| Harbor/Portage | **Podman** (WSL2 machine OK) | **Podman** | **apple-container** |
| Hub messaging | **Podman** compose (NATS) — never Docker Engine | same | n/a (spoke) |
| Observability | Langfuse | Langfuse | Langfuse |

## Native optimalities

- **Windows:** CUDA/TRT/vLLM in WSL2 Linux-FS; no `/mnt/d` CUDA binaries; containers via **Podman** only.
- **Linux:** CUDA/ROCm + **Podman** sandboxes.
- **macOS:** MLX/Metal first; `HARBOR_ENV=apple-container`.

## Forbidden

- Metal required for hub CI.
- Merging hwLedger git into omlx (federated service stays separate).
- **Docker Engine / Docker Desktop** on this fleet — use **Podman** (Win/Linux) or **apple-container** (Mac).
