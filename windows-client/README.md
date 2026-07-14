# phenotype-omlx Windows Client

Native Windows client that talks to a remote phenotype-omlx server
(macOS / Linux) over the fleet protocol. Runs MLX inference via a
PowerShell / Windows Terminal launcher.

## Status

Stub. Production Windows client will ship once the fleet protocol
is finalized (see `perf-core/fleet-proto/src/lib.rs`).

## Planned surface

- `omlx-research.ps1`        — PowerShell entry-point mirroring the bash launcher
- `omlx-research-gui.exe`    — Tauri-based desktop GUI (cross-platform with macOS)
- WSL2 fallback              — runs the Linux client under WSL2 for MLX access

## Build prerequisites

- Windows 10/11 22H2 or later
- PowerShell 7.4+
- Rust toolchain 1.78+ (for the `omlx-research.exe` proxy)
- Tauri 2.0 (for the desktop GUI)

## Quick start (WSL2 path today)

```powershell
wsl --install -d Ubuntu
# inside WSL2:
git clone https://github.com/KooshaPari/phenotype-omlx.git
cd phenotype-omlx
./cli/bin/omlx-research doctor
```

## Native path (planned)

```powershell
# Native Windows binary
.\omlx-research.exe status
.\omlx-research.exe inference --prompt "Hello"
.\omlx-research.exe web --port 8081
```

## Architecture parity

The Windows client uses the same `omlx_research.*` Python surface, the same
`perf-core` Rust crates, and the same `fleet-proto` JSON-RPC envelope as the
macOS / Linux clients. The only Windows-specific code is the launcher shell
and the Tauri window — all backend logic is identical.
