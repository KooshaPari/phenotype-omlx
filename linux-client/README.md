# phenotype-omlx Linux Client

Linux CLI / desktop client. Provides full backend access to the
phenotype-omlx stack without the macOS OMLX app bundle.

## Status

Stub. The Linux client uses the same `cli/bin/omlx-research` bash launcher
as the macOS setup, with the difference that on Linux we talk directly to
the bundled CPython (no `/Applications/oMLX.app` framework path needed —
system or venv Python is used).

## Quick start

```bash
git clone https://github.com/KooshaPari/phenotype-omlx.git
cd phenotype-omlx
./cli/bin/omlx-research doctor
./cli/bin/omlx-research status
./cli/bin/omlx-research web --port 8080
```

## MLX on Linux

MLX is currently Apple-Silicon-only. On Linux, the stack automatically
falls back to PyTorch + CUDA / ROCm via the `tensorrt` or `llamacpp`
backends. See `python/omlx_research/backends/` for the adapter layer.

## Desktop GUI

The Linux desktop GUI ships as a Tauri 2.0 application that talks to the
local `omlx-research web` admin over a Unix socket. Build with:

```bash
cargo install tauri-cli --version "^2.0"
cargo tauri build --config gui/desktop/linux/tauri.conf.json
```

## Architecture parity

Same Python surface, same Rust `perf-core` crates, same `fleet-proto` JSON-RPC
envelope as macOS / Windows. The Linux-specific differences are:

- No MLX framework injection (system Python or venv only)
- Native ROCm / CUDA backends in addition to Metal
- GTK4 / Tauri for the desktop shell
