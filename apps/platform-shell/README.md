# platform-shell

Thin **Win/Linux** native host for the federated synthetic monolith (ADR-006).

| OS | Role |
|----|------|
| Windows / Linux | This Tauri 2 shell → embeds/navigates hub UI (`http://127.0.0.1:8090`) |
| macOS | Prefer existing oMLX.app — this shell is optional parity |

## Dev

Prereq: bench-cockpit listening on `:8090`, `bun` + Rust toolchain.

```bash
# from apps/platform-shell
bun install
bunx tauri dev
```

Release:

```bash
bunx tauri build
```

Windows artifacts: `src-tauri/target/release/platform-shell.exe` and `src-tauri/target/release/bundle/msi/Phenotype Shell_0.1.0_x64_en-US.msi`.

## Scope

- **In:** window chrome, deep-links to cockpit / Langfuse panel, future tray
- **Out:** inference kernels, Harbor, Metal — those stay plugins / Mac spoke

## Pull Mac V5 (when Tailscale Mac is up)

```bash
ssh -F ~/.ssh/config.pheno kooshas-laptop \
  'ls ~/CodeProjects/Phenotype/pheno-harness/bench/results/stock-vs-ours/run-v5*'
# then scp into apps/bench-cockpit/data/ and set BENCH_DATA
```

See `scripts/pull-mac-v5.sh`.
