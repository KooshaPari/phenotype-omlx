# Portage / Harbor on Windows — Podman only (never Docker Engine)

## Required env

```text
PATH=C:\Users\koosh\bin;...   # podman-compose.cmd shim first
PODMAN_COMPOSE_PROVIDER=C:\Users\koosh\bin\podman-compose.cmd
COMPOSE_PROVIDER=C:\Users\koosh\bin\podman-compose.cmd
PODMAN_COMPOSE_WARNING_LOGS=false
TEMP=D:\koosh\tmp\harbor-temp
TMP=D:\koosh\tmp\harbor-temp
# Scrub Docker Desktop from PATH for the session
```

Shim content:

```bat
@echo off
"C:\Users\koosh\portage\.venv\Scripts\python.exe" -m podman_compose %*
```

Install once: `cd C:\Users\koosh\portage && uv pip install podman-compose`

## C: disk pressure / Errno 28 (before every harbor run)

Harbor and Python tempfile default to `%TEMP%` / `%TMP%` on **C:**. When C: fills,
runs fail with **`[Errno 28] No space left on device`**. Keep PATH scrub + compose
shim, and **redirect TEMP/TMP off C:** before any `uv run harbor …`:

```powershell
New-Item -ItemType Directory -Force -Path 'D:\koosh\tmp\harbor-temp' | Out-Null
$env:TEMP = 'D:\koosh\tmp\harbor-temp'
$env:TMP  = 'D:\koosh\tmp\harbor-temp'
```

Do **not** kill TB containers (`chess-best-move` / `fix-git` / `pheno-nats`) to free
space — use the D: TEMP redirect instead. Never Docker Engine.

## Concurrent containers

Do **not** run Harbor TB cells while unrelated Podman workloads hold the machine
busy (e.g. `helios_bench__*` from helios-cli). Concurrent compose/exec has correlated
with `Failed to start tmux session`, `tmux capture-pane` timeouts, and socket refused
(exit 125). Stop foreign TB containers before launch: `podman ps` then `podman stop …`.

If Harbor dies with `podman … exit 125` / `dial tcp 127.0.0.1:…. connectex: … refused`
mid-compose or mid-`podman cp`, the machine dropped. Cycle before retry:

```powershell
podman machine stop; Start-Sleep 8; podman machine start
# if "already running" but still refused: stop again, wait longer, start
podman start pheno-nats   # or recreate: nats:2.10-alpine -js -m 8222
```

Confirm `podman ps` works before relaunching harbor. SGLang (WSL Ubuntu) is separate
from the Podman machine — restart with a **home-disk copy** of the start script
(`cp …/wsl_start_sglang_qwen.sh ~/tmp/…` then `setsid nohup`) if `:30000` is down.
Current knobs: `--mem-fraction-static 0.92` + `--context-length 8192` (0.85/0.88 can
fail with "increase mem-fraction" after weight load). If WSL itself hangs:
`wsl --shutdown` then restart Ubuntu + `podman machine start`.

## Smoke

```text
uv run harbor run -p examples/tasks/hello-world -a oracle -e docker --ek container_runtime=podman -n 1 -k 1 -o <out>
```

Proven 2026-07-22: reward **1.0** → `platform/federation/out/portage-hello5`

## Next TB cell (terminal-bench-2 oracle × Podman)

Dataset confirmed 2026-07-22: `uv run harbor datasets list --legacy` → **terminal-bench @ 2.0** (89 tasks; git source laude-institute/terminal-bench-2). Hub package slug: `terminal-bench/terminal-bench-2`. Never Docker Engine.

PowerShell session (TEMP→D: + PATH scrub + compose shim), from `C:\Users\koosh\portage`:

```powershell
New-Item -ItemType Directory -Force -Path 'D:\koosh\tmp\harbor-temp' | Out-Null
$env:TEMP = 'D:\koosh\tmp\harbor-temp'
$env:TMP  = 'D:\koosh\tmp\harbor-temp'
$env:PATH = (@('C:\Users\koosh\bin') + ($env:PATH -split ';' | Where-Object { $_ -and $_ -notmatch '(?i)Docker' })) -join ';'
$env:PODMAN_COMPOSE_PROVIDER = 'C:\Users\koosh\bin\podman-compose.cmd'
$env:COMPOSE_PROVIDER = 'C:\Users\koosh\bin\podman-compose.cmd'
$env:PODMAN_COMPOSE_WARNING_LOGS = 'false'

# Prefer light task by full slug (-i); bare `-l 1` picks first hub task (make-mips-interpreter).
# --verifier-timeout-multiplier 3 for heavy verifiers (mips hit effective 300s).
uv run harbor run -d terminal-bench/terminal-bench-2 -a oracle -e docker --ek container_runtime=podman -n 1 -k 1 -l 1 -i terminal-bench/fix-git --verifier-timeout-multiplier 3 -o D:\koosh\phenotype-omlx\platform\federation\out\portage-tb2-oracle2
```

Proven 2026-07-22 tick#5: **fix-git** reward **1.0** (~4m25s) → `out/portage-tb2-oracle2`. Legacy: `-d terminal-bench@2.0`. `-i` / `-t` filters need full names (`terminal-bench/<task>`), not bare ids.

## Next real non-oracle agent cell (TB2 × Podman)

`uv run harbor run --help` agents include **`terminus` / `terminus-1` / `terminus-2`** and **`openhands` / `openhands-sdk`** (also `nop`, `oracle`, …). **Do not invent** agent names outside that list.

Which `-a` need `-m`:

- **Needs `-m`:** any LLM agent — e.g. `terminus`, `terminus-2`, `openhands` (LiteLLM model string).
- **No `-m`:** `oracle`, `nop` (path proofs only; not a model cell).

Session hygiene (same as oracle/nop): **TEMP/TMP → `D:\koosh\tmp\harbor-temp`**, PATH scrub (no Docker Desktop), compose shim, **never Docker Engine**. Do not start a long agent run from a docs tick.

**Agent eval policy (2026-07-23):** **local models only**. Prefer **SGLang Qwen3.5-9B** `@ :30000` over llama.cpp **0.8b** `@ :8000` (0.8b times out on TB2). Omniroute (`:20128`) is **out of scope**.

**Windows launch hygiene (required):**

- `PYTHONUTF8=1` + `PYTHONIOENCODING=utf-8` — Rich Live spinner crashes with `UnicodeEncodeError` on cp1252 when stdout is redirected (harbor exit -1).
- Prefer a `.cmd` wrapper writing logs to files (see `out/portage-tb2-terminus-local3d/run.cmd`).
- TEMP/TMP → `D:\koosh\tmp\harbor-temp`; PATH scrub; compose shim; **never Docker Engine**.

Canonical local recipe (SGLang 9B — proven tick#32 openssl **reward 1.0**; tick#34 adds **model_info**):

```bat
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
REM TEMP/PATH/compose as above; from C:\Users\koosh\portage
REM enable_thinking=false via base64json llm_call_kwargs; max_turns=12
REM model_info REQUIRED — without it LiteLLM falls back to 1M ctx → summarization death spiral
REM proactive_summarization_threshold must be << context (default 8000 fires every turn on 8192)
harbor.exe run -d terminal-bench/terminal-bench-2 -a terminus-2 -m openai//home/kooshapari/models/Qwen3.5-9B --ak api_base=http://127.0.0.1:30000/v1 --ak model_info=base64json:eyJtYXhfaW5wdXRfdG9rZW5zIjo4MTkyLCJtYXhfb3V0cHV0X3Rva2VucyI6MjA0OCwiaW5wdXRfY29zdF9wZXJfdG9rZW4iOjAsIm91dHB1dF9jb3N0X3Blcl90b2tlbiI6MH0= --ak proactive_summarization_threshold=1500 --ak llm_call_kwargs=base64json:eyJleHRyYV9ib2R5Ijp7ImNoYXRfdGVtcGxhdGVfa3dhcmdzIjp7ImVuYWJsZV90aGlua2luZyI6ZmFsc2V9fX0= --ak llm_kwargs={\"max_tokens\":2048} --ak max_turns=12 -e docker --ek container_runtime=podman -n 1 -k 1 -l 1 -i terminal-bench/fix-git --verifier-timeout-multiplier 3 --agent-setup-timeout-multiplier 3 -o D:\koosh\phenotype-omlx\platform\federation\out\portage-tb2-terminus-local5
```

Decoded `model_info`: `{"max_input_tokens":8192,"max_output_tokens":2048,"input_cost_per_token":0,"output_cost_per_token":0}` (must match SGLang `--context-length`).

If `:30000` is down: `wsl … bash scripts/restart_sglang_once.sh` (or `.\scripts\wsl_restart_sglang.ps1`). Launch knobs in `scripts/wsl_start_sglang_qwen.sh`: `--mem-fraction-static 0.88` + `--context-length 8192` (0.75 → "increase mem-fraction"; too-low free VRAM → `zeros: Dimension size must be non-negative`).

**Fallback 0.8b** (`:8000` / `local/qwen35-08b`) — expect `AgentTimeoutError` on TB2 terminus-2 (local1/local2).

**Historical (out of scope):** Omniroute `auto/best-coding` @ `:20128`. Do not start new Omniroute cells.

## WinError 2 on image OS validation

Log line `Skipping image OS validation ... [WinError 2]` meant Harbor called
hardcoded `docker inspect` while only Podman is installed (intentional — no
Docker Engine). Fixed in `portage` `DockerEnvironment._validate_image_os` to
use `self._container_binary` (podman). That skip is benign for reward; recent
tick failure `portage-hello-tick-043336` was a separate **verifier 120s timeout**
during compose exec (image `localhost/hello-world__*__env_main` already exists).
