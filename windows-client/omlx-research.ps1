#!/usr/bin/env pwsh
# omlx-research PowerShell entry-point. Stub for the Windows client.
#
# Until the full Tauri GUI is ready, this launcher sources the same env
# file as the bash version and runs the Python `omlx_research` subcommand
# CLI. WSL2 users can call `./cli/bin/omlx-research` directly.

$ErrorActionPreference = "Stop"

$PHENOTYPE_OMLX_HOME = if ($env:PHENOTYPE_OMLX_HOME) {
    $env:PHENOTYPE_OMLX_HOME
} else {
    "$HOME/repos/phenotype-omlx"
}

$env:PYTHONPATH = "$PHENOTYPE_OMLX_HOME/python;$env:PYTHONPATH"

& python -m omlx_research.cli @args
exit $LASTEXITCODE
