# Pull Mac V5 EvaluationReport when Tailscale Mac answers SSH.
# Usage: powershell -File scripts\pull-mac-v5.ps1
$ErrorActionPreference = 'Stop'
$cfg = Join-Path $env:USERPROFILE '.ssh\config.pheno'
$out = 'D:\koosh\phenotype-omlx\apps\bench-cockpit\data'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$remoteDir = '~/CodeProjects/Phenotype/pheno-harness/bench/results/stock-vs-ours'
$remote = ssh -F $cfg -o ConnectTimeout=15 kooshas-laptop "ls -1 $remoteDir/run-v5*.json 2>/dev/null | head -1"
if (-not $remote) { throw "no run-v5*.json on Mac" }
$base = Split-Path $remote -Leaf
scp -F $cfg -o ConnectTimeout=15 "kooshas-laptop:${remote}" (Join-Path $out $base)
Write-Host "wrote $(Join-Path $out $base)"
Write-Host "BENCH_DATA=$(Join-Path $out $base)"
