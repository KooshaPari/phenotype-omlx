# Windows launch script for Bench Cockpit (bench-cockpit on :8090) - hardened.
# Mirrors scripts/start-dev.sh for Windows hosts with these additions:
#   - reuses an already-running :8090 instance (no duplicate bind)
#   - builds once to .run\cockpit.exe (real PID, fast start, killable)
#   - waits for /api/health before opening the browser
#   - writes .run\cockpit.pid for scripts\stop-dev-windows.ps1
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\start-dev-windows.ps1
#   # with overrides:
#   $env:BENCH_DATA='C:\path\results.json'; $env:BENCH_EXTRA_DATA='C:\path\matrix.json'
#   powershell -ExecutionPolicy Bypass -File scripts\start-dev-windows.ps1
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$RunDir = Join-Path $Root '.run'
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

# --- resolve Go toolchain (Go is not on PATH in this environment) ---
$goCandidates = @(
    'C:\Program Files\Go\bin\go.exe',
    'C:\Users\koosh\go1.25\bin\go.exe',
    (Get-Command go -ErrorAction SilentlyContinue).Source
) | Where-Object { $_ -and (Test-Path $_) }
if (-not $goCandidates) { throw 'Go toolchain not found; install Go or set PATH.' }
$goExe = $goCandidates | Select-Object -First 1
Write-Host "==> go: $goExe"

# --- reuse a running instance instead of double-binding :8090 ---
$listener = Get-NetTCPConnection -LocalPort 8090 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    Write-Host "==> cockpit already serving :8090 (PID $($listener.OwningProcess)) - opening browser"
    Start-Process 'http://localhost:8090'
    exit 0
}

# --- data path resolution (same precedence as the Go server's resolveDataPath) ---
$canonicalV5Native = 'C:\Users\koosh\pheno-harness\bench\results\stock-vs-ours\run-v5-qwen35-08b.json'
$canonicalV5Contract = 'C:\Users\koosh\pheno-harness\bench\results\stock-vs-ours\run-v5-qwen35-08b-contract.json'
$canonicalMinimax = 'C:\Users\koosh\pheno-harness\bench\results\minimax-m3-full\matrix.json'
$localV5Native = Join-Path $Root 'data\run-v5-qwen35-08b.json'
$localV5Contract = Join-Path $Root 'data\run-v5-qwen35-08b-contract.json'
$smoke = Join-Path $Root 'fixtures\smoke_results.json'

$dataPath = $env:BENCH_DATA
foreach ($cand in @($canonicalV5Native, $canonicalV5Contract, $localV5Native, $localV5Contract, $smoke)) {
    if (-not $dataPath -and (Test-Path $cand)) { $dataPath = $cand }
}
if (-not $dataPath) { throw 'no results file found (set BENCH_DATA, or use canonical V5 / data\ / fixtures\smoke_results.json)' }
$extraPath = $env:BENCH_EXTRA_DATA
if (-not $extraPath -and (Test-Path $canonicalMinimax)) { $extraPath = $canonicalMinimax }

# --- build once, run the binary (real PID, no `go run` child indirection) ---
$serverDir = Join-Path $Root 'server'
$exe = Join-Path $RunDir 'cockpit.exe'
Write-Host '==> building cockpit (go build)'
Push-Location $serverDir
& $goExe build -o $exe .
$buildRc = $LASTEXITCODE
Pop-Location
if ($buildRc -ne 0) { throw "go build failed (rc=$buildRc)" }

$args = @('-data', $dataPath, '-port', '8090')
if ($extraPath) { $args += @('-extra', $extraPath) }
$logOut = Join-Path $RunDir 'cockpit.out.log'
$logErr = Join-Path $RunDir 'cockpit.err.log'
$p = Start-Process -FilePath $exe -ArgumentList $args -WorkingDirectory $serverDir `
    -RedirectStandardOutput $logOut -RedirectStandardError $logErr -WindowStyle Hidden -PassThru
$p.Id | Out-File (Join-Path $RunDir 'cockpit.pid') -Encoding ascii

# --- wait for readiness before opening the browser ---
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $h = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8090/api/health' -TimeoutSec 2
        if ($h.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
    if ($p.HasExited) { throw "cockpit exited early (rc=$($p.ExitCode)); see $logErr" }
}
if (-not $ready) { throw "cockpit did not become ready on :8090 within 30s; see $logErr" }

Write-Host "==> data : $dataPath"
Write-Host "==> extra: $extraPath"
Write-Host "==> pid  : $($p.Id) (stop via scripts\stop-dev-windows.ps1)"
Write-Host "==> logs : $logOut / $logErr"
Write-Host '==> dashboard: http://localhost:8090'
Start-Process 'http://localhost:8090'
