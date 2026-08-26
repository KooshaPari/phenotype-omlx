# Stop the Bench Cockpit instance started by scripts\start-dev-windows.ps1.
# Uses .run\cockpit.pid; refuses to touch :8090 when no pid file is present
# (the listener may belong to another launch path - report, don't kill).
$RunDir = Join-Path (Split-Path -Parent $PSScriptRoot) '.run'
$pidFile = Join-Path $RunDir 'cockpit.pid'

if (Test-Path $pidFile) {
    $savedPid = [int](Get-Content $pidFile)
    $proc = Get-Process -Id $savedPid -ErrorAction SilentlyContinue
    if ($proc) {
        Stop-Process -Id $savedPid -Force
        Write-Host "==> stopped cockpit (pid $savedPid)"
    } else {
        Write-Host "==> pid $savedPid not running (already stopped)"
    }
    Remove-Item $pidFile -Force
} else {
    $listener = Get-NetTCPConnection -LocalPort 8090 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($listener) {
        Write-Host "no pid file; :8090 is owned by PID $($listener.OwningProcess) - not touched (not ours or launched via another path)"
    } else {
        Write-Host 'cockpit not running'
    }
}
