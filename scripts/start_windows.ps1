# opus — Windows start script
#
# Usage:
#     .\scripts\start_windows.ps1
#
# Activates the venv and launches the bot. Leave this PowerShell window open.
# Close the window or press Ctrl+C to stop.
#
# For unattended 24/7 operation see scripts\autostart_windows.md

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSCommandPath | Split-Path -Parent)

$venvPy = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "ERROR: venv not found. Run .\scripts\install_windows.ps1 first." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path ".env")) {
    Write-Host "ERROR: .env not found. Copy .env.example to .env and edit it." -ForegroundColor Red
    exit 1
}

Write-Host "Starting opus (Ctrl+C to stop) ..." -ForegroundColor Cyan
& $venvPy -m backend.main
