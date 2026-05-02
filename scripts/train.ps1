# Train opus ML MVP-1 models on Windows.
# Usage:
#   .\scripts\train.ps1                   # default: global model, all symbols, all horizons
#   .\scripts\train.ps1 -Symbols BTCUSDT  # restrict to one symbol
#   .\scripts\train.ps1 -PerSymbol        # one model per symbol per horizon
#   .\scripts\train.ps1 -Optuna 30        # run 30 Optuna trials per horizon
#   .\scripts\train.ps1 -MaxRowsPerSymbol 300000  # low-RAM training sample

[CmdletBinding()]
param(
    [string[]]$Symbols = @(),
    [string[]]$Horizons = @(),
    [switch]$PerSymbol,
    [int]$Optuna = 0,
    [int]$Rounds = 800,
    [int]$MaxRowsPerSymbol = 0,
    [string]$ModelsDir = "models",
    [string]$LogLevel = "INFO"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv")) {
    Write-Error "No .venv found. Run scripts\install_windows.ps1 first."
    exit 1
}

# Ensure ML deps are installed.
& .\.venv\Scripts\python.exe -c "import lightgbm" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing ML deps..."
    & .\.venv\Scripts\python.exe -m pip install -e ".[ml]"
}

$argList = @(
    "-m", "backend.ml.train",
    "--models-dir", $ModelsDir,
    "--rounds", $Rounds,
    "--log-level", $LogLevel
)
if ($Symbols.Count -gt 0)   { $argList += "--symbols";  $argList += $Symbols }
if ($Horizons.Count -gt 0)  { $argList += "--horizons"; $argList += $Horizons }
if ($PerSymbol)             { $argList += "--per-symbol" }
if ($Optuna -gt 0)          { $argList += "--optuna-trials"; $argList += $Optuna }
if ($MaxRowsPerSymbol -gt 0) { $argList += "--max-rows-per-symbol"; $argList += $MaxRowsPerSymbol }

Write-Host "Running: python $($argList -join ' ')"
& .\.venv\Scripts\python.exe @argList
exit $LASTEXITCODE
