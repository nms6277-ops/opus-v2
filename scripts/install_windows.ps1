# opus — Windows native installer (PowerShell)
#
# Usage (in PowerShell, in the repo directory):
#     .\scripts\install_windows.ps1
#
# Requires: Python 3.11 x64 from python.org, "Add python.exe to PATH" checked
# during install. If `py -3.11 --version` works, you are good.
#
# What this does:
#   1. Creates a venv in .\.venv
#   2. Installs dependencies from pyproject.toml
#   3. Installs the Polars compatibility runtime for older CPUs
#      (avoids RuntimeWarning: Missing required CPU features avx2/fma/bmi1/...)
#   4. Copies .env.example to .env if not present
#   5. Creates data/ and logs/ directories

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSCommandPath | Split-Path -Parent)

Write-Host ""
Write-Host "=== opus Windows installer ===" -ForegroundColor Cyan
Write-Host "cwd: $(Get-Location)"
Write-Host ""

# --- Python version check -----------------------------------------------------
$pythonCmd = $null
foreach ($candidate in @("py -3.11", "python3.11", "python")) {
    try {
        $null = Invoke-Expression "$candidate --version 2>&1"
        if ($LASTEXITCODE -eq 0) {
            $version = (Invoke-Expression "$candidate --version 2>&1").ToString()
            if ($version -match "3\.(11|12|13)") {
                $pythonCmd = $candidate
                Write-Host "Found Python: $version via '$candidate'" -ForegroundColor Green
                break
            }
        }
    } catch {}
}
if (-not $pythonCmd) {
    Write-Host "ERROR: Python 3.11+ not found. Install from https://www.python.org/downloads/windows/" -ForegroundColor Red
    Write-Host "Be sure to check 'Add python.exe to PATH' during install."
    exit 1
}

# --- venv ---------------------------------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "Creating venv in .\.venv ..." -ForegroundColor Yellow
    Invoke-Expression "$pythonCmd -m venv .venv"
}

$venvPy = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "ERROR: venv creation failed" -ForegroundColor Red
    exit 1
}

# --- pip install --------------------------------------------------------------
Write-Host "Upgrading pip + wheel ..." -ForegroundColor Yellow
& $venvPy -m pip install --upgrade pip wheel | Out-Null

Write-Host "Installing project dependencies (including ML extras) ..." -ForegroundColor Yellow
# ``ml`` is required by the training CLI (backend.ml.train -> lightgbm).
# ``dev`` ships pytest / ruff for the local workflow.
& $venvPy -m pip install -e ".[dev,ml]"

# --- Polars compatibility runtime (older CPUs, no AVX/AVX2) -------------------
Write-Host "Installing Polars compatibility runtime (older CPU safe) ..." -ForegroundColor Yellow
& $venvPy -m pip uninstall -y polars-runtime-32 polars-runtime-64 polars-lts-cpu | Out-Null
& $venvPy -m pip install --upgrade "polars[rtcompat]>=1.34"

# --- .env ---------------------------------------------------------------------
if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example" -ForegroundColor Green
} else {
    Write-Host ".env already exists, not overwriting" -ForegroundColor DarkGray
}

# --- dirs ---------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path data  | Out-Null
New-Item -ItemType Directory -Force -Path logs  | Out-Null
New-Item -ItemType Directory -Force -Path models | Out-Null

Write-Host ""
Write-Host "=== Installation OK ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Edit .env   (notepad .env)"
Write-Host "     - set OPUS_DATA_DIR=D:/opus   (or any folder with 30+ GB free)"
Write-Host "     - set OPUS_DAILY_LOSS_LIMIT_USD and OPUS_MAX_POSITION_USD"
Write-Host "  2. Start the bot:  .\scripts\start_windows.ps1"
Write-Host "  3. Open http://127.0.0.1:8080 in your browser"
Write-Host ""
