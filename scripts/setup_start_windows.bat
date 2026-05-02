@echo off
setlocal EnableExtensions

rem setup_start_windows.bat
rem Creates .venv if missing, installs dependencies, writes .env, starts opus.
rem
rem Usage:
rem   scripts\setup_start_windows.bat
rem   scripts\setup_start_windows.bat --python "C:\full\path\to\python.exe"
rem
rem If .venv already exists, system Python is not needed.

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul
set "OPUS_HOME=%CD%"
popd >nul

set "PY=%OPUS_HOME%\.venv\Scripts\python.exe"
set "BASE_PY=%OPUS_BASE_PYTHON%"

if /I "%~1"=="--python" (
  set "BASE_PY=%~2"
)

if defined BASE_PY set "BASE_PY=%BASE_PY:"=%"

echo.
echo opus setup + start
echo repo: %OPUS_HOME%
echo.

if not exist "%OPUS_HOME%\pyproject.toml" (
  echo ERROR: pyproject.toml not found. Put this script inside the opus scripts folder.
  exit /b 1
)

if not exist "%PY%" (
  if not defined BASE_PY (
    echo ERROR: .venv is missing and base Python was not provided.
    echo Run:
    echo   scripts\setup_start_windows.bat --python "C:\full\path\to\python.exe"
    echo or set OPUS_BASE_PYTHON to the full python.exe path.
    exit /b 1
  )
  if not exist "%BASE_PY%" (
    echo ERROR: Python not found at "%BASE_PY%"
    exit /b 1
  )

  echo Creating Python venv...
  pushd "%OPUS_HOME%"
  "%BASE_PY%" -m venv .venv
  if errorlevel 1 (
    echo Standard venv creation failed. Retrying without bundled pip...
    "%BASE_PY%" -m venv --without-pip .venv
  )
  popd

  if not exist "%PY%" (
    echo ERROR: could not create .venv using "%BASE_PY%"
    exit /b 1
  )
)

if not exist "%PY%" (
  echo ERROR: venv python not found at "%PY%"
  exit /b 1
)

pushd "%OPUS_HOME%"

echo Upgrading pip...
"%PY%" -m pip install -U pip
if errorlevel 1 exit /b %ERRORLEVEL%

echo Installing opus dependencies...
"%PY%" -m pip install -e ".[dev,ml]"
if errorlevel 1 exit /b %ERRORLEVEL%

echo Installing Polars compatibility runtime for legacy CPUs...
"%PY%" -m pip uninstall -y polars polars-runtime-32 polars-runtime-64 polars-lts-cpu >nul 2>&1
"%PY%" -m pip install --upgrade --force-reinstall --no-cache-dir "polars[rtcompat]>=1.34"
if errorlevel 1 exit /b %ERRORLEVEL%
"%PY%" -c "import polars as pl; raise SystemExit(0 if hasattr(pl, 'DataFrame') else 1)"
if errorlevel 1 (
  echo ERROR: Polars import smoke-test failed. The installed package does not expose pl.DataFrame.
  exit /b 1
)

if not exist ".env" (
  if not exist ".env.example" (
    echo ERROR: .env.example not found.
    popd
    exit /b 1
  )
  copy /Y ".env.example" ".env" >nul
  echo Created .env from .env.example
)

if not exist "data" mkdir "data"
if not exist "logs" mkdir "logs"
if not exist "models" mkdir "models"

echo.
echo Starting opus. Open http://127.0.0.1:8081
echo Press Ctrl+C in this window to stop.
echo.

"%PY%" -m backend.main
popd
exit /b %ERRORLEVEL%
