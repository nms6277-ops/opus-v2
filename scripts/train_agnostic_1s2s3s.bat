@echo off
setlocal EnableExtensions

rem Train a symbol-agnostic LightGBM model on collector snapshots.
rem Copy this file to D:\opus\scripts and run:
rem   D:\opus\scripts\train_agnostic_1s2s3s.bat
rem Optional extra args are forwarded to backend.ml.train, e.g.:
rem   D:\opus\scripts\train_agnostic_1s2s3s.bat --from-date 2026-04-30

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul
set "OPUS_HOME=%CD%"
popd >nul

set "PY=%OPUS_HOME%\.venv\Scripts\python.exe"
set "DATA_DIR=%OPUS_HOME%\data"
set "MODELS_PARENT=%OPUS_HOME%\models\agnostic_next"
set "HORIZONS=1s 2s 3s"
set "ROUNDS=800"
set "EARLY_STOPPING=50"
set "HELP_MODE=0"
if /I "%~1"=="--help" set "HELP_MODE=1"
if /I "%~1"=="-h" set "HELP_MODE=1"

if not exist "%PY%" (
  echo ERROR: venv not found at "%PY%"
  echo Run scripts\install_windows.ps1 first.
  exit /b 1
)

if not exist "%DATA_DIR%\snapshots" (
  echo ERROR: snapshots not found at "%DATA_DIR%\snapshots"
  exit /b 1
)

echo.
echo Training symbol-agnostic models
echo   repo          = %OPUS_HOME%
echo   data_dir      = %DATA_DIR%
echo   output_parent = %MODELS_PARENT%
echo   output_models = %MODELS_PARENT%\global\h1s,h2s,h3s
echo   horizons      = %HORIZONS%
echo   extra args    = %*
echo.

pushd "%OPUS_HOME%"
"%PY%" -c "import lightgbm" >nul 2>&1
if errorlevel 1 (
  echo Installing ML dependencies...
  "%PY%" -m pip install -e ".[ml]"
  if errorlevel 1 exit /b 1
)

"%PY%" -m backend.ml.train ^
  --data-dir "%DATA_DIR%" ^
  --models-dir "%MODELS_PARENT%" ^
  --horizons %HORIZONS% ^
  --no-symbol-feature ^
  --rounds %ROUNDS% ^
  --early-stopping %EARLY_STOPPING% ^
  --log-level INFO ^
  %*
set "RC=%ERRORLEVEL%"
popd

if not "%RC%"=="0" (
  echo.
  echo TRAIN FAILED with code %RC%
  exit /b %RC%
)

if "%HELP_MODE%"=="1" exit /b 0

echo.
echo TRAIN OK.
echo New models are in:
echo   %MODELS_PARENT%\global
exit /b 0
