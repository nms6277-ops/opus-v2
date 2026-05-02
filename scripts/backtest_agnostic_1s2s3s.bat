@echo off
setlocal EnableExtensions

rem Backtest the candidate symbol-agnostic 1s/2s/3s models.
rem Run after train_agnostic_1s2s3s.bat.
rem Optional extra args are forwarded to backend.ml.backtest, e.g.:
rem   D:\opus\scripts\backtest_agnostic_1s2s3s.bat --symbols UBUSDT MEGAUSDT

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul
set "OPUS_HOME=%CD%"
popd >nul

set "PY=%OPUS_HOME%\.venv\Scripts\python.exe"
set "DATA_DIR=%OPUS_HOME%\data"
set "MODELS_PARENT=%OPUS_HOME%\models\agnostic_next"
set "OUT_DIR=%MODELS_PARENT%\backtest"
set "HORIZONS=1s 2s 3s"
set "TARGET_TRADE_FRAC=0.05"
set "HELP_MODE=0"
if /I "%~1"=="--help" set "HELP_MODE=1"
if /I "%~1"=="-h" set "HELP_MODE=1"

if not exist "%PY%" (
  echo ERROR: venv not found at "%PY%"
  exit /b 1
)

if "%HELP_MODE%"=="0" if not exist "%MODELS_PARENT%\global" (
  echo ERROR: candidate models not found at "%MODELS_PARENT%\global"
  echo Run scripts\train_agnostic_1s2s3s.bat first.
  exit /b 1
)

echo.
echo Backtesting candidate models
echo   repo              = %OPUS_HOME%
echo   data_dir          = %DATA_DIR%
echo   models_parent     = %MODELS_PARENT%
echo   out_dir           = %OUT_DIR%
echo   horizons          = %HORIZONS%
echo   target_trade_frac = %TARGET_TRADE_FRAC%
echo   extra args        = %*
echo.

pushd "%OPUS_HOME%"
"%PY%" -m backend.ml.backtest ^
  --data-dir "%DATA_DIR%" ^
  --models-dir "%MODELS_PARENT%" ^
  --horizons %HORIZONS% ^
  --target-trade-frac %TARGET_TRADE_FRAC% ^
  --out-dir "%OUT_DIR%" ^
  --log-level INFO ^
  %*
set "RC=%ERRORLEVEL%"
popd

if not "%RC%"=="0" (
  echo.
  echo BACKTEST FAILED with code %RC%
  exit /b %RC%
)

if "%HELP_MODE%"=="1" exit /b 0

echo.
echo BACKTEST OK.
echo Reports:
echo   %OUT_DIR%\summary.json
echo   %OUT_DIR%\h1s.json
echo   %OUT_DIR%\h2s.json
echo   %OUT_DIR%\h3s.json
exit /b 0
