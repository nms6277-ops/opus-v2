@echo off
setlocal EnableExtensions

rem Promote models\agnostic_next\global to models\agnostic\global.
rem Existing production models are copied to models\agnostic_backup_<timestamp>.
rem Run this only after reading backtest reports.

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul
set "OPUS_HOME=%CD%"
popd >nul

set "SRC=%OPUS_HOME%\models\agnostic_next\global"
set "DST_PARENT=%OPUS_HOME%\models\agnostic"
set "DST=%DST_PARENT%\global"

if not exist "%SRC%" (
  echo ERROR: source model dir not found:
  echo   %SRC%
  exit /b 1
)

for /f %%i in ('powershell -NoLogo -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%i"
set "BACKUP=%OPUS_HOME%\models\agnostic_backup_%STAMP%"

echo.
echo Promoting candidate models
echo   source = %SRC%
echo   dest   = %DST%
echo   backup = %BACKUP%
echo.

if exist "%DST%" (
  echo Backing up current production model...
  robocopy "%DST_PARENT%" "%BACKUP%" /E /XD "%BACKUP%" >nul
  if errorlevel 8 (
    echo ERROR: backup failed.
    exit /b 1
  )
)

if not exist "%DST_PARENT%" mkdir "%DST_PARENT%"
robocopy "%SRC%" "%DST%" /MIR
set "RC=%ERRORLEVEL%"
if %RC% GEQ 8 (
  echo ERROR: promote copy failed with robocopy code %RC%
  exit /b 1
)

echo.
echo PROMOTE OK.
echo Production model is now:
echo   %DST%
exit /b 0
