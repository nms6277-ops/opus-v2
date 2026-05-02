@echo off
setlocal EnableExtensions

rem Train and then backtest candidate symbol-agnostic 1s/2s/3s models.
rem Extra args are forwarded to both train and backtest.

set "SCRIPT_DIR=%~dp0"

call "%SCRIPT_DIR%train_agnostic_1s2s3s.bat" %*
if errorlevel 1 exit /b %ERRORLEVEL%

call "%SCRIPT_DIR%backtest_agnostic_1s2s3s.bat" %*
exit /b %ERRORLEVEL%
