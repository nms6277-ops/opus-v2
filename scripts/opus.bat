@echo off
setlocal EnableExtensions EnableDelayedExpansion
rem ============================================================
rem  opus.bat - single launcher for the opus trading bot.
rem
rem  Place this file (and this whole `scripts\` folder) inside
rem  D:\opus.  Then run from anywhere:
rem
rem      D:\opus\scripts\opus.bat <command>  [args...]
rem
rem  Or add D:\opus\scripts to your PATH and just type:
rem      opus <command>
rem ============================================================

rem -- Resolve the repo root (= parent of this script's folder) --
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul
set "OPUS_HOME=%CD%"
popd >nul

set "PY=%OPUS_HOME%\.venv\Scripts\python.exe"
set "PIDFILE=%OPUS_HOME%\logs\opus.pid"

if "%~1"=="" goto :help
set "CMD=%~1"
rem ``%*`` does NOT respect ``shift`` in cmd.exe, so we precompute REST
rem (= every original argument *after* %1) ourselves. Each handler uses
rem !REST! when forwarding args to Python.
set "REST="
for /f "tokens=1,*" %%a in ("%*") do set "REST=%%b"

if /I "%CMD%"=="help"     goto :help
if /I "%CMD%"=="-h"       goto :help
if /I "%CMD%"=="--help"   goto :help
if /I "%CMD%"=="start"    goto :start
if /I "%CMD%"=="stop"     goto :stop
if /I "%CMD%"=="restart"  goto :restart
if /I "%CMD%"=="status"   goto :status
if /I "%CMD%"=="logs"     goto :logs
if /I "%CMD%"=="train"    goto :train
if /I "%CMD%"=="backtest" goto :backtest
if /I "%CMD%"=="config"   goto :config
if /I "%CMD%"=="open-ui"  goto :openui
if /I "%CMD%"=="trades"   goto :trades

echo Unknown command: %CMD%
echo.
goto :help

rem ============================================================
:help
echo.
echo   opus -- launcher for the opus trading bot
echo   repo: %OPUS_HOME%
echo.
echo   Commands:
echo     opus start                start the bot in this window (Ctrl+C to stop)
echo     opus stop                 stop a running bot (looks up PID in logs\opus.pid)
echo     opus restart              stop + start
echo     opus status               check whether the bot is running and on which mode
echo     opus logs                 tail the latest log file
echo     opus open-ui              open http://127.0.0.1:8081 in the default browser
echo     opus trades [date]        print today's paper-trade summary (or YYYY-MM-DD)
echo.
echo     opus train  [args]        train LightGBM models on collected snapshots
echo                               examples:
echo                                 opus train
echo                                 opus train --horizons 2s 5s 15s
echo                                 opus train --symbols AIOTUSDT BTCUSDT
echo                                 opus train --from-date 2026-04-22
echo                                 opus train --per-symbol --optuna-trials 8
echo                                 opus train --no-symbol-feature
echo                                     ^^ symbol-agnostic model that
echo                                     transfers to unseen coins
echo.
echo     opus backtest [args]      replay test slice on trained models
echo                               examples:
echo                                 opus backtest
echo                                 opus backtest --horizons 2s 15s
echo                                 opus backtest --symbols AIOTUSDT
echo.
echo     opus config               print current settings (mode, horizon, symbols, ...)
echo     opus help                 show this message
echo.
echo   Run `opus train -h` or `opus backtest -h` for full Python-level options.
echo.
goto :eof

rem ============================================================
:start
if not exist "%PY%" (
  echo ERROR: venv not found at "%PY%"
  echo Run scripts\install_windows.ps1 first.
  exit /b 1
)
if not exist "%OPUS_HOME%\.env" (
  echo ERROR: .env not found in %OPUS_HOME%
  echo Copy .env.example to .env, edit it, and try again.
  exit /b 1
)
echo Starting opus (Ctrl+C to stop) ...
echo PID dir: %OPUS_HOME%\logs
if not exist "%OPUS_HOME%\logs" mkdir "%OPUS_HOME%\logs"
rem Record our own console PID so `opus stop` can find us.
echo %ERRORLEVEL% > nul
for /f "tokens=2 delims=," %%P in ('tasklist /FI "PID eq %PROCESSOR_ARCHITECTURE%" /FO CSV /NH 2^>nul') do (rem nop)
rem Use wmic to grab the cmd.exe PID hosting this script.
for /f "tokens=2 delims==" %%i in ('wmic process where "Name='cmd.exe' and CommandLine like '%%opus.bat%%'" get ProcessId /value 2^>nul ^| find "="') do set "OPUS_PID=%%i"
if defined OPUS_PID > "%PIDFILE%" echo !OPUS_PID!
pushd "%OPUS_HOME%"
"%PY%" -m backend.main !REST!
set "RC=%ERRORLEVEL%"
popd
if exist "%PIDFILE%" del "%PIDFILE%" >nul 2>&1
exit /b %RC%

rem ============================================================
:stop
echo Stopping opus...
rem Try graceful taskkill of any python.exe whose command line contains backend.main
set "FOUND=0"
for /f "tokens=2 delims==" %%i in ('wmic process where "Name='python.exe' and CommandLine like '%%backend.main%%'" get ProcessId /value 2^>nul ^| find "="') do (
  echo   killing PID %%i
  taskkill /PID %%i /T /F >nul 2>&1
  set "FOUND=1"
)
if "%FOUND%"=="0" (
  echo No running opus process found.
) else (
  echo Done.
)
if exist "%PIDFILE%" del "%PIDFILE%" >nul 2>&1
exit /b 0

rem ============================================================
:restart
call "%~f0" stop
timeout /t 2 /nobreak >nul
call "%~f0" start !REST!
exit /b %ERRORLEVEL%

rem ============================================================
:status
set "RUNNING="
for /f "tokens=2 delims==" %%i in ('wmic process where "Name='python.exe' and CommandLine like '%%backend.main%%'" get ProcessId /value 2^>nul ^| find "="') do set "RUNNING=%%i"
if defined RUNNING (
  echo opus is RUNNING -- PID !RUNNING!
) else (
  echo opus is NOT running.
)
echo.
echo --- effective config (.env via Pydantic) ---
pushd "%OPUS_HOME%"
"%PY%" -c "from backend.config import settings as s; print(f'mode             = {s.mode}'); print(f'data_dir         = {s.data_dir}'); print(f'models_dir       = {s.models_dir}'); print(f'trade_horizon    = {s.trade_horizon}'); print(f'trade_symbols    = {s.trade_symbols}'); print(f'conf_threshold   = {s.trade_conf_threshold}'); print(f'notional_usd     = {s.trade_notional_usd}'); print(f'taker_fee_bp     = {s.taker_fee_bp}')"
popd
exit /b 0

rem ============================================================
:logs
set "LATEST="
for /f "delims=" %%f in ('dir /b /o:-d "%OPUS_HOME%\logs\*.log" 2^>nul') do (
  if not defined LATEST set "LATEST=%%f"
)
if not defined LATEST (
  echo No log files in %OPUS_HOME%\logs
  exit /b 1
)
echo Tailing %OPUS_HOME%\logs\!LATEST!  -- Ctrl+C to stop
powershell -NoLogo -Command "Get-Content -Wait -Tail 50 '%OPUS_HOME%\logs\!LATEST!'"
exit /b 0

rem ============================================================
:openui
start "" "http://127.0.0.1:8081"
exit /b 0

rem ============================================================
:trades
rem REST holds everything after "trades"; first token is optional date.
set "TDATE="
for /f "tokens=1" %%a in ("!REST!") do set "TDATE=%%a"
if "%TDATE%"=="" (
  for /f %%i in ('powershell -NoLogo -Command "[DateTime]::UtcNow.ToString('yyyy-MM-dd')"') do set "TDATE=%%i"
)
pushd "%OPUS_HOME%"
"%PY%" -c "import polars as pl, sys; from backend.config import settings as s; from pathlib import Path; p = Path(s.data_dir) / 'trades' / f'%TDATE%.parquet'; print(f'reading {p}'); df = pl.read_parquet(p); print(f'rows: {df.height}'); print(); print(df.group_by('symbol','side','exit_reason').agg(pl.len().alias('n'), pl.col('net_bp').mean().alias('avg_net'), pl.col('net_bp').sum().alias('sum_net'), (pl.col('net_bp')>0).sum().alias('wins')).sort('symbol','side')); print(); print('TOTAL  n=', df.height, ' avg_net_bp=', round(float(df['net_bp'].mean()),3), ' win_rate=', round(float((df['net_bp']>0).mean()),3))"
popd
exit /b %ERRORLEVEL%

rem ============================================================
:train
rem train.py itself appends ``global/`` (or ``per_symbol/...``) to the
rem ``--models-dir`` value, so we MUST pass the parent folder here. If
rem we pass ``models\global`` the artefacts end up in
rem ``models\global\global\h{horizon}\`` and inference + backtest can't
rem find them.
echo Training models -- this can take 5-30 min depending on data size.
echo Default output: %OPUS_HOME%\models\global\h^<horizon^>\
echo (override with --models-dir, e.g. --models-dir models\agnostic)
echo.
pushd "%OPUS_HOME%"
"%PY%" -m backend.ml.train --models-dir "%OPUS_HOME%\models" !REST!
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%

rem ============================================================
:backtest
pushd "%OPUS_HOME%"
"%PY%" -m backend.ml.backtest --models-dir "%OPUS_HOME%\models\global" !REST!
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%

rem ============================================================
:config
pushd "%OPUS_HOME%"
"%PY%" -c "from backend.config import settings; import json; d = settings.model_dump(); print(json.dumps({k: (str(v) if hasattr(v,'__fspath__') else v) for k,v in d.items()}, indent=2, default=str, ensure_ascii=False))"
popd
exit /b 0
