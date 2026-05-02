# Автозапуск opus на Windows при старте системы

Чтобы бот запускался сам после перезагрузки и работал в фоне (без видимого окна),
используй Task Scheduler — это встроенный в Windows планировщик.

## Вариант 1. Через графический Task Scheduler

1. Win → набери `Task Scheduler` (или "Планировщик заданий") → Enter.
2. Справа → **Create Task...** (Создать задачу...).
3. Вкладка **General**:
   - Name: `opus`
   - Security options: **Run whether user is logged on or not** (работает даже когда ты разлогинен)
   - Check: **Run with highest privileges**
   - Configure for: **Windows 10** (или 11, как есть).
4. Вкладка **Triggers** → **New...**:
   - Begin the task: **At startup**
   - Enabled: yes.
5. Вкладка **Actions** → **New...**:
   - Action: **Start a program**
   - Program/script: `powershell.exe`
   - Add arguments:
     ```
     -ExecutionPolicy Bypass -WindowStyle Hidden -File "D:\opus\scripts\start_windows.ps1"
     ```
     (замени `D:\opus` на твой реальный путь).
   - Start in: `D:\opus`
6. Вкладка **Conditions**:
   - **Uncheck** "Start the task only if the computer is on AC power" (чтобы работало на батарее тоже).
7. Вкладка **Settings**:
   - Check: "If the task fails, restart every: 1 minute"
   - Check: "Attempt to restart up to: 99 times"
   - Check: "If the running task does not end when requested, force it to stop"
8. OK → введи пароль Windows-юзера (нужно для опции "run whether logged on or not").

Готово. Перезагружаешь → бот стартует сам в фоне → UI доступен на http://127.0.0.1:8080.

## Вариант 2. Через PowerShell (команда)

Скопируй и выполни в PowerShell **от администратора** (замени путь):

```powershell
$opusDir = "D:\opus"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$opusDir\scripts\start_windows.ps1`"" `
    -WorkingDirectory $opusDir
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 99 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 365)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType S4U -RunLevel Highest
Register-ScheduledTask -TaskName "opus" `
    -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
    -Description "opus: Binance Futures microstructure collector"
```

Проверить задачу:
```powershell
Get-ScheduledTask -TaskName "opus"
Start-ScheduledTask -TaskName "opus"     # запустить сейчас (проверить что работает)
Stop-ScheduledTask -TaskName "opus"      # остановить
Unregister-ScheduledTask -TaskName "opus" -Confirm:$false   # удалить задачу
```

## Отключение сна и обновлений Windows

Без этого Windows может уснуть и/или перезагрузиться ради обновлений, потеряв часы данных.

1. **Отключить сон:**
   - Settings → System → Power & battery → Screen and sleep → в обоих dropdown'ах поставить **Never**.
   - Settings → System → Power & battery → Power mode → **Best performance**.

2. **Отключить hibernation:**
   ```powershell
   # В PowerShell от администратора:
   powercfg /hibernate off
   ```

3. **Отложить автоматические обновления:**
   - Settings → Windows Update → **Pause updates** на максимум (обычно 5 недель).
   - Либо Settings → Windows Update → Advanced options → Active hours → выставь `0:00 - 23:59` чтобы ребут не делался автоматически.

Этого достаточно чтобы бот не спал и не перезагружался внезапно.

## Посмотреть логи бота

Логи пишутся в `OPUS_LOGS_DIR` из `.env` (по умолчанию `./logs/opus.log`).

```powershell
# Последние 50 строк:
Get-Content D:\opus\logs\opus.log -Tail 50

# В реальном времени (как tail -f):
Get-Content D:\opus\logs\opus.log -Wait -Tail 20
```
