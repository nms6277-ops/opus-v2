# opus

Микроструктурный торговый бот под Binance USDT-M Futures с параллельным сбором данных с Bybit для cross-exchange фичей. Фаза MVP-0: коллектор данных + UI + заглушки paper/live (без ML).

> **Project state:** development paused after the v2 ML experiment.
> The full briefing is in [`docs/`](docs/README.md):
> [ARCHITECTURE](docs/ARCHITECTURE.md) ·
> [CONFIGURATION](docs/CONFIGURATION.md) ·
> [RESULTS](docs/RESULTS.md) ·
> [ROADMAP](docs/ROADMAP.md). Read those first if you are returning to
> the project after a break.

## Текущий статус (MVP-0)

-   Сбор стакана и aggTrade с Binance Futures по 4-5 монетам одновременно.
-   LOB reconstructor с корректной синхронизацией snapshot + diff и обнаружением sequence gaps.
-   Компактные снапшоты фичей (top-20 уровней + агрегаты) каждые 250 мс в parquet + zstd.
-   Чтение `orderbook.50` и `publicTrade` с Bybit для cross-exchange фичей (пока не используется в торговле).
-   Три режима: **COLLECT** (только сбор), **PAPER** (сбор + виртуальные ордера, позже), **LIVE** (реальная торговля, позже).
-   Safety guards: daily loss limit, max position size, max orders/min, max live symbols, network kill-switch.
-   FastAPI backend + минимальный HTML/JS UI с realtime WebSocket.
-   systemd unit для автозапуска на VPS.

**Что ещё не готово:** реальная торговая логика (сигналы, fill simulator, ML-модель). Следующей итерацией.

## Быстрый старт

### Предусловия

-   Ubuntu 22.04 / 24.04 (или совместимый Linux), либо Windows 10/11 с WSL2.
-   Python 3.11+ (`sudo apt install python3.11 python3.11-venv` если нет).
-   Доступ в интернет до `fstream.binance.com` и `stream.bybit.com`.
-   Минимум **30 GB свободного места** на диске с `OPUS_DATA_DIR` — парк данных за 10 дней × 5 монет.

### Установка на Linux (Ubuntu) или в WSL2 на Windows

```bash
git clone https://github.com/nms6277-ops/opus.git
cd opus
bash scripts/install_local.sh
```

Это создаст `.venv/`, поставит зависимости и скопирует `.env.example` в `.env`. **Отредактируй `.env`** перед запуском:

```dotenv
# Минимум:
OPUS_MODE=collect                  # стартуем с COLLECT
OPUS_DAILY_LOSS_LIMIT_USD=5.0     # максимум, что готов потерять за день (USD)
OPUS_MAX_POSITION_USD=50.0        # максимум на позицию (USD)
# Хранение данных (можно указать любую папку, напр. /mnt/d/opus-data в WSL):
OPUS_DATA_DIR=./data
OPUS_LOGS_DIR=./logs
```

API-ключи Binance нужны ТОЛЬКО для `LIVE` режима и пока можно не заполнять. Для сбора данных ключи не нужны — всё идёт через публичные стримы.

### Запуск

```bash
source .venv/bin/activate
python -m backend.main
```

Откроется на `http://127.0.0.1:8080` — открой в браузере на той же машине.

---

### Windows (native) — полная пошаговая инструкция

Простейший путь: Python + PowerShell, без WSL. Подходит для 24/7 коллектора.

1.  **Ставим Python 3.11** с https://www.python.org/downloads/windows/
    — обязательно галочка **"Add python.exe to PATH"** при установке.

2.  **Распаковываем код** в удобную папку, например `D:\opus`.

3.  **В PowerShell**, открытом в папке `D:\opus`, запускаем установщик:

    ```powershell
    .\scripts\install_windows.ps1
    ```

    Если появится ошибка `running scripts is disabled on this system` — сделай **один раз**:

    ```powershell
    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
    ```

    и повтори установку. Скрипт:
    - создаст `.venv`
    - поставит все зависимости
    - **подменит** `polars` на `polars-lts-cpu` (лечит предупреждение про AVX2/FMA/BMI на старых CPU)
    - скопирует `.env.example` в `.env`
    - создаст папки `data/`, `logs/`, `models/`.

4.  **Отредактируй `.env`** (`notepad .env`), минимум:

    ```
    OPUS_DATA_DIR=D:/opus-data
    OPUS_LOGS_DIR=D:/opus-logs
    OPUS_DAILY_LOSS_LIMIT_USD=5.0
    OPUS_MAX_POSITION_USD=50.0
    ```

    Пути через прямой слеш (`/`) — Python его понимает на Windows.

5.  **Запуск** (оставь окно открытым):

    ```powershell
    .\scripts\start_windows.ps1
    ```

6.  **UI** — в любом браузере: `http://127.0.0.1:8080`.

7.  **24/7 автозапуск** — см. `scripts/autostart_windows.md`.
    Короткая версия: Task Scheduler → Create Task → At startup → `powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File D:\opus\scripts\start_windows.ps1`.

8.  **Просмотр логов**:

    ```powershell
    Get-Content D:\opus\logs\opus.log -Wait -Tail 20
    ```

---

### Windows + WSL2 — полная пошаговая инструкция

1.  **Ставим WSL2 с Ubuntu 24.04** (в PowerShell от администратора):

    ```powershell
    wsl --install -d Ubuntu-24.04
    ```

    После установки — перезагрузка, открой «Ubuntu» в меню Пуск, задай юзера и пароль.

2.  **В Ubuntu-терминале — базовые пакеты:**

    ```bash
    sudo apt update
    sudo apt install -y python3.11 python3.11-venv git tmux
    ```

3.  **Клонируем код:**

    ```bash
    cd ~
    git clone https://github.com/nms6277-ops/opus.git
    cd opus
    bash scripts/install_local.sh
    ```

4.  **(Опционально) Хранить данные на диске `D:\`**

    В WSL Windows-диски доступны как `/mnt/c/`, `/mnt/d/` и т.д. Отредактируй `.env`:

    ```bash
    nano .env
    ```

    Меняем две строки:

    ```
    OPUS_DATA_DIR=/mnt/d/opus-data
    OPUS_LOGS_DIR=/mnt/d/opus-logs
    ```

    Папки создадутся сами при запуске. Убедись что на D:\ есть минимум 30 GB.

5.  **Запуск в tmux** (чтобы бот работал 24/7, даже если закроешь окно WSL):

    ```bash
    tmux new -s opus
    source .venv/bin/activate
    python -m backend.main
    ```

    `Ctrl+B`, потом `D` — отсоединяешься, бот продолжает работать.
    `tmux attach -t opus` — вернуться к логам.

6.  **Открываешь UI** в обычном Windows-браузере (Chrome / Edge):

    ```
    http://127.0.0.1:8080
    ```

    WSL2 автоматически пробрасывает порт `127.0.0.1:8080` из Linux на Windows.

7.  **Обновление до новой версии** (когда я запушу что-то новое):

    ```bash
    cd ~/opus
    git pull
    source .venv/bin/activate
    pip install -e '.[dev]'
    ```

    Перезапуск в tmux: `Ctrl+C` в tmux-сессии, потом `python -m backend.main`.

**Troubleshooting WSL:**

-   UI не открывается в Windows-браузере → проверь, что бот действительно слушает: `ss -ltnp | grep 8080` в WSL.
-   I/O медленный на `/mnt/d/` → это нормально, WSL2 пишет на NTFS через 9P. Для высоких нагрузок клади данные в `~/opus/data` (ext4 — быстрее), а архивируй на D:\ через `rsync` раз в сутки.
-   WSL «засыпает» когда нет активности → в `.wslconfig` в `C:\Users\<user>\`:
    ```
    [wsl2]
    memory=2GB
    ```
    (ограничивает RAM, но остаётся живым).

### Автозапуск

**В WSL2 / Linux** — через systemd:

```bash
sudo cp systemd/opus.service /etc/systemd/system/opus@$USER.service
sudo systemctl daemon-reload
sudo systemctl enable --now opus@$USER
sudo systemctl status opus@$USER
journalctl -u opus@$USER -f
```

(Для WSL2 нужен включённый systemd — в Ubuntu 24 он включён по умолчанию, на старых версиях добавь `[boot] systemd=true` в `/etc/wsl.conf` и перезагрузи WSL через `wsl --shutdown`.)

### SSH-tunnel (не нужен для локального запуска)

SSH-tunnel нужен ТОЛЬКО если запускаешь бот на удалённой машине (например, Tokyo VPS для LIVE-режима). Тогда с ноутбука:

```bash
ssh -L 8080:127.0.0.1:8080 user@vps-ip
```

И открой `http://127.0.0.1:8080` уже в браузере ноутбука. На локальной Windows-машине это не нужно — просто открываешь браузер и идёшь на `http://127.0.0.1:8080`.

## Архитектура

```
                                  ┌─────────────────────────────────────────────┐
                                  │  FastAPI / WebSocket (127.0.0.1:8080)       │
                                  │    • REST  /api/*                            │
                                  │    • WS    /ws  (realtime UI)                │
                                  └────────────▲────────────────────────────────┘
                                               │
 ┌────────────┐   depth+aggTrade   ┌───────────┴─────────────┐    parquet + zstd
 │ Binance FUT├────────── WS ──────►  Runtime (asyncio)       ├──► data/snapshots/
 └────────────┘                     │   ├── LOB per symbol    │    {sym}/{date}/{hh}.parquet
 ┌────────────┐   book + trades    │   ├── Trade buffer      │
 │ Bybit lin. ├────────── WS ──────►   ├── Snapshot loop     │
 └────────────┘                     │   ├── Paper/Live trader │
                                    │   └── Safety guards     │
                                    └─────────────────────────┘
```

-   `backend/runtime.py` — главный оркестратор.
-   `backend/collector/lob.py` — реконструктор стакана (snapshot + diff sync).
-   `backend/collector/writer.py` — parquet writer с ротацией.
-   `backend/features/snapshot.py` — вычисление фичей (20 top-levels + микроцена + OFI-like imbalance).
-   `backend/exchanges/{binance,bybit}_ws.py` — WebSocket клиенты с reconnect.
-   `backend/exchanges/binance_rest.py` — REST для snapshot + (в будущем) ордера.
-   `backend/traders/{paper,live}.py` — заглушки торговых движков (MVP-0).
-   `backend/safety/guards.py` — предтрейдовые проверки.
-   `backend/api/{rest,ws}.py` — HTTP и WS endpoints для UI.
-   `frontend/` — HTML/CSS/JS (без фреймворков).

## Структура данных на диске

```
data/snapshots/
  BTCUSDT/
    2025-04-28/
      00.parquet
      01.parquet
      ...
  SOLUSDT/
    ...
```

Каждый parquet-файл — зона времени (по умолчанию 1 час), сжат zstd. Схема row:

-   `ts_ms`, `symbol`
-   `best_bid`, `best_ask`, `best_bid_qty`, `best_ask_qty`
-   `spread`, `spread_bp`, `mid`, `microprice`
-   `imbalance_top1`, `imbalance_top5`, `imbalance_top20`
-   `bid_vol_top5`, `ask_vol_top5`, `bid_vol_top20`, `ask_vol_top20`
-   `bid_weighted_depth_20`, `ask_weighted_depth_20`
-   `buy_volume_win`, `sell_volume_win`, `trade_count_win`, `vwap_win` (окно = 1 сек)
-   `bid_p_00..bid_p_19`, `bid_q_00..bid_q_19`, `ask_p_00..ask_p_19`, `ask_q_00..ask_q_19`

## Workflow

1. **COLLECT** (по умолчанию): добавь 4-5 монет в watchlist через UI, бот собирает данные 24/7.
2. Параллельно каждую ночь перекачивай `data/` на домашний сервер для тренировки моделей:
    ```bash
    rsync -az --progress user@vps:~/opus/data/snapshots/ ~/opus-archive/
    ```
3. Через 1-2 недели первый archive → тренируем cross-sectional модель локально.
4. Выкатываем модель в `models/` и переключаем mode в **PAPER**.
5. После стабильного paper — очень маленький **LIVE** с явными лимитами.

## Проверки здоровья

UI показывает:

-   Статус WS (binance / bybit) + задержку от последнего сообщения.
-   По каждой монете: количество snapshots записано, reconnect count, sequence gaps.
-   Текущие bid/ask/spread/microprice.
-   Daily PnL + order count.
-   Emergency stop баннер + кнопка clear.

Логи в `logs/opus.log` (ротация 50 MB × 10 файлов).

## Безопасность (must read)

-   **`.env` никогда не коммитим.** Он в `.gitignore`.
-   **API-ключи Binance:** в настройках ключа включи IP whitelist на IP твоего VPS. Откажись от прав на вывод средств.
-   **UI на 127.0.0.1** — не выставляй наружу. Доступ через SSH tunnel.
-   **LIVE mode** пока ничего не отправляет (safe stub). Когда будет включён, `daily_loss_limit_usd` и `max_position_usd` в `.env` и в UI — жёсткий cap.

## ML pipeline (MVP-1)

Когда накоплено достаточно данных (минимум ~1M строк на символ, ~10 дней
24/7 для пяти топ-символов), можно тренировать модели.

```powershell
# Windows: установит lightgbm+optuna+sklearn в venv и запустит train
.\scripts\train.ps1                                    # global модель, все символы
.\scripts\train.ps1 -Symbols BTCUSDT ETHUSDT           # подмножество символов
.\scripts\train.ps1 -Horizons 1s 5s                    # только эти горизонты
.\scripts\train.ps1 -PerSymbol                         # одна модель на символ
.\scripts\train.ps1 -Optuna 30                         # 30 trials поиска параметров
```

```bash
# Linux:
pip install -e ".[ml]"
python -m backend.ml.train --models-dir ./models
```

Что делает:

1.  **`backend.ml.dataset`** — сканирует `OPUS_DATA_DIR/snapshots/`, объединяет
    parquet файлы по каждому символу, делает **time-aware split** 70/15/15
    (никогда не shuffle, чтобы избежать look-ahead bias).
2.  **`backend.ml.features`** — считает производные фичи: mid-returns на
    лагах (0.25s..2min), rolling volatility, top-of-book OFI, нормализованный
    spread, bucket-volume концентрация, trade-flow imbalance.
3.  **`backend.ml.labels`** — 3-class direction (UP/FLAT/DOWN) на горизонтах
    1s, 5s, 30s. Threshold между UP/FLAT/DOWN — **per-symbol**, по медиане
    |return| на train слайсе.
4.  **`backend.ml.train`** — LightGBM multi-class классификатор на каждый
    горизонт; категориальная фича `_symbol_id` чтобы одна модель работала на
    всех символах. Опционально Optuna для поиска гиперпараметров.
5.  **`backend.ml.eval`** — на test слайсе считает: AUC(UP vs DOWN),
    hit-rate@confidence-threshold, средний net return на сделку с учётом
    taker-fee 4 bp × 2, наивный Sharpe per day.

Результат сохраняется в `models/global/h{1s,5s,30s}/`:

```
models/
  global/
    h1s/
      model.lgb              # LightGBM booster
      meta.json              # фичи, thresholds, ts тренировки
      eval.json              # AUC / hit-rate / sim Sharpe
      feature_importance.json
    h5s/...
    h30s/...
  summary.json               # сводный отчёт по всем горизонтам
```

Если AUC < 0.55 на val — сигнала нет, нужны новые фичи или больше данных.
AUC 0.58+ — есть edge для построения стратегии. AUC 0.62+ — золото.

## Paper-trading on a trained model

Once you have models in `models/global/h{1s,5s,30s}/`, switch the bot
into PAPER mode and the trader will start opening virtual market-taker
positions whenever the predictor's confidence exceeds a threshold.

Set the runtime parameters via `.env` (or `OPUS_*` env vars) — see
[`.env.example`](./.env.example) for the full list:

```dotenv
OPUS_MODE=paper
OPUS_MODEL_DIR=./models/global
OPUS_TRADE_HORIZON=5s
OPUS_TRADE_SYMBOLS=AIOTUSDT,BSBUSDT,ZKJUSDT  # decoupled from collection set
OPUS_TRADE_CONF_THRESHOLD=0.10               # |P(UP)-P(DOWN)| gate
OPUS_TRADE_NOTIONAL_USD=10.0
OPUS_TRADE_STOP_LOSS_BP=50.0
```

Then start the bot as usual (`python -m backend.main`). The paper trader:

1. Subscribes to the runtime snapshot loop (every 250 ms per symbol).
2. Buffers the last ~600 snapshots per symbol to recompute derived features.
3. Calls the LightGBM booster for the configured horizon.
4. If `|confidence| > threshold` AND no open position: opens long (conf>0)
   or short (conf<0) at the current ask/bid (taker assumption).
5. Holds until: horizon timer elapses, predictor flips sign, or stop-loss.
6. Computes net PnL = gross - 2 × `OPUS_TAKER_FEE_BP`.
7. Appends a record to `OPUS_DATA_DIR/trades/{YYYY-MM-DD}.parquet`. Each
   record carries the full prediction (probabilities + confidence) plus
   the realised entry / exit / PnL — useful as MVP-2 training input.

Live status (open positions, daily PnL, win rate) shows in the UI
under the **paper trades** card. The REST endpoints
`/api/trader` and `/api/predictor` expose the same data programmatically.

## Maker vs taker backtest

`python -m backend.ml.backtest` runs a vectorised replay on the test
slice of the labelled dataset and reports per-horizon, per-fee-scenario
PnL, fill rate, and a naive Sharpe. Scenarios:

- `taker` — pay 4 bp/side (default Binance Futures), always filled.
- `maker_best` — pay 2 bp/side, always filled (ceiling).
- `maker_zero` — 0 bp/side (VIP/rebated), always filled.
- `maker_cross` — pay 2 bp/side, only counts rows where the next tick
  crossed our limit price (proxy for queue/fill realism).

```powershell
.\.venv\Scripts\python.exe -m backend.ml.backtest `
    --models-dir models\global `
    --target-trade-frac 0.05
```

Output goes to `models/global/backtest/h{horizon}.json` plus a printed
summary table on stdout. Use this to decide whether the maker scenario
is profitable on each (symbol, horizon) pair before running the live
maker engine.

## Разработка

```bash
source .venv/bin/activate
ruff check .          # lint
ruff format .         # формат
pytest                # тесты (будут)
```

Перезапуск с auto-reload для разработки:

```bash
OPUS_RELOAD=1 python -m backend.main
```
