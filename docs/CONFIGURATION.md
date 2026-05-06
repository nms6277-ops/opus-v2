# Configuration

Every runtime knob in `opus-v2`. Two layers:

1. **`.env` (boot-time).** Loaded once by `backend/config.py` via Pydantic
   Settings. Values here become hard ceilings for anything the UI is
   allowed to set later. **Secrets and hard caps live here.**
2. **UI-editable runtime settings.** Persisted to
   `{OPUS_DATA_DIR}/runtime_settings.json` via `backend/settings_store.py`.
   The UI can lower values but never raise them above the corresponding
   `OPUS_HARD_*` ceiling.

This doc lists every knob, its default, valid range, and what happens
when you change it. Cross-references to the source are included so you
can find the implementation fast.

---

## 1. `.env` reference

Source: `backend/config.py` (class `Settings`) and `.env.example`.

All variables are prefixed with `OPUS_`. Names are case-insensitive on
the Pydantic side.

### 1.1 Mode and runtime profile

| Variable | Default | Range | Meaning |
|---|---|---|---|
| `OPUS_MODE` | `collect` | `collect` / `paper` / `live` | Initial top-level mode at startup. Switch live from the UI without a restart. |
| `OPUS_RUNTIME_PROFILE` | `local` | `local` / `vps_live` | `vps_live` keeps the server light (no raw tick logs, narrow watchlist, hard-cap-bounded UI). |

### 1.2 HTTP server

| Variable | Default | Notes |
|---|---|---|
| `OPUS_HOST` | `127.0.0.1` | Keep `127.0.0.1` and use SSH tunnel; use `0.0.0.0` only behind a trusted firewall. |
| `OPUS_PORT` | `8081` | (`README` shows `8080`; both are fine — match what your UI bookmarks expect.) |
| `OPUS_UI_USER` | empty | Optional HTTP Basic auth user. Leave both empty to disable auth. |
| `OPUS_UI_PASSWORD` | empty | Optional HTTP Basic auth password. |

### 1.3 Storage

| Variable | Default | Notes |
|---|---|---|
| `OPUS_DATA_DIR` | `./data` | Root for parquet snapshots, runtime model pointer, runtime settings. ≥30 GB recommended (≈300 MB/day after zstd × number of symbols). |
| `OPUS_LOGS_DIR` | `./logs` | Runtime logs. Rotated by the OS. |
| `OPUS_PARQUET_ROTATION_MIN` | `60` | Parquet rotation interval in minutes. 60 = one file per hour per symbol. Lower = smaller files but more file overhead. |

### 1.4 Collector

| Variable | Default | Range | Notes |
|---|---|---|---|
| `OPUS_SNAPSHOT_INTERVAL_MS` | `100` | ≥10 | v2 default 100 ms (was 250 ms). Higher rate → more training data on a short window; ≈2.4× disk vs 250 ms. |
| `OPUS_LOB_DEPTH` | `200` | ≥10 | Top-N levels held in memory per side. |
| `OPUS_SNAPSHOT_DEPTH` | `40` | 1..200 | Top-N levels written into each snapshot. |
| `OPUS_COLLECT_RAW_DEPTH` | `false` | bool | Tick-level depth log (every `depthUpdate`). ≈15-20 GB / 10 days / 5 symbols after zstd. |
| `OPUS_COLLECT_RAW_TRADES` | `false` | bool | Tick-level trade log. Auto-enabled when `feed_sdk_from_trade_log=true`. ≈10-30 GB / 10 days / 5 symbols. |

### 1.5 Feature engineering

| Variable | Default | Notes |
|---|---|---|
| `OPUS_BUCKET_BPS` | `5,10,25,50` | Cumulative ±bp aggregation buckets (legacy v1 columns). |
| `OPUS_BAND_BPS_MAX` | `50` | v2 wide-band buckets: half-width in bp around mid. `0` disables band columns. |
| `OPUS_BAND_BPS_STEP` | `5` | v2 band step in bp. `(50, 5)` → 20 bid + 20 ask non-cumulative slices. |
| `OPUS_ENABLE_ADAPTIVE_SDK` | `true` | Feed every Binance trade tick into `backend.adaptive_sdk`; sample VPIN / flow Z-scores / realized vol into `sdk_*` columns at each snapshot. |

### 1.6 Exchanges

| Variable | Default | Notes |
|---|---|---|
| `OPUS_BINANCE_WS_PUBLIC` | `wss://fstream.binance.com/public/stream` | High-rate depth feed. |
| `OPUS_BINANCE_WS_MARKET` | `wss://fstream.binance.com/market/stream` | Trades, ticker, kline, markPrice. |
| `OPUS_BINANCE_WS_PRIVATE` | `wss://fstream.binance.com/private/ws` | User-data stream (orders/account); only relevant in `live` mode. |
| `OPUS_BINANCE_WS` | `wss://fstream.binance.com/stream` | Legacy combined URL; unused when `_PUBLIC` and `_MARKET` are set. |
| `OPUS_BINANCE_REST` | `https://fapi.binance.com` | REST base URL. |
| `OPUS_BINANCE_CONNECT_TIMEOUT_S` | `15` | TCP connect timeout for Binance REST/WS. |
| `OPUS_BINANCE_WS_OPEN_TIMEOUT_S` | `15` | WS open timeout. |
| `OPUS_BYBIT_WS` | `wss://stream.bybit.com/v5/public/linear` | Read-only Bybit feed; used only when `OPUS_ENABLE_BYBIT=true`. |
| `OPUS_BYBIT_REST` | `https://api.bybit.com` | (currently unused) |
| `OPUS_ENABLE_BYBIT` | `false` | Toggle Bybit cross-exchange feed. |

### 1.7 Binance API keys (required only for `live`)

| Variable | Default | Notes |
|---|---|---|
| `OPUS_BINANCE_API_KEY` | empty | **Generate with futures-trading scope only**, no withdrawals, IP-whitelist your VPS. Never commit. |
| `OPUS_BINANCE_API_SECRET` | empty | See above. |

### 1.8 Soft safety guards (initial values; UI may lower)

These also exist on `RuntimeSettings`; the `.env` value is the **starting
point** for the runtime guards. The UI may lower them; it cannot raise
them above the matching `OPUS_HARD_*` cap.

| Variable | Default | Range | Meaning |
|---|---|---|---|
| `OPUS_DAILY_LOSS_LIMIT_USD` | `5.0` | ≥0 | Stop trading for the current UTC day when realised PnL ≤ -value. |
| `OPUS_MAX_POSITION_USD` | `50.0` | ≥0 | Max notional per symbol. |
| `OPUS_MAX_LIVE_SYMBOLS` | `1` | ≥1 | Max concurrent live symbols. |
| `OPUS_MAX_ORDERS_PER_MIN` | `60` | ≥1 | Runaway-protection rate limit. |
| `OPUS_WS_STALE_MS` | `1500` | ≥500 | If Binance WS has no message for this many ms, cancel-all + flatten. |

### 1.9 Hard caps (UI cannot exceed these)

These are **boot-time** ceilings applied by `state.Guards.__post_init__`
via `min(soft, hard)`, so even a fresh boot with no UI interaction
respects the envelope.

| Variable | Default | Range | Notes |
|---|---|---|---|
| `OPUS_HARD_DAILY_LOSS_USD` | `2.0` | ≥0 | Daily realised PnL ceiling. |
| `OPUS_HARD_12H_LOSS_USD` | `2.0` | ≥0 | Sliding-12h realised PnL ceiling. |
| `OPUS_HARD_SYMBOL_LOSS_USD` | `0.30` | ≥0 | Per-symbol 12h realised PnL ceiling. |
| `OPUS_HARD_MAX_LIVE_SYMBOLS` | `5` | ≥1 | Per-symbol concurrent ceiling. |
| `OPUS_HARD_MAX_NOTIONAL_USD` | `20.0` | ≥0 | Per-symbol notional ceiling. |
| `OPUS_HARD_MAX_LEVERAGE` | `10` | ≥1 | Per-symbol leverage ceiling sent to Binance. |
| `OPUS_HARD_MAX_ORDERS_PER_MIN` | `120` | ≥1 | Order rate ceiling. |

### 1.10 ML / paper trader

| Variable | Default | Notes |
|---|---|---|
| `OPUS_MODEL_DIR` | `./models/global` | Directory containing `h{horizon}/{model.lgb, meta.json}` subfolders. The runtime model pointer in `data_dir/runtime_model.json` overrides this when set. |
| `OPUS_TRADE_HORIZON` | `5s` | Which trained horizon drives live predictions. Must exist as `{model_dir}/h{horizon}/`. |
| `OPUS_TRADE_SYMBOLS` | empty | Comma-separated whitelist of trade-allowed symbols (subset of subscribed symbols). Empty = trade any subscribed symbol. |
| `OPUS_TRAIN_SYMBOL_BLOCKLIST` | `BTCUSDT,ETHUSDT` | Symbols **excluded** from training by default (still collectable). BTC/ETH have <1 bp spreads and would distort spread-aware labelling. Override with `--include-symbols` on the train CLI. |
| `OPUS_TRADE_CONF_THRESHOLD` | `0.10` | Minimum `|P(UP) - P(DOWN)|` to fire a trade. Higher → fewer, higher-conviction trades. Typical sweep `0.10 → 0.50`. |
| `OPUS_TRADE_NOTIONAL_USD` | `10.0` | Notional per paper trade. PnL scales linearly; per-trade bp is unchanged. |
| `OPUS_TAKER_FEE_BP` | `4.0` | Per-side taker fee in bp. Binance Futures default. Round-trip = 8 bp. |
| `OPUS_MAKER_FEE_BP` | `2.0` | Per-side maker fee. Negative for VIP / rebated accounts. |
| `OPUS_TRADE_STOP_LOSS_BP` | `50.0` | Per-trade stop-loss in bp against open position. |
| `OPUS_TRADE_TAKE_PROFIT_BP` | `0.0` | Per-trade take-profit in bp. `0` disables; non-zero arms a reduce-only `TAKE_PROFIT_MARKET` order on Binance so the exit fires even if the bot is offline. |

### 1.11 Telegram notifications (optional)

| Variable | Default | Notes |
|---|---|---|
| `OPUS_TELEGRAM_BOT_TOKEN` | empty | If both this and the chat id are set, the runtime sends mode changes / kill-switch / serious errors to Telegram. |
| `OPUS_TELEGRAM_CHAT_ID` | empty | Numeric chat id (negative for groups). |

---

## 2. UI-editable runtime settings

Source: `backend/settings_store.py` (`RuntimeSettings`).

These are persisted to `{OPUS_DATA_DIR}/runtime_settings.json` and edited
through `POST /api/settings`. The store clamps every field against
`SettingsHardCaps` built from `.env`. Default boot values use the same
hard cap (so out of the box the soft = hard).

| Field | Default | Hard cap source | Meaning |
|---|---|---|---|
| `leverage` | `10` | `OPUS_HARD_MAX_LEVERAGE` | Per-symbol leverage on Binance. |
| `max_live_symbols` | `5` | `OPUS_HARD_MAX_LIVE_SYMBOLS` | Concurrent live symbols. |
| `daily_loss_limit_usd` | `2.0` | `OPUS_HARD_DAILY_LOSS_USD` | Trading stops for the UTC day below this. |
| `loss_12h_limit_usd` | `2.0` | `OPUS_HARD_12H_LOSS_USD` | Trading stops on a 12h sliding window. |
| `symbol_loss_limit_usd` | `0.30` | `OPUS_HARD_SYMBOL_LOSS_USD` | Per-symbol 12h pause. |
| `probation_notional_usd` | `6.0` | `OPUS_HARD_MAX_NOTIONAL_USD` | Notional on `probation_live` state. |
| `active_notional_usd` | `20.0` | `OPUS_HARD_MAX_NOTIONAL_USD` | Notional on `active_live` state. |
| `min_expected_gross_bp` | `12.0` | — | Reject signals whose expected gross bp is below this; protects against thin-edge trades. |
| `global_profit_giveback_pct` | `0.30` | — | Pause all trading when total realised PnL drawdown from peak ≥ this fraction. |
| `symbol_profit_giveback_pct` | `0.30` | — | Same, per symbol. |
| `loss_streak_limit` | `4` | — | Consecutive losses on a symbol triggers a pause. |
| `rolling_guard_trades` | `15` | — | Window size for rolling regime checks. |
| `rolling_min_win_rate` | `0.35` | — | If rolling win rate drops below this on the window, pause. |
| `rolling_min_loss_net_bp` | `50.0` | — | If sum of rolling net bp ≤ -this, pause. |
| `rolling_min_drawdown_pct` | `0.10` | — | Symbol drawdown trigger. |
| `global_guard_min_trades` | `20` | — | Don't apply rolling guards globally below this trade count. |
| `symbol_guard_min_trades` | `10` | — | Same per symbol. |
| `probation_trades` | `7` | — | After this many positive `probation_live` trades a symbol is promoted to `active_live`. |
| `cooldown_hours` | `12` | — | Cooldown duration after a regime guard pause. |

---

## 3. Per-symbol live state machine

Each symbol has an `execution_mode` (`paper` or `live`) and a
`live_state` driven by `safety/regime.py`:

| State | Notional | Triggered by | Promoted to |
|---|---|---|---|
| `paper` | virtual | default for new symbols, also after `cooldown_hours` of a regime trip | `probation_live` (operator action in UI) |
| `probation_live` | `probation_notional_usd` | operator promotes a paper symbol that has accumulated edge | `active_live` after `probation_trades` positive trades |
| `active_live` | `active_notional_usd` | promotion from `probation_live` | back to `paper` after a regime guard trips |

A regime trip is any of: drawdown breach, loss streak, rolling win rate
floor, rolling net bp floor, daily/12h/symbol loss caps. The guard
records a `RiskEvent` and starts a `cooldown_hours` timer; trading
resumes in `paper` once the cooldown ends.

---

## 4. CLI knobs (training / backtest)

`backend/ml/train.py`:

```
python -m backend.ml.train \
    --symbols LABUSDT UBUSDT ... \
    --mode global|per-symbol \
    --window 24h \
    --models-dir models_100msSDK \
    --horizons 2s 5s 15s \
    --optuna-trials 0
```

| Flag | Default | Notes |
|---|---|---|
| `--symbols` | empty (all discovered minus `train_symbol_blocklist`) | Whitelist symbols to train on. |
| `--include-symbols` | `false` | Override the blocklist. |
| `--mode` | `global` | `global` adds `symbol` as a categorical feature; `per-symbol` trains independent models. |
| `--window` | `24h` | Time slice off the tail of the parquet stream. Accepts `h`/`d`. |
| `--models-dir` | `./models/global` | Output directory. Per-symbol mode writes `{models_dir}/per_symbol/{SYM}/h{H}/...`. |
| `--horizons` | `2s 5s 15s` | List of label horizons to train. |
| `--cost-mode` | `taker` | Label generation mode (`mid` for v1 back-compat). |
| `--optuna-trials` | `0` | Optuna hyperparameter search trials. `0` = use the fixed config. |

`backend/ml/backtest.py`:

```
python -m backend.ml.backtest \
    --symbols LABUSDT \
    --window 1h \
    --models-dir models_100msSDK/per_symbol/LABUSDT \
    --target-trade-frac 0.05
```

| Flag | Default | Notes |
|---|---|---|
| `--symbols` | empty (all in models dir) | Subset to evaluate. |
| `--window` | full test slice | Tail window (e.g. `1h`, `6h`) to evaluate; useful for forward-test passes. |
| `--models-dir` | `./models/global` | Where to find `h{H}/model.lgb`. |
| `--horizons` | autodetect | Infer from subfolders if omitted. |
| `--target-trade-frac` | `0.05` | Top-N threshold so only ~5% of rows fire trades. |

The summary table (`taker / maker_best / maker_zero / maker_cross` ×
horizons) is printed at the end of `main()` and dumped as JSON to
`{models_dir}/backtest/summary.json`. **Maker numbers are theoretical
upper bounds**; only the `taker` line corresponds to what `paper.py` /
`live.py` actually execute (see `RESULTS.md` and `ROADMAP.md` for why).

---

## 5. Recommended starting configurations

### 5.1 Local dev box (collect-only, full data)

```dotenv
OPUS_MODE=collect
OPUS_RUNTIME_PROFILE=local
OPUS_HOST=127.0.0.1
OPUS_PORT=8080
OPUS_DATA_DIR=./data
OPUS_LOGS_DIR=./logs
OPUS_COLLECT_RAW_DEPTH=true
OPUS_COLLECT_RAW_TRADES=true
OPUS_ENABLE_ADAPTIVE_SDK=true
```

### 5.2 VPS production (live or paper)

```dotenv
OPUS_MODE=collect            # promote to paper / live from the UI
OPUS_RUNTIME_PROFILE=vps_live
OPUS_HOST=127.0.0.1
OPUS_PORT=8081
OPUS_COLLECT_RAW_DEPTH=false
OPUS_COLLECT_RAW_TRADES=false

OPUS_BINANCE_API_KEY=...
OPUS_BINANCE_API_SECRET=...

OPUS_UI_USER=opus
OPUS_UI_PASSWORD=...           # pwgen 24 1

OPUS_HARD_DAILY_LOSS_USD=2.0
OPUS_HARD_12H_LOSS_USD=2.0
OPUS_HARD_SYMBOL_LOSS_USD=0.30
OPUS_HARD_MAX_LIVE_SYMBOLS=5
OPUS_HARD_MAX_NOTIONAL_USD=20.0
OPUS_HARD_MAX_LEVERAGE=10

OPUS_TELEGRAM_BOT_TOKEN=...
OPUS_TELEGRAM_CHAT_ID=...
```

Reach the UI through SSH tunnel: `ssh -L 8081:127.0.0.1:8081 vps`.

### 5.3 Training rig (Windows, big data)

```dotenv
OPUS_MODE=collect
OPUS_DATA_DIR=D:/opus-data
OPUS_LOGS_DIR=D:/opus-logs
OPUS_ENABLE_ADAPTIVE_SDK=true
OPUS_BAND_BPS_MAX=50
OPUS_BAND_BPS_STEP=5
```

Pagefile recommended ≥16 GB initial / 32 GB max for 16 GB RAM machines —
training a 24 h × 7 symbol dataset peaks at ~1.5–2 GB on the X matrix
(see `docs/RESULTS.md` §2.2 for the memory fix history).

---

## 6. Where the values live on disk

| Type | Path | Touched by |
|---|---|---|
| `.env` (boot config) | `{repo}/.env` | `backend/config.py::Settings` |
| Runtime UI settings | `{OPUS_DATA_DIR}/runtime_settings.json` | `backend/settings_store.py` |
| Active model bundle pointer | `{OPUS_DATA_DIR}/runtime_model.json` | `backend/runtime.py` |
| Live state checkpoint | `{OPUS_DATA_DIR}/live_state.json` | `backend/traders/live_state.py` |
| Snapshots | `{OPUS_DATA_DIR}/snapshots/{SYM}/{YYYY-MM-DD}/{HH}.parquet` | `backend/collector/writer.py` |
| Paper trade log | `{OPUS_DATA_DIR}/trades/{YYYY-MM-DD}.parquet` | `backend/collector/trade_log.py` |
| Live trade log | `{OPUS_DATA_DIR}/live_trades/{YYYY-MM-DD}.parquet` | `backend/collector/live_trade_log.py` |
| Models | `{model_dir}/h{H}/{model.lgb,meta.json,eval.json}` | `backend/ml/train.py` |
