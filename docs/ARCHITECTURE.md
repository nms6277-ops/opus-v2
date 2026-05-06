# Architecture

This document is a navigable map of `opus-v2` — what each module does, how
data flows through the system, and which invariants are load-bearing. It
is the answer to "where would I look to change X?" and a primer for
anyone (including future-you after a long pause) coming back to the code.

---

## 1. One-paragraph summary

`opus-v2` is a microstructure trading bot for Binance USDT-M Futures. A
24/7 collector ingests order-book diffs and aggressor trades over
WebSocket, reconstructs the LOB, and writes pre-aggregated 100 ms
feature snapshots to parquet. An offline ML pipeline (LightGBM) trains
short-horizon direction classifiers from those snapshots. At runtime,
the same snapshot loop feeds a paper trader (virtual fills) or a live
trader (real Binance orders), both gated by a multi-layer safety
system. A FastAPI process exposes a small REST + WebSocket API for a
local-only HTML UI. The whole system is sized for a 2 GB RAM / 2 vCPU
VPS — the realtime hot path uses pure numpy/struct, no pandas.

---

## 2. Repo layout

```
backend/
  adaptive_sdk/      # VPIN engine, exhaustion detector, Thompson MAB
  api/               # FastAPI: REST endpoints + WebSocket fanout
  cli/               # CLI entry points used by scripts/opus.bat
  collector/         # LOB reconstruction, parquet writer, trade logs
  exchanges/         # Binance public/private WS, REST, filters; Bybit WS
  features/          # Compact snapshot dataclass + serialisation
  ml/                # Dataset / labels / features / train / inference / backtest / eval
  safety/            # Pre-trade guards + post-trade regime guards
  traders/           # Paper trader + Live trader on a shared base
  config.py          # Pydantic Settings loaded from .env
  log.py             # Logger factory
  main.py            # FastAPI entry point ("python -m backend.main")
  model_registry.py  # Discover trained model bundles on disk
  runtime.py         # Singleton orchestrator (the central nervous system)
  settings_store.py  # UI-editable runtime settings, persisted to data dir
  state.py           # AppState: process-wide state container for the UI
  telegram.py        # Optional alerts
docs/                # ← you are here
frontend/            # Static HTML/CSS/JS for the local UI (no framework)
scripts/             # install_*.sh / install_*.ps1, opus.bat launcher
systemd/             # opus@user.service for VPS autostart
tests/               # pytest suite (lots of regression tests)
data/                # Parquet snapshots (gitignored; configurable via OPUS_DATA_DIR)
logs/                # Runtime logs (gitignored)
models/              # Trained ML artefacts (gitignored)
```

---

## 3. Top-level data flow

```
+---------------------+              +---------------------+
| Binance Futures WS  |              | Bybit WS (read-only |
|  /public  /market   |              |  cross-exchange)    |
|  /private (orders)  |              +----------+----------+
+----------+----------+                         |
           |                                    |
           v                                    v
+--------------------+    aggTrade   +--------------------+
|   collector/lob    +-------------->|  adaptive_sdk      |
|   OrderBook (top-N)|               |  VPIN, exhaustion  |
+----------+---------+               +----------+---------+
           |                                    |
           |   periodic snapshot every 100 ms   |
           v                                    v
+----------+------------------------------------+----------+
|         features/snapshot.py (Snapshot dataclass)        |
|     book stats + bands + sdk_*  →  parquet rows          |
+----+-----------+--------------------------------+--------+
     |           |                                |
     |           |  on every fresh snapshot       |  every N rotation_min
     v           v                                v
+------+   +-----+---------+              +-------+----------+
|state |   | trader        |              | collector/writer |
|.py   |<--+ paper or live |              | parquet on disk  |
+--+---+   +-----+---------+              +-------+----------+
   ^             |                                |
   |             |  predict via Predictor         |
   |             v                                v
   |       +-----+---------+              +-------+----------+
   |       |  ml/inference |              |   ml/dataset     |
   |       |  Predictor    |              |   parquet → DF   |
   |       +-----+---------+              +-------+----------+
   |             ^                                |
   |             |  loaded from disk              v
   |             |                          +-----+----------+
   |       +-----+---------+                | ml/features    |
   |       |  models/...   |                | ml/labels      |
   |       |  *.lgb + meta |                | ml/train (LGB) |
   |       +---------------+                | ml/backtest    |
   |                                        +----------------+
   v
+---+----+      +----------+      +--------+
|  api   |<-----+ runtime  +----->|  state |
| REST/WS|      | (singleton)     +--------+
+--------+      +----------+
     ^
     |
+----+------+
|  UI       |
| (browser) |
+-----------+
```

---

## 4. Module reference

### 4.1 `backend.collector`
*Owns: market data ingestion + persistence.*

- `collector/lob.py` — `OrderBook` reconstructs top-N bids/asks from a
  Binance USDT-M Futures `@depth@100ms` stream. The futures sync rule
  is **different** from spot (`pu == prev.u`, not `U == prev.u + 1`)
  and is documented at the top of the file. Detects sequence gaps and
  triggers a REST resync.
- `collector/writer.py` — `SnapshotWriter` rotates one parquet per hour
  per symbol at `{data_dir}/snapshots/{SYMBOL}/{YYYY-MM-DD}/{HH}.parquet`,
  zstd-compressed.
- `collector/trade_log.py` — append-only parquet of paper-trade outcomes.
- `collector/live_trade_log.py` — same shape for live trades.

Guarantees:
- Snapshots are append-only by `ts_ms`. Sequence numbers are not exposed
  to the writer; only the LOB cares about them.
- The collector NEVER calls `pandas`. Hot path is dataclasses → orjson →
  bytes for the WS layer, then a tight per-symbol struct in `OrderBook`.

### 4.2 `backend.exchanges`
*Owns: exchange protocol details.*

- `binance_ws.py` — public/market WS clients. Auto-reconnect with
  exponential backoff, sequence-gap detection, idle-message watchdog
  (`OPUS_WS_STALE_MS`). Two separate sockets: `/public` for high-rate
  depth, `/market` for aggTrade + ticker (Binance's own recommendation).
- `binance_private_ws.py` — user-data stream (account/order updates).
- `binance_rest.py` — REST client used for: snapshot fetch, leverage
  setup, exchangeInfo (precision filters), order placement. Quote and
  qty are formatted via `Decimal` to avoid scientific notation that
  Binance rejects.
- `binance_filters.py` — symbol precision (price tick, lot step, min
  notional) and validators used before every order.
- `bybit_ws.py` — read-only cross-exchange feed; behind
  `OPUS_ENABLE_BYBIT`. Currently surfaces top-of-book to the runtime
  but is not used for trading decisions.

### 4.3 `backend.features.snapshot`
*Owns: the snapshot row schema.*

A single `Snapshot` dataclass with ~100 numeric fields. Composed of:
- top-of-book (`best_bid/ask`, qty, microprice, imbalance ratios at
  top1/top5/top20),
- depth aggregates (`bid_vol_top5/20`, weighted depth proxies),
- legacy cumulative buckets `bid_bkt_qty_{5,10,25,50}bp`,
- v2 wide-band buckets `band_bid_qty_offX_offY` / `band_ask_qty_offX_offY`
  (default ±50 bp every 5 bp = 20 bid + 20 ask columns; these are the
  features the v2 ML uses),
- optional `sdk_*` columns from `adaptive_sdk` when enabled.

The legacy absolute-price columns (`bid_p_NN`, `ask_p_NN`) are still
written for backward compat but the v2 feature selector drops them — a
model trained on bands generalises across symbols of any tick size.

### 4.4 `backend.adaptive_sdk`
*Owns: trade-flow features and stateful triggers.*

A small vendored library that consumes per-trade ticks and exposes:
- `TrueVPINEngine` — Volume-synchronised PIN (toxic flow indicator).
- `ExhaustionDetector` — detects rolling extremes (Z-scores of buy/sell
  flow).
- `ThompsonSamplingMAB` — explore/exploit on Z-thresholds.

The runtime calls `sdk.on_trade(tick)` from the trade stream and
`sdk.on_book_snapshot(book)` from the snapshot loop, then samples the
state into `sdk_vpin`, `sdk_buy_flow_z`, `sdk_sell_flow_z`,
`sdk_realized_vol`, `sdk_buckets_filled`, `sdk_pending_signals`,
`sdk_is_ready`. These are present in every parquet row when
`OPUS_ENABLE_ADAPTIVE_SDK=true`.

### 4.5 `backend.ml`
*Owns: offline training and online inference.*

- `dataset.py` — discovers + loads parquet snapshots, returns a single
  Polars DataFrame sorted by `(symbol, ts_ms)` with a `part` column
  (`train`/`val`/`test`) added by **time-based** quantile split (no
  shuffling — mandatory in microstructure ML to avoid look-ahead).
- `features.py` — derives rolling features from the raw snapshot
  columns: lag returns, rolling vol, OFI proxies, bucket imbalances.
- `labels.py` — adds 3-class direction labels per horizon. **Spread-aware**
  (`cost_mode="taker"`, default): a row is `UP` only if the long
  round-trip P&L (buy at offer, sell at bid one horizon later) beats a
  symbol-specific threshold = mean spread + commission + safety margin.
  Most rows that look like signal under a naive mid-to-mid label
  collapse to `FLAT` here, and that is the point.
- `train.py` — trains one LightGBM multi-class classifier per horizon.
  `--mode global` (default) adds `symbol` as a categorical feature;
  `--mode per-symbol` trains independent models. Memory is carefully
  managed: the X matrix is built column-by-column into a pre-allocated
  C-order float32 buffer to avoid the ~2× transient peak from Polars'
  Fortran-ordered `to_numpy` + `np.ascontiguousarray` path.
- `inference.py` — `Predictor` loads a model bundle and serves
  predictions one snapshot at a time. Maintains a per-symbol numpy ring
  buffer to recompute derived features online without rebuilding the
  whole DataFrame each tick.
- `backtest.py` — vectorised replay of a model on its `part=="test"`
  slice. Reports `taker`, `maker_best`, `maker_zero`, `maker_cross`
  scenarios. **Important caveat**: maker numbers are theoretical
  upper bounds — queue position, partial fills, cancellations and
  adverse selection are NOT modelled (see ROADMAP.md §3 and RESULTS.md).
- `eval.py` — confusion matrix / per-class precision-recall / per-symbol
  breakdowns dumped as JSON next to the model.

### 4.6 `backend.traders`
*Owns: order generation.*

- `traders/base.py` — minimal `Trader` interface (`on_snapshot`, `start`,
  `stop`, `flatten_all`).
- `traders/paper.py` — virtual market-taker: on every fresh snapshot
  asks the predictor; if `|conf| > threshold` and no open position,
  opens at the offer/bid; closes on horizon timeout, opposite signal
  flip, stop-loss, or runtime stop. PnL is mid-to-mid at exit minus
  taker fee × 2.
- `traders/live.py` — real Binance orders. Same prediction loop as
  paper, but goes through `binance_rest` MARKET orders, sets up
  leverage per symbol, listens to private-WS for fills, manages
  reduce-only TP/SL orders, dedupes private-trade replays on reconnect.
  Lives behind every safety guard in §4.7.
- `traders/live_state.py` — small dataclass + serialiser for the
  per-symbol live state machine (`paper / probation_live / active_live`).

### 4.7 `backend.safety`
*Owns: pre-trade and post-trade circuit breakers.*

- `safety/guards.py` — `check(intent)` returns a non-empty reason to
  reject, otherwise `""`. Enforces: daily loss limit, 12h loss limit,
  per-symbol notional cap, per-symbol 12h loss cap, max live symbols,
  max orders/minute, network kill-switch (no WS message for
  `OPUS_WS_STALE_MS`). Also tracks `pnl_events` (12 h sliding window).
- `safety/regime.py` — runs after every closed trade. Pauses a symbol
  (or all symbols) if the recent stream is no longer behaving like the
  validated paper/backtest regime: rolling win rate too low, rolling
  net bp too negative, drawdown above limit, loss streak above limit.
  The validated paper/backtest regime is the implicit baseline; if a
  live cohort drifts away, this is the layer that pulls the plug.

### 4.8 `backend.runtime`
*Owns: lifecycle and orchestration.*

A singleton that wires everything together:
- holds per-symbol `OrderBook`, `SnapshotWriter`, snapshot task;
- owns the two Binance WS clients (public + market), the optional
  private WS, the optional Bybit WS, the optional `AdaptiveAnalyticsSDK`;
- selects an active trader (`PaperTrader` or `LiveTrader`) based on
  `Mode`; switches it on mode flip;
- owns the `Predictor` and re-loads it when the operator picks a new
  model bundle from the UI;
- propagates settings changes from `SettingsStore` into `Guards` and
  the live trader without restart;
- is the only component that **mutates** per-symbol runtime state.

The API/UI talks to the runtime; the runtime mutates `AppState`; the
WebSocket fanout in `backend.api.ws` snapshots `AppState` on every tick.

### 4.9 `backend.api`
*Owns: HTTP/WS surface.*

- `api/rest.py` — endpoints for: status, watchlist add/remove, mode
  change, model bundle pick, settings update, manual flatten, trade log
  query.
- `api/ws.py` — pushes `AppState` snapshots to connected UIs at a
  bounded cadence.
- `api/auth.py` — optional HTTP Basic via `OPUS_UI_USER` /
  `OPUS_UI_PASSWORD`.

### 4.10 `backend.state`
*Owns: process-wide UI-visible state.*

- `AppState` — singleton instance `app_state`. Holds a `Mode`, a dict
  of `SymbolStats`, a `Guards` dataclass instance, and a few global
  fields (last error, daily PnL, etc.).
- `SymbolStats` — per-symbol row shown in the UI: best bid/ask, current
  position, fills, rejects, block reason, live state, recent trade net
  bp, drawdown, rolling win rate.
- `Guards` — initialised from `Settings` and clamped to hard caps so
  even a fresh boot with no UI interaction respects the hard envelope.

### 4.11 `backend.settings_store`
*Owns: persisted UI-editable trading settings.*

`RuntimeSettings` is the dataclass that the UI POSTs into. The store
clamps every field against `SettingsHardCaps` (built from `.env`'s
`OPUS_HARD_*`) and persists to `{data_dir}/runtime_settings.json`.

### 4.12 `backend.cli`
*Owns: developer-friendly CLI wrappers (`scripts/opus.bat` calls these).*

- `cli/active.py` — bind one trained model bundle as the live runtime
  model; persisted to `{data_dir}/runtime_model.json`.

---

## 5. Three execution modes

The runtime supports three modes, switched live from the UI without a
restart:

| Mode | Collector | Predictor | Trader | Order routing |
|---|---|---|---|---|
| **collect** | yes | no | no | — |
| **paper** | yes | yes | `PaperTrader` | virtual fills only |
| **live** | yes | yes | `LiveTrader` | real Binance orders |

`live` requires `OPUS_BINANCE_API_KEY` / `OPUS_BINANCE_API_SECRET` and a
live-state machine (`paper → probation_live → active_live`) per symbol.
A symbol can only be promoted from `probation_live` to `active_live`
after `probation_trades` successful trades — set in `RuntimeSettings`.

---

## 6. Hot-path constraints

These are load-bearing and worth respecting in any future change:

1. **No pandas in the realtime loop.** Every 100 ms snapshot, every
   trade tick, every WS message lives in pure numpy / stdlib / orjson.
   pandas is only allowed offline (training, backtest, eval). Polars is
   allowed for batched offline reads.
2. **No blocking I/O in the snapshot path.** Disk writes go through a
   bounded queue inside `SnapshotWriter`; if the queue is full we drop
   the oldest unflushed row rather than block.
3. **The runtime is a singleton.** All mutations of `app_state`,
   `Guards`, per-symbol `OrderBook`, and active trader happen here. If
   you reach for a global from elsewhere, restructure.
4. **Time-based splits only.** `dataset.load_dataset` never shuffles.
   Look-ahead bias kills microstructure ML. The `part` column is
   mandatory.
5. **Memory cap.** The VPS target is 2 GB. The training memory fix in
   `ml/train._xy` is there because we OOM'd on Windows when Polars
   returned a Fortran-ordered view and `ascontiguousarray` allocated a
   second copy. Stay column-by-column on big matrices.

---

## 7. Where to look for X

| Want to change… | Look at… |
|---|---|
| What columns get written to parquet | `backend/features/snapshot.py` |
| How labels are defined | `backend/ml/labels.py` |
| Which features go into LightGBM | `backend/ml/features.py` (see `feature_columns`) |
| Training hyperparameters | `backend/ml/train.py` |
| Backtest scenarios | `backend/ml/backtest.py` (`SCENARIOS` tuple) |
| Pre-trade rejections | `backend/safety/guards.py` |
| Post-trade circuit breaker | `backend/safety/regime.py` |
| Mode transition (collect → paper → live) | `backend/runtime.py::Runtime.set_mode` |
| What the UI sees | `backend/state.py::SymbolStats.to_dict` and `backend/api/rest.py` |
| Settings the UI can edit | `backend/settings_store.py::RuntimeSettings` |
| Hard caps on those settings | `backend/config.py` (`hard_*` fields) |

Pair this document with `docs/CONFIGURATION.md` (every `.env` knob),
`docs/RESULTS.md` (what we have actually tested) and `docs/ROADMAP.md`
(what we have NOT tested but think might work).
