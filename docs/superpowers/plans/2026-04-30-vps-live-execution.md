# VPS Live Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a VPS-light `opus` runtime that can trade Binance USDT-M Futures live per symbol with strict risk limits, probation sizing, UI control, private WS reconciliation, and Telegram status.

**Architecture:** Keep the home server responsible for broad collection, model training, and backtests. The VPS runtime becomes a narrow execution agent: max 5 live symbols, no raw data logging, private Binance account stream, one live order path, persistent runtime settings, and a per-symbol state machine. Live is armed per symbol from the UI; new symbols always start in paper.

**Tech Stack:** Python 3.11+, FastAPI, asyncio, httpx, websockets, Polars/Parquet only for lightweight trade logs, static HTML/JS frontend, pytest/pytest-asyncio.

---

## Scope Notes

This plan implements the first VPS live release. It intentionally does not implement auto-discovery or weekly retraining. Those are later slices after live safety is proven.

Current checkout at `C:\codex\opus` is not a git repository. Commit steps are included for a normal git checkout, but should be skipped in this local copy unless `.git` exists.

Spec: `docs/superpowers/specs/2026-04-30-vps-live-execution-design.md`

---

## File Map

Create:

- `backend/settings_store.py` - persisted UI-editable runtime settings with `.env` hard-cap enforcement.
- `backend/exchanges/binance_private_ws.py` - Binance listenKey lifecycle and private user-data stream.
- `backend/exchanges/binance_filters.py` - exchange filter parsing, quantity/price rounding helpers.
- `backend/traders/live_state.py` - per-symbol live state machine and probation accounting.
- `backend/telegram.py` - Telegram bot polling, hourly status, `/status`.
- `tests/test_settings_store.py`
- `tests/test_live_state.py`
- `tests/test_binance_filters.py`
- `tests/test_guards_live_limits.py`
- `tests/test_ws_auth.py`
- `tests/test_live_trader_order_flow.py`

Modify:

- `backend/config.py` - VPS defaults, hard caps, Telegram settings, raw logging disabled by default.
- `.env.example` - document live/VPS variables.
- `backend/state.py` - symbol execution mode/state, global 12h/daily risk status, runtime settings snapshot.
- `backend/safety/guards.py` - correct max live symbols, 12h loss cap, per-symbol loss cap.
- `backend/collector/writer.py` - avoid read-concat-rewrite in VPS hot path or gate snapshot writing off for VPS.
- `backend/runtime.py` - wire settings store, private WS, per-symbol execution mode, managed-symbol reconciliation.
- `backend/exchanges/binance_rest.py` - leverage, account positions, exchange filters, cancel/flatten helpers.
- `backend/traders/live.py` - real live market-taker execution, reduce-only exits, emergency flatten.
- `backend/traders/paper.py` - share decision/risk hooks where useful; preserve existing paper behavior.
- `backend/api/rest.py` - runtime settings endpoints, per-symbol paper/live endpoint, emergency endpoints.
- `backend/api/ws.py` - auth for WebSocket.
- `frontend/index.html`, `frontend/app.js`, `frontend/style.css` - trading cockpit UI.
- `README.md` - VPS live deployment, SSH tunnel, safety warning.

---

### Task 1: VPS Config Defaults And Hard Caps

**Files:**
- Modify: `backend/config.py`
- Modify: `.env.example`
- Test: `tests/test_settings_store.py`

- [ ] **Step 1: Write failing tests for hard caps and defaults**

Create `tests/test_settings_store.py` with tests for:

```python
from backend.settings_store import RuntimeSettings, clamp_to_hard_caps


def test_runtime_settings_clamped_to_hard_caps():
    settings = RuntimeSettings(
        leverage=25,
        max_live_symbols=99,
        daily_loss_limit_usd=10.0,
        loss_12h_limit_usd=10.0,
        symbol_loss_limit_usd=5.0,
        probation_notional_usd=50.0,
        active_notional_usd=500.0,
    )
    capped = clamp_to_hard_caps(
        settings,
        hard_max_leverage=10,
        hard_max_live_symbols=5,
        hard_daily_loss_usd=2.0,
        hard_12h_loss_usd=2.0,
        hard_symbol_loss_usd=0.30,
        hard_notional_usd=20.0,
    )
    assert capped.leverage == 10
    assert capped.max_live_symbols == 5
    assert capped.daily_loss_limit_usd == 2.0
    assert capped.loss_12h_limit_usd == 2.0
    assert capped.symbol_loss_limit_usd == 0.30
    assert capped.probation_notional_usd == 20.0
    assert capped.active_notional_usd == 20.0
```

- [ ] **Step 2: Run test and verify it fails**

Run: `python -m pytest tests/test_settings_store.py -q`

Expected: FAIL because `backend.settings_store` does not exist.

- [ ] **Step 3: Implement `backend/settings_store.py`**

Add a `RuntimeSettings` dataclass with defaults:

- leverage `10`
- max live symbols `5`
- daily loss `$2`
- 12h loss `$2`
- symbol loss `$0.30`
- probation notional `$6`
- active notional `$20`
- probation trades `7`
- cooldown hours `12`

Add:

- `clamp_to_hard_caps(settings, ...) -> RuntimeSettings`
- `SettingsStore(path, hard_caps)` with `load()`, `save()`, `snapshot()`
- JSON persistence under `settings.data_dir / "runtime_settings.json"`

- [ ] **Step 4: Modify `backend/config.py`**

Add:

- `runtime_profile: str = "local"`
- `hard_daily_loss_usd: float = 2.0`
- `hard_12h_loss_usd: float = 2.0`
- `hard_symbol_loss_usd: float = 0.30`
- `hard_max_live_symbols: int = 5`
- `hard_max_notional_usd: float = 20.0`
- `hard_max_leverage: int = 10`
- `telegram_bot_token: str = ""`
- `telegram_chat_id: str = ""`

Change default:

- `collect_raw_depth: bool = False`
- `collect_raw_trades: bool = False`

- [ ] **Step 5: Update `.env.example`**

Add a VPS section with `OPUS_HOST=127.0.0.1`, `OPUS_PORT=8081`, hard caps, Telegram token/chat id, and `OPUS_COLLECT_RAW_DEPTH=false`.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_settings_store.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/config.py backend/settings_store.py .env.example tests/test_settings_store.py
git commit -m "feat: add vps runtime settings and hard caps"
```

---

### Task 2: WebSocket Auth And UI Binding Safety

**Files:**
- Modify: `backend/api/ws.py`
- Modify: `backend/api/rest.py` if shared auth helper is moved
- Test: `tests/test_ws_auth.py`

- [ ] **Step 1: Write failing WS auth tests**

Create tests that configure `settings.ui_user/settings.ui_password` and assert:

- `/ws` rejects unauthenticated connection;
- `/ws` accepts valid Basic auth;
- `/ws` remains open when auth is disabled.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_ws_auth.py -q`

Expected: FAIL because `/ws` currently accepts without auth.

- [ ] **Step 3: Extract shared auth helper**

Move REST auth verification into a reusable helper, for example:

- `backend/api/auth.py`
- `check_basic_auth_header(authorization: str | None) -> bool`

Keep REST behavior unchanged.

- [ ] **Step 4: Protect `/ws`**

In `backend/api/ws.py`, inspect `ws.headers.get("authorization")` before
`accept()`. If auth is configured and invalid, close with policy violation or
raise `WebSocketException`.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_ws_auth.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/api tests/test_ws_auth.py
git commit -m "fix: require auth for ui websocket"
```

---

### Task 3: Live State Machine

**Files:**
- Create: `backend/traders/live_state.py`
- Modify: `backend/state.py`
- Test: `tests/test_live_state.py`

- [ ] **Step 1: Write failing state-machine tests**

Cover:

- new symbol starts as `paper`;
- switching live starts `probation_live`;
- 7 trades with `wins >= 4` and positive net promotes to `active_live`;
- `wins <= 2`, `sum_net_bp < -20`, or `consecutive_losses >= 4` moves to cooldown;
- symbol loss `<= -0.30` disables/cools down symbol.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_live_state.py -q`

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement `live_state.py`**

Add enums:

- `ExecutionMode`: `paper`, `live`
- `LiveSymbolState`: `paper`, `watch_only`, `probation_live`, `active_live`, `cooldown`, `disabled`

Add dataclass:

- `LiveSymbolRuntime`
  - symbol
  - execution_mode
  - live_state
  - live_trade_count
  - wins
  - losses
  - consecutive_losses
  - realized_pnl_usd
  - sum_net_bp
  - cooldown_until
  - current_notional_usd

Add methods:

- `arm_live(settings)`
- `record_closed_trade(net_bp, pnl_usd, settings)`
- `can_trade(now, tradeability_ok)`
- `disable(reason)`

- [ ] **Step 4: Extend `SymbolStats`**

Add serializable fields to `backend/state.py`:

- `execution_mode`
- `live_state`
- `live_trade_count`
- `live_wins`
- `live_losses`
- `consecutive_losses`
- `symbol_realized_pnl_12h`
- `current_notional_usd`
- `cooldown_until`
- `block_reason`

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_live_state.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/traders/live_state.py backend/state.py tests/test_live_state.py
git commit -m "feat: add per-symbol live state machine"
```

---

### Task 4: Guard Fixes And 12h/Symbol Risk Caps

**Files:**
- Modify: `backend/safety/guards.py`
- Modify: `backend/state.py`
- Test: `tests/test_guards_live_limits.py`

- [ ] **Step 1: Write failing guard tests**

Cover:

- max live symbols counts symbols with `execution_mode=live` and state in `probation_live` or `active_live`;
- opening the 6th live symbol is rejected when max is 5;
- daily loss cap rejects;
- 12h loss cap rejects;
- symbol loss cap rejects only that symbol.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_guards_live_limits.py -q`

Expected: FAIL on current `fills_count` logic.

- [ ] **Step 3: Modify guards**

Change `check()` to accept or derive:

- managed symbol runtime state;
- current runtime settings;
- 12h rolling PnL;
- per-symbol PnL.

Replace `fills_count > 0` live-symbol counting with explicit live state.

- [ ] **Step 4: Add rolling 12h accounting**

Keep a small deque/list of closed live trade PnL events in memory:

- timestamp
- symbol
- pnl_usd

Use it for 12h cap and per-symbol cap. Persist closed trades separately.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_guards_live_limits.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/safety/guards.py backend/state.py tests/test_guards_live_limits.py
git commit -m "fix: enforce live symbol and rolling loss caps"
```

---

### Task 5: Binance Filters And REST Live Helpers

**Files:**
- Create: `backend/exchanges/binance_filters.py`
- Modify: `backend/exchanges/binance_rest.py`
- Test: `tests/test_binance_filters.py`

- [ ] **Step 1: Write failing filter tests**

Cover:

- quantity rounds down to step size;
- price rounds to tick size;
- min notional is enforced;
- invalid symbols fail closed.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_binance_filters.py -q`

Expected: FAIL because helpers do not exist.

- [ ] **Step 3: Implement filter parser**

Create:

- `SymbolFilters`
  - tick_size
  - step_size
  - min_qty
  - min_notional
  - price_precision
  - quantity_precision
- `parse_symbol_filters(exchange_info_symbol) -> SymbolFilters`
- `round_qty_down(qty, filters)`
- `round_price(price, filters)`
- `validate_notional(price, qty, filters)`

- [ ] **Step 4: Extend REST client**

In `binance_rest.py`, add:

- `server_time()`
- `set_leverage(symbol, leverage)`
- `position_risk(symbol=None)`
- `account()`
- `get_open_position_amt(symbol)`
- `market_close_position(symbol, position_amt, reduce_only=True)`
- `cached_symbol_filters(symbol)`

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_binance_filters.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/exchanges/binance_filters.py backend/exchanges/binance_rest.py tests/test_binance_filters.py
git commit -m "feat: add binance futures filters and live rest helpers"
```

---

### Task 6: Binance Private WebSocket

**Files:**
- Create: `backend/exchanges/binance_private_ws.py`
- Modify: `backend/runtime.py`
- Test: `tests/test_binance_private_ws.py`

- [ ] **Step 1: Write failing private WS tests**

Use fake REST/listenKey and fake websocket messages. Cover:

- creates listenKey on start;
- renews listenKey periodically;
- emits order update callback on `ORDER_TRADE_UPDATE`;
- emits account update callback on `ACCOUNT_UPDATE`;
- marks stale/disconnected on reconnect.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_binance_private_ws.py -q`

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement private WS client**

Add `BinancePrivateWS` with:

- `start()`
- `stop()`
- `_keepalive_loop()`
- `_read_loop()`
- callbacks `on_order_update`, `on_account_update`, `on_state`

Use Binance private stream URL from settings and listenKey lifecycle REST
helpers.

- [ ] **Step 4: Wire runtime state**

In `runtime.py`:

- construct private WS only when API keys exist;
- expose private connection health in `AppState`;
- live trading cannot arm if private WS is disabled/stale.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_binance_private_ws.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/exchanges/binance_private_ws.py backend/runtime.py tests/test_binance_private_ws.py
git commit -m "feat: add binance private websocket reconciliation"
```

---

### Task 7: LiveTrader Market Taker Execution

**Files:**
- Modify: `backend/traders/live.py`
- Modify: `backend/runtime.py`
- Modify: `backend/traders/base.py`
- Test: `tests/test_live_trader_order_flow.py`

- [ ] **Step 1: Write failing order-flow tests**

Use fake REST/private callbacks. Cover:

- live trader rejects when private WS is stale;
- sets leverage before first order per symbol;
- opens long at market with `OPUS_` client order id;
- closes with reduce-only order;
- emergency stop cancels and flattens managed symbols;
- does not trade symbols in `paper`, `watch_only`, `cooldown`, or `disabled`.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_live_trader_order_flow.py -q`

Expected: FAIL because `LiveTrader` is still a stub.

- [ ] **Step 3: Implement live decision entrypoint**

Mirror `PaperTrader.on_snapshot()` but route through live state:

- get predictor output;
- check symbol execution mode;
- check tradeability;
- check guards;
- determine notional `$6` or `$20`;
- submit market order;
- record pending order state until private WS confirms fill.

- [ ] **Step 4: Implement close logic**

Close on:

- horizon timeout;
- opposing signal;
- stop-loss;
- emergency;
- manual remove/disable.

Use reduce-only market exits.

- [ ] **Step 5: Implement reconciliation callbacks**

Private WS updates:

- fill entry price/qty;
- fill exit price/qty;
- compute realised pnl/net bp;
- update `LiveSymbolRuntime`;
- update risk PnL windows;
- write trade record.

- [ ] **Step 6: Implement flatten**

`LiveTrader.stop()` and emergency:

- cancel open orders;
- fetch open position for each managed symbol;
- send reduce-only market order for non-zero position;
- log success/failure;
- set runtime to waiting-for-user after global stop.

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_live_trader_order_flow.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/traders backend/runtime.py tests/test_live_trader_order_flow.py
git commit -m "feat: implement binance live market execution"
```

---

### Task 8: REST API For Runtime Settings And Per-Symbol Mode

**Files:**
- Modify: `backend/api/rest.py`
- Modify: `backend/runtime.py`
- Test: `tests/test_runtime_api.py`

- [ ] **Step 1: Write failing API tests**

Cover:

- `GET /api/settings`
- `POST /api/settings`
- `POST /api/watchlist/mode` with `{symbol, execution_mode}`
- cannot set UI values above hard caps;
- new symbols default to paper.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_runtime_api.py -q`

Expected: FAIL because endpoints do not exist.

- [ ] **Step 3: Implement endpoints**

Add:

- `GET /api/settings`
- `POST /api/settings`
- `POST /api/watchlist/mode`
- `POST /api/watchlist/disable`
- `POST /api/emergency/stop`

Keep existing endpoints backward compatible.

- [ ] **Step 4: Wire runtime**

Add runtime methods:

- `set_runtime_settings()`
- `set_symbol_execution_mode(symbol, mode)`
- `disable_symbol(symbol, reason)`
- `emergency_stop(reason)`

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_runtime_api.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/api/rest.py backend/runtime.py tests/test_runtime_api.py
git commit -m "feat: add runtime settings and symbol execution api"
```

---

### Task 9: Frontend Trading Cockpit

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/app.js`
- Modify: `frontend/style.css`

- [ ] **Step 1: Add UI controls**

For each symbol row, show:

- paper/live switch;
- live state badge;
- current notional;
- live trade count;
- wins/losses;
- consecutive losses;
- symbol PnL;
- tradeability/block reason.

- [ ] **Step 2: Add settings panel**

Controls:

- leverage;
- max live symbols;
- daily loss;
- 12h loss;
- symbol loss;
- probation notional;
- active notional;
- probation trades;
- cooldown hours.

- [ ] **Step 3: Add global live health panel**

Show:

- Binance public WS;
- Binance market WS;
- Binance private WS;
- API key configured/not configured;
- emergency state;
- managed open positions.

- [ ] **Step 4: Add actions**

Buttons:

- apply settings;
- switch symbol paper/live;
- disable symbol;
- emergency stop;
- clear emergency after manual review.

- [ ] **Step 5: Manual browser check**

Run backend locally, open `http://127.0.0.1:8081`, verify:

- controls fit on desktop;
- no text overlap;
- switching a symbol to live asks for confirmation;
- settings cannot exceed caps;
- websocket reconnect still updates UI.

- [ ] **Step 6: Commit**

```bash
git add frontend/index.html frontend/app.js frontend/style.css
git commit -m "feat: add vps live trading cockpit"
```

---

### Task 10: Telegram Status And Alerts

**Files:**
- Create: `backend/telegram.py`
- Modify: `backend/runtime.py`
- Modify: `backend/config.py`
- Test: `tests/test_telegram_status.py`

- [ ] **Step 1: Write failing Telegram tests**

Use fake HTTP client. Cover:

- hourly status sends summary;
- `/status` returns symbols, states, PnL, caps, WS health;
- emergency sends immediate alert;
- Telegram disabled when token/chat id missing.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_telegram_status.py -q`

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement Telegram client**

Use `httpx.AsyncClient`.

Add:

- `TelegramNotifier.start()`
- `stop()`
- `send_message()`
- `send_hourly_status()`
- `poll_commands()`
- `/status` handler.

- [ ] **Step 4: Wire runtime**

Start Telegram notifier during runtime startup if configured.

On events:

- symbol cooldown;
- global stop;
- private WS stale;
- flatten result.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_telegram_status.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/telegram.py backend/runtime.py backend/config.py tests/test_telegram_status.py
git commit -m "feat: add telegram live status notifications"
```

---

### Task 11: VPS Hot-Path Storage Cleanup

**Files:**
- Modify: `backend/collector/writer.py`
- Modify: `backend/runtime.py`
- Modify: `backend/config.py`
- Test: `tests/test_vps_storage_profile.py`

- [ ] **Step 1: Write failing storage-profile tests**

Cover:

- in `vps_live` profile, raw depth/trades are disabled;
- snapshot parquet writing can be disabled or reduced;
- trade log still writes closed trades;
- writer does not read-concat-rewrite in live hot path.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_vps_storage_profile.py -q`

Expected: FAIL until profile behavior exists.

- [ ] **Step 3: Implement profile gates**

In `runtime.py`, when `settings.runtime_profile == "vps_live"`:

- do not create raw writers;
- optionally do not create snapshot writer, or use lower-frequency decision logs;
- keep trade log enabled.

- [ ] **Step 4: Make writer mode explicit**

If snapshot writing remains enabled, add append-chunk rotation instead of
hourly read-concat-rewrite. Prefer writing separate chunk files:

`snapshots/{symbol}/{date}/{hour}-{chunk_id}.parquet`

This avoids rewriting existing files.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_vps_storage_profile.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/collector/writer.py backend/runtime.py backend/config.py tests/test_vps_storage_profile.py
git commit -m "fix: reduce vps live storage and writer load"
```

---

### Task 12: Documentation And Deployment Check

**Files:**
- Modify: `README.md`
- Create: `docs/vps-live-runbook.md`
- Modify: `systemd/opus.service`

- [ ] **Step 1: Write VPS runbook**

Document:

- install;
- `.env`;
- SSH tunnel;
- `OPUS_PORT=8081`;
- funding-sniper coexistence on `8080`;
- Binance API key permissions and IP whitelist;
- first live test procedure;
- emergency stop procedure;
- logs to inspect.

- [ ] **Step 2: Update systemd unit**

Ensure unit uses:

- correct working directory;
- `.venv`;
- restart policy;
- environment file if desired;
- bound host/port from `.env`.

- [ ] **Step 3: Run lint/tests**

Run:

```bash
python -m compileall -q backend
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Expected: all pass.

- [ ] **Step 4: Local smoke test**

Run:

```bash
OPUS_RUNTIME_PROFILE=vps_live OPUS_HOST=127.0.0.1 OPUS_PORT=8081 python -m backend.main
```

Verify:

- UI loads at `127.0.0.1:8081`;
- symbol can be added;
- default mode is paper;
- live switch refuses if API/private WS unavailable;
- emergency stop works without exception.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/vps-live-runbook.md systemd/opus.service
git commit -m "docs: add vps live deployment runbook"
```

---

## Execution Order

Recommended order:

1. Task 1 - settings/hard caps.
2. Task 2 - WS auth.
3. Task 3 - symbol state machine.
4. Task 4 - guards.
5. Task 5 - Binance filters/REST helpers.
6. Task 6 - private WS.
7. Task 7 - LiveTrader execution.
8. Task 8 - API.
9. Task 9 - UI.
10. Task 10 - Telegram.
11. Task 11 - storage cleanup.
12. Task 12 - docs/deployment.

Do not start live order submission before Tasks 1-6 pass.

---

## Acceptance Criteria

- VPS runtime can run on `127.0.0.1:8081`.
- No raw depth/trades are written by default.
- UI and WS auth are enforced when configured.
- Binance public, market, and private WS states are visible.
- A symbol added in UI starts in paper.
- Switching a symbol live starts probation with `$6` notional.
- Passing probation promotes to `$20` notional.
- Failing probation moves symbol to cooldown.
- Daily `$2`, 12h `$2`, and symbol `$0.30` caps stop trading.
- Emergency stop cancels orders and flattens managed Binance positions.
- Telegram sends hourly status and responds to `/status`.
- Max live symbols is enforced from explicit live state, not `fills_count`.
