# opus VPS Live Execution Design

Date: 2026-04-30

## Goal

Prepare `opus` for a first VPS live release on a 2 GB RAM / 2 vCPU server.
The VPS process must be able to trade Binance USDT-M Futures live immediately,
but only for symbols explicitly armed in the UI. Heavy research, broad data
collection, model training, and backtests stay on the home server.

This release is a lightweight execution agent, not a full research node.

## Deployment Shape

- `opus` runs on the VPS bound to `127.0.0.1:8081`.
- Access is via SSH tunnel:
  `ssh -L 8081:127.0.0.1:8081 user@vps`.
- `funding-sniper` may keep using `8080` and trade Bybit during tests.
- During `opus` Binance tests, `funding-sniper` must not trade Binance.
- `opus` is the only Binance Futures execution owner during tests.
- Use Binance API keys from `.env`.
- UI and WebSocket must both require auth when UI auth is configured.

## Configuration Split

`.env` stores secrets and hard maximums:

- `OPUS_HOST=127.0.0.1`
- `OPUS_PORT=8081`
- `OPUS_BINANCE_API_KEY`
- `OPUS_BINANCE_API_SECRET`
- `OPUS_TELEGRAM_BOT_TOKEN`
- `OPUS_TELEGRAM_CHAT_ID`
- hard caps for daily loss, 12h loss, symbol loss, max live symbols, and max
  notional.

Runtime trading settings are editable in the UI and persisted locally:

- default leverage: `10x`
- max live symbols: `5`
- daily loss cap: `$2`
- 12h loss cap: `$2`
- per-symbol loss cap: `$0.30`
- probation notional: `$6`
- active notional: `$20`
- probation trades: `7`
- cooldown: `6-12h`

UI values cannot exceed `.env` hard caps.

## Binance Connectivity

Follow Binance USD-M Futures WebSocket split:

- Public WS: `wss://fstream.binance.com/public/stream`
  - depth/bookTicker streams
- Market WS: `wss://fstream.binance.com/market/stream`
  - aggTrade and regular market data
- Private WS: `wss://fstream.binance.com/private/...`
  - listenKey events: `ORDER_TRADE_UPDATE`, `ACCOUNT_UPDATE`

Live trading is forbidden unless:

- public WS is connected and fresh;
- market WS is connected and fresh;
- private user-data WS is connected and fresh;
- REST ping/server-time check passes;
- exchange filters for the symbol are known;
- leverage has been set or verified.

REST is used for depth snapshots, exchangeInfo, leverage setup, order
submission/cancel, position reconciliation, and emergency flatten.

## Data And Resource Profile

For VPS live profile:

- raw depth logging disabled;
- raw trade logging disabled;
- no broad multi-symbol collector;
- no training/backtesting;
- keep only the active watchlist, max 5 live symbols;
- write lightweight logs:
  - closed trades;
  - decisions;
  - risk state changes;
  - hourly health snapshots.

Snapshot parquet collection can remain available for local/dev mode, but the
VPS live profile should avoid hourly read-concat-rewrite loops in hot paths.

## Symbol Lifecycle

Each UI watchlist row has an execution switch:

- `paper`
- `live`

Newly added symbols default to `paper`.

Live state machine:

1. `PAPER`
2. `WATCH_ONLY`
3. `PROBATION_LIVE`
4. `ACTIVE_LIVE`
5. `COOLDOWN`
6. `DISABLED`

Switching a symbol to live starts `PROBATION_LIVE`, not full-size trading.

## Probation Rules

Probation trades use `$6` notional.

After the first `7` closed live trades:

Promote to `ACTIVE_LIVE` at `$20` notional if:

- wins `>= 4`;
- `sum_net_bp > 0`;
- no `consecutive_losses >= 3`;
- symbol PnL is above `-$0.10`;
- tradeability score is still above threshold.

Move to `COOLDOWN` if:

- wins `<= 2`;
- `sum_net_bp < -20 bp`;
- `consecutive_losses >= 4`;
- symbol PnL reaches `-$0.30`;
- tradeability score drops below threshold.

Cooldown lasts 6-12 hours. A symbol can re-enter only after manual UI action or
after a future auto-discovery implementation explicitly requalifies it.

## Tradeability Filter

The filter exists to prevent `RIVERUSDT`-type failures.

Before a symbol can enter live probation:

- model confidence must be above the current threshold;
- recent gross movement proxy must exceed taker fee plus buffer;
- spread must not consume the expected move;
- recent realised movement must be sufficient for `1s` taker trades.

Initial threshold:

- require estimated gross edge around `>= 12 bp`;
- taker fee round-trip is treated as `8 bp`;
- the extra buffer covers spread, latency, and adverse selection.

The first implementation can use an approximate score based on:

- `abs(predicted_confidence)`;
- recent `vol_w120_bp` or equivalent online volatility;
- recent gross realised movement from paper/live trades;
- current spread bp.

This is called a tradeability score, not a mathematically exact expected value.

## Risk Limits

Global live stop:

- daily realised PnL `<= -$2`;
- 12h realised PnL `<= -$2`;
- private WS stale;
- public/market WS stale;
- REST/order errors exceed limit;
- emergency stop clicked in UI or Telegram.

Symbol stop:

- symbol realised PnL `<= -$0.30`;
- probation failure;
- consecutive losses limit;
- stale book for that symbol;
- Binance rejects/order lifecycle mismatch.

On global stop:

- stop opening new trades;
- cancel open orders for managed symbols;
- flatten managed Binance positions;
- switch runtime to waiting-for-user state.

## Live Order Rules

All Binance live orders must go through one order path.

Required behavior:

- set/verify leverage `10x`;
- respect tick size, step size, min notional;
- use `newClientOrderId` with `OPUS_` prefix;
- market-taker entries for MVP;
- reduce-only exits;
- reconcile fills from private WS;
- never trust local order state without Binance confirmation;
- on startup, reconcile open positions before allowing live.

If the account is in one-way mode, `opus` treats each managed symbol position as
fully owned by `opus` during Binance test periods.

## UI Changes

The UI becomes a trading cockpit:

- watchlist table keeps add/remove symbol flow;
- each symbol row has `paper/live` switch;
- live state badge: `paper`, `watch`, `probation`, `active`, `cooldown`,
  `disabled`;
- per-symbol PnL, wins/losses, consecutive losses, trade count, current notional;
- tradeability score and reason if blocked;
- global risk panel with editable runtime settings;
- emergency stop button;
- private WS/account status;
- managed live positions.

Settings edited in UI persist locally and are bounded by `.env` hard caps.

## Telegram

Telegram is operational visibility, not the primary control plane.

Required:

- hourly status message;
- `/status` command:
  - current live/paper symbols;
  - state per symbol;
  - PnL daily/12h/per-symbol;
  - open positions;
  - WS/account health;
  - active caps.
- emergency notifications:
  - symbol cooldown;
  - global stop;
  - private WS stale;
  - flatten result.

Telegram failure should show a UI warning but not by itself block trading.

## Existing Review Findings Included

This design includes the earlier review fixes:

1. raw depth disabled by default for VPS;
2. avoid parquet read-concat-rewrite in the VPS hot path;
3. WebSocket UI auth;
4. correct max live symbols guard;
5. real live flatten, not only cancel orders.

## First Implementation Slice

The first slice should implement live capability without auto-discovery:

1. VPS config/runtime settings.
2. WebSocket auth and hard caps.
3. Binance private WS and account/order reconciliation.
4. LiveTrader market orders with leverage, exchange filters, `OPUS_` client IDs.
5. Risk engine with daily, 12h, symbol caps.
6. Per-symbol paper/live UI switch and probation state.
7. Emergency cancel + flatten.
8. Telegram hourly and `/status`.

Auto-discovery and weekly retraining remain later slices.

## Open Decisions

- Exact persisted settings store: JSON file vs SQLite.
- Exact cooldown duration: 6h, 12h, or configurable.
- Whether active size should ever auto-scale beyond `$20` in later versions.
- Whether auto-discovery is implemented before or after the first VPS live test.
