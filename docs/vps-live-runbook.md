# VPS Live Runbook

Target box: 2 GB RAM, 2 vCPU, 30 GB NVMe. UI is local-only and should be reached through SSH tunnel.

## Minimal `.env`

```dotenv
OPUS_MODE=collect
OPUS_RUNTIME_PROFILE=vps_live
OPUS_HOST=127.0.0.1
OPUS_PORT=8081

OPUS_COLLECT_RAW_DEPTH=false
OPUS_COLLECT_RAW_TRADES=false

OPUS_BINANCE_API_KEY=...
OPUS_BINANCE_API_SECRET=...

OPUS_UI_USER=...
OPUS_UI_PASSWORD=...

OPUS_HARD_DAILY_LOSS_USD=2.0
OPUS_HARD_12H_LOSS_USD=2.0
OPUS_HARD_SYMBOL_LOSS_USD=0.30
OPUS_HARD_MAX_LIVE_SYMBOLS=5
OPUS_HARD_MAX_NOTIONAL_USD=20.0
OPUS_HARD_MAX_LEVERAGE=10

OPUS_TELEGRAM_BOT_TOKEN=
OPUS_TELEGRAM_CHAT_ID=
```

## SSH tunnel

```bash
ssh -N -L 8081:127.0.0.1:8081 user@vps
```

Open `http://127.0.0.1:8081` locally.

## Operating flow

1. Start process in `collect` or `paper`.
2. Add symbols in the UI watchlist.
3. Set runtime profile in UI: leverage, max live symbols, notional, min expected gross bp, daily/12h/symbol loss caps.
4. Switch global mode to `LIVE`.
5. Enable live only on selected symbol rows with the paper/live switch.
6. Use `STOP + FLATTEN` if anything looks wrong. It trips emergency, cancels orders, and closes managed live symbols best-effort.

## Notes

- Raw depth/trade logs are off by default for VPS.
- Snapshot parquet is written as immutable chunks, not read-concat-rewrite.
- Per-symbol live execution is blocked when `abs(confidence) * vol_w120_bp` is below the UI min gross threshold.
- Binance public, market, and private websocket URLs are split for the current Binance Futures WS layout.
- Funding sniper default `8080` does not conflict with opus default `8081`.
