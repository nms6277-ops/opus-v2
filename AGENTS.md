# AGENTS.md

Notes for AI assistants working on this repository.

## Project

`opus` — microstructure trading bot for Binance USDT-M Futures with
cross-exchange analysis via Bybit. Owner uses it for short-horizon
in-play altcoin trading.

## Key constraints

-   Target VPS: 2 GB RAM, 2 vCPU, 30 GB NVMe. Hot path must stay in pure
    `numpy`/`numba`/`struct`; **no pandas in the realtime loop**.
-   Data collection runs 24/7. Disk budget ≈ 300 MB/day after zstd — so
    snapshots are **pre-aggregated** at 250 ms intervals, not raw L2.
-   Binance Futures (USDT-M) is the only trading venue. Bybit is
    read-only for cross-exchange features.
-   UI is local, served on 127.0.0.1 behind an optional SSH tunnel.

## Commands

-   Install: `bash scripts/install_local.sh`
-   Run: `source .venv/bin/activate && python -m backend.main`
-   Lint: `ruff check . && ruff format --check .`

## Safety rules

-   Never commit `.env`, API keys, or anything under `data/` or `models/`.
-   `LIVE` trading is gated by several guards (see `backend/safety/guards.py`);
    keep those checks in any new order-submission path.
-   When adding a new feature, also record a new column in
    `backend/features/snapshot.py` and update the README's data-schema
    section.

## Repo layout

```
backend/
  api/         # REST + WS for the UI
  collector/   # LOB, snapshot writer
  exchanges/   # Binance / Bybit WS + Binance REST
  features/    # Snapshot feature computation
  safety/      # Pre-trade guards
  traders/     # Paper + live engines
  config.py    # Pydantic settings from .env
  log.py
  runtime.py   # Orchestrator (singleton)
  state.py     # Runtime state for the UI
  main.py      # FastAPI entry point
frontend/      # Static HTML/JS, no framework
scripts/       # install_local.sh, ...
systemd/       # opus@user.service
data/          # Parquet snapshots (gitignored)
logs/          # Runtime logs (gitignored)
models/        # Trained ML artefacts (gitignored)
```
