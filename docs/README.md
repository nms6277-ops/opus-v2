# `opus-v2` documentation

Project state: **paused.** Direction prediction with 100 ms time bars
in taker mode on micro-cap altcoin futures was tested end-to-end and
does not produce a deployable edge under realistic execution costs.
Infrastructure is correct, the experiment is conclusive, the project
is suspended pending a different direction.

If you need to come back later, this folder is the briefing pack.

| Document | What you will find there |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Module-by-module map of the backend, data flow diagram, hot-path constraints, "where do I look to change X?" cheatsheet. |
| [CONFIGURATION.md](CONFIGURATION.md) | Every `.env` variable and every UI-editable runtime setting, with defaults, valid ranges, and what each one actually does. Recommended starting configurations for local / VPS / training boxes. |
| [RESULTS.md](RESULTS.md) | What we trained, how it backtested, and how it fell apart on a fresh out-of-sample window. Per-symbol numbers, honest selection-bias analysis, list of things we deliberately did NOT do. Reproduction commands. |
| [ROADMAP.md](ROADMAP.md) | Possible directions if the project is reactivated, sized honestly: dollar bars, mean-reversion-after-toxic-flow, larger symbols, maker-mode, alternative strategies. The recommended sequence is at the end. |
| [vps-live-runbook.md](vps-live-runbook.md) | Existing operator runbook for VPS deployments. Pre-dates this pause. |

The cardinal rule extracted from all of this:

> **Do not believe `maker_*` numbers from `backend/ml/backtest.py`.**
> Queue position, partial fills, cancellations and adverse selection
> are not modelled. Validation of any maker-style strategy must be
> done in paper-mode for ≥24 h, never via backtest alone.

For high-level project framing and operator usage see the top-level
[`README.md`](../README.md), [`AGENTS.md`](../AGENTS.md) and
[`OPUS-MANUAL.md`](../OPUS-MANUAL.md).
