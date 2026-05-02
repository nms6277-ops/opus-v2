"""Paper trader: executes virtual market-taker trades against live data.

Wired into the snapshot loop. On every fresh snapshot for an allowed
symbol it asks the :class:`Predictor` for a directional prediction; if
``|confidence| > threshold`` and we have no open position, it opens a
long (confidence > 0) or short (confidence < 0) at the current
ask/bid. Open positions exit when:

- the holding horizon elapses (default: same as ``trade_horizon`` — 5s),
- the predictor flips to the opposite confidence (early reversal exit),
- the stop-loss in basis points is hit, or
- the runtime stops.

PnL is computed against the mid at exit (taker assumption; we cross the
half-spread on entry and exit). Fees are subtracted in basis points
based on ``settings.taker_fee_bp``. Every closed trade is appended to a
:class:`TradeLogWriter` instance shared across symbols.

Safety: every trade goes through :mod:`backend.safety.guards` checks
(daily_loss_limit, max_position_usd, max_orders_per_min). If a guard
trips, the trader logs and skips — it never raises into the snapshot
loop.

This trader is intentionally minimal: it's the **simplest** complete
paper-trade system that exercises the full prediction → trade →
journal cycle. We can layer maker fills, multi-leg strategies, and
proper queue position modelling on top later (see
:mod:`backend.ml.backtest` for the maker-aware backtest).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from backend.collector.trade_log import TradeLogWriter, TradeRecord
from backend.config import settings
from backend.ml.labels import parse_horizon_ms
from backend.safety.guards import (
    OrderIntent,
    check,
    record_order_sent,
    record_pnl,
)
from backend.safety.guards import (
    reset_if_new_day as _guards_reset_if_new_day,
)
from backend.safety.regime import RiskEvent, record_trade_outcome
from backend.settings_store import RuntimeSettings
from backend.state import AppState
from backend.traders.base import Trader

if TYPE_CHECKING:
    from backend.ml.inference import Predictor

log = logging.getLogger("backend.traders.paper")


def _resolve_horizon_ms(spec: str) -> int:
    """Resolve a horizon spec to milliseconds.

    Custom horizons (e.g. ``2s``, ``500ms``, ``15s``) are first-class —
    ``opus train --horizons 2s 15s`` produces matching models, and the
    paper trader must hold positions for the **exact** same duration so
    PnL is comparable to backtest. Falls back to 5_000 ms only if the
    spec is unparseable, which is itself a misconfiguration we log
    loudly.
    """
    try:
        return parse_horizon_ms(spec)
    except ValueError as e:
        log.error(
            "paper: cannot parse trade_horizon=%r (%s); defaulting to 5000ms. "
            "Set OPUS_TRADE_HORIZON to e.g. 1s, 2s, 5s, 30s, 500ms, 1m.",
            spec, e,
        )
        return 5_000


@dataclass
class OpenPosition:
    """One open paper-trade leg per symbol."""

    symbol: str
    side: str               # "long" | "short"
    qty: float
    notional_usd: float
    entry_price: float
    ts_open_ms: int
    horizon: str
    horizon_ms: int
    pred_confidence: float
    pred_p_up: float
    pred_p_flat: float
    pred_p_down: float


class PaperTrader(Trader):
    """Virtual taker that drives trades from :class:`Predictor` output."""

    def __init__(
        self,
        state: AppState,
        predictor: Predictor | None = None,
        trade_log: TradeLogWriter | None = None,
        runtime_settings: RuntimeSettings | None = None,
        on_risk_event: Callable[[RiskEvent], Awaitable[None]] | None = None,
    ) -> None:
        self.state = state
        self.predictor = predictor
        self.trade_log = trade_log
        self.runtime_settings = runtime_settings or RuntimeSettings()
        self._on_risk_event = on_risk_event
        self._running = False

        self._positions: dict[str, OpenPosition] = {}
        self._allowed: set[str] = set(s.upper() for s in settings.trade_symbols)
        self._horizon = settings.trade_horizon
        self._horizon_ms = _resolve_horizon_ms(self._horizon)
        self._conf_thr = settings.trade_conf_threshold
        self._notional = settings.trade_notional_usd
        self._taker_fee_bp = settings.taker_fee_bp
        self._stop_loss_bp = settings.trade_stop_loss_bp

        # Per-day counters for UI / stats. ``_day_bucket`` is the current
        # UTC day index (YYYYMMDD as int); when it rolls over we reset the
        # three counters below. Same pattern used by safety.guards.
        self._day_bucket = 0
        self._daily_trades = 0
        self._daily_wins = 0
        self._daily_losses = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        self._running = True
        if self.predictor is None or not getattr(self.predictor, "enabled", False):
            log.info("paper trader: started (idle — no model loaded)")
        else:
            log.info(
                "paper trader: started horizon=%s conf_thr=%.3f notional=$%.2f symbols=%s",
                self._horizon,
                self._conf_thr,
                self._notional,
                sorted(self._allowed) if self._allowed else "(any)",
            )

    async def stop(self) -> None:
        self._running = False
        # Close any open positions at the last known mid.
        open_syms = list(self._positions.keys())
        for symbol in open_syms:
            await self._close(symbol, exit_reason="manual", exit_price=None)
        log.info("paper trader: stopped (closed %d open positions)", len(open_syms))

    async def cancel_all(self, symbol: str) -> None:
        symbol = symbol.upper()
        if symbol in self._positions:
            await self._close(symbol, exit_reason="manual", exit_price=None)

    async def on_book_update(self, symbol: str) -> None:
        # We act on snapshot ticks (`on_snapshot`), not book updates.
        return

    # ------------------------------------------------------------------
    # Snapshot-driven trading loop
    # ------------------------------------------------------------------
    async def on_snapshot(self, symbol: str, snap: dict) -> None:
        """Called from the runtime snapshot loop after each new snapshot."""
        if not self._running:
            return
        symbol = symbol.upper()
        if self._allowed and symbol not in self._allowed:
            return
        if self.predictor is None or not getattr(self.predictor, "enabled", False):
            return

        try:
            preds = self.predictor.predict(symbol, snap)
        except Exception as e:
            log.error("paper: predict %s failed: %s", symbol, e)
            return
        pred = preds.get(self._horizon) if preds else None

        # Manage existing position first (maybe close it).
        if symbol in self._positions:
            await self._maybe_close(symbol, snap, pred)

        # Then consider opening a new one.
        if pred is not None and symbol not in self._positions:
            await self._maybe_open(symbol, snap, pred)

    # ------------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------------
    async def _maybe_open(self, symbol: str, snap: dict, pred) -> None:
        if abs(pred.confidence) < self._conf_thr:
            return

        stats = self.state.symbols.get(symbol)
        if stats is not None and stats.live_state in {
            "awaiting_operator",
            "watch_only",
            "cooldown",
            "disabled",
        }:
            stats.rejects_count += 1
            return

        side = "long" if pred.confidence > 0 else "short"
        # Taker entry: long buys at ask, short sells at bid.
        entry_price = float(snap.get("best_ask" if side == "long" else "best_bid", 0.0))
        if entry_price <= 0:
            return
        qty = self._notional / entry_price

        # Pre-trade safety: notional, daily loss, rate-limit, etc.
        intent = OrderIntent(
            symbol=symbol,
            side="BUY" if side == "long" else "SELL",
            qty=qty,
            price=entry_price,
            notional_usd=self._notional,
        )
        reason = check(self.state, intent, self.state.mode.value)
        if reason is not None:
            log.debug("paper: pre-trade rejected %s: %s", symbol, reason)
            stats = self.state.symbols.get(symbol)
            if stats is not None:
                stats.rejects_count += 1
            return

        ts_open_ms = int(snap.get("ts_ms", time.time() * 1000))

        pos = OpenPosition(
            symbol=symbol,
            side=side,
            qty=qty,
            notional_usd=self._notional,
            entry_price=entry_price,
            ts_open_ms=ts_open_ms,
            horizon=self._horizon,
            horizon_ms=self._horizon_ms,
            pred_confidence=pred.confidence,
            pred_p_up=pred.p_up,
            pred_p_flat=pred.p_flat,
            pred_p_down=pred.p_down,
        )
        self._positions[symbol] = pos

        # Update UI state
        stats = self.state.symbols.get(symbol)
        if stats is not None:
            stats.position_base = qty if side == "long" else -qty
            stats.position_entry = entry_price
            stats.fills_count += 1

        # Tally rate-limit counter for the next minute window
        record_order_sent(self.state.guards)

        log.info(
            "paper: OPEN %s %s qty=%.6f @ %.6f conf=%+.3f horizon=%s",
            side, symbol, qty, entry_price, pred.confidence, self._horizon,
        )

    async def _maybe_close(self, symbol: str, snap: dict, pred) -> None:
        pos = self._positions[symbol]
        now_ms = int(snap.get("ts_ms", time.time() * 1000))
        held_ms = now_ms - pos.ts_open_ms

        # Compute current mark for stop-loss / opposing-signal checks.
        # For long we exit at bid, for short at ask (taker exit).
        mark_price = float(snap.get("best_bid" if pos.side == "long" else "best_ask", 0.0))
        if mark_price <= 0:
            return

        gross_bp = self._gross_pnl_bp(pos, mark_price)

        # Reasons to exit (in priority order)
        if gross_bp <= -self._stop_loss_bp:
            await self._close(symbol, exit_reason="stop_loss", exit_price=mark_price)
            return
        if pred is not None:
            opposing = (pos.side == "long" and pred.confidence < -self._conf_thr) or (
                pos.side == "short" and pred.confidence > self._conf_thr
            )
            if opposing:
                await self._close(symbol, exit_reason="opposing_signal", exit_price=mark_price)
                return
        if held_ms >= pos.horizon_ms:
            await self._close(symbol, exit_reason="timeout", exit_price=mark_price)
            return

        # Otherwise keep position; surface unrealised PnL to UI.
        net_bp = gross_bp - 2 * self._taker_fee_bp
        unrealized_usd = pos.notional_usd * (net_bp / 10_000.0)
        stats = self.state.symbols.get(symbol)
        if stats is not None:
            stats.unrealized_pnl = unrealized_usd

    @staticmethod
    def _gross_pnl_bp(pos: OpenPosition, mark_price: float) -> float:
        if pos.entry_price <= 0:
            return 0.0
        sign = 1.0 if pos.side == "long" else -1.0
        return sign * (mark_price - pos.entry_price) / pos.entry_price * 10_000.0

    async def _close(
        self,
        symbol: str,
        *,
        exit_reason: str,
        exit_price: float | None,
    ) -> None:
        pos = self._positions.pop(symbol, None)
        if pos is None:
            return
        # Fall back to last known mid from state if exit_price missing.
        if exit_price is None or exit_price <= 0:
            stats = self.state.symbols.get(symbol)
            exit_price = stats.best_bid if (stats and pos.side == "long" and stats.best_bid > 0) \
                else stats.best_ask if (stats and pos.side == "short" and stats.best_ask > 0) \
                else pos.entry_price
        gross_bp = self._gross_pnl_bp(pos, float(exit_price))
        fee_bp = 2.0 * self._taker_fee_bp
        net_bp = gross_bp - fee_bp
        pnl_usd = pos.notional_usd * (net_bp / 10_000.0)
        ts_close_ms = int(time.time() * 1000)

        # Update stats (resetting first if we crossed a UTC day boundary).
        self._reset_if_new_day()
        self._daily_trades += 1
        if pnl_usd > 0:
            self._daily_wins += 1
        elif pnl_usd < 0:
            self._daily_losses += 1
        record_pnl(self.state.guards, pnl_usd, symbol=symbol)

        stats = self.state.symbols.get(symbol)
        if stats is not None:
            stats.position_base = 0.0
            stats.position_entry = 0.0
            stats.realized_pnl += pnl_usd
            stats.unrealized_pnl = 0.0
            stats.fills_count += 1

        event = record_trade_outcome(
            self.state,
            symbol=symbol,
            pnl_usd=pnl_usd,
            net_bp=net_bp,
            settings=self.runtime_settings,
        )
        if event is not None and self._on_risk_event is not None:
            await self._on_risk_event(event)

        log.info(
            "paper: CLOSE %s %s @ %.6f gross=%+.2fbp net=%+.2fbp pnl=$%+.4f reason=%s",
            pos.side, symbol, exit_price, gross_bp, net_bp, pnl_usd, exit_reason,
        )

        if self.trade_log is not None:
            rec = TradeRecord(
                ts_open_ms=pos.ts_open_ms,
                ts_close_ms=ts_close_ms,
                symbol=symbol,
                side=pos.side,
                qty=pos.qty,
                notional_usd=pos.notional_usd,
                entry_price=pos.entry_price,
                exit_price=float(exit_price),
                pnl_bp=gross_bp,
                fee_bp=fee_bp,
                net_bp=net_bp,
                pnl_usd=pnl_usd,
                horizon=pos.horizon,
                predicted_confidence=pos.pred_confidence,
                p_up=pos.pred_p_up,
                p_flat=pos.pred_p_flat,
                p_down=pos.pred_p_down,
                exit_reason=exit_reason,
                is_maker_entry=False,
                is_maker_exit=False,
            )
            try:
                await self.trade_log.append(rec)
            except Exception as e:
                log.error("paper: trade_log append failed: %s", e)

    # ------------------------------------------------------------------
    # UI accessors
    # ------------------------------------------------------------------
    def _reset_if_new_day(self) -> None:
        """Roll the trader's daily counters when the UTC day changes.

        Also calls :func:`backend.safety.guards.reset_if_new_day` so that the
        UI numbers (trades / wins / losses / win-rate / paper PnL) stay in
        sync with the guards' independent daily PnL counter even on days
        with no trades.
        """
        _guards_reset_if_new_day(self.state.guards)
        today = int(datetime.now(tz=UTC).strftime("%Y%m%d"))
        if self._day_bucket != today:
            if self._day_bucket != 0:
                log.info("paper trader: new UTC day, resetting daily counters")
            self._day_bucket = today
            self._daily_trades = 0
            self._daily_wins = 0
            self._daily_losses = 0

    def stats_dict(self) -> dict:
        """Snapshot of trader stats for inclusion in the UI status payload."""
        self._reset_if_new_day()
        n_trades = self._daily_trades
        win_rate = (self._daily_wins / n_trades) if n_trades else 0.0
        return {
            "open_positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "qty": p.qty,
                    "notional_usd": p.notional_usd,
                    "entry_price": p.entry_price,
                    "ts_open_ms": p.ts_open_ms,
                    "horizon": p.horizon,
                    "confidence": p.pred_confidence,
                }
                for p in self._positions.values()
            ],
            "daily_trades": n_trades,
            "daily_wins": self._daily_wins,
            "daily_losses": self._daily_losses,
            "daily_win_rate": win_rate,
            "daily_pnl_usd": self.state.guards.daily_pnl,
            "horizon": self._horizon,
            "conf_threshold": self._conf_thr,
            "notional_usd": self._notional,
            "allowed_symbols": sorted(self._allowed) if self._allowed else [],
        }
