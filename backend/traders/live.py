"""Live trader for Binance USDT-M Futures.

This is the first live-capable implementation: market-taker entries, guarded
per-symbol live state, Binance private-WS health gate, leverage setup, and
emergency flatten for managed symbols.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from backend.config import settings
from backend.exchanges.binance_filters import round_qty_down, validate_notional
from backend.exchanges.binance_rest import BinanceRest, get_client
from backend.log import get_logger
from backend.ml.labels import parse_horizon_ms
from backend.safety.guards import OrderIntent, check, record_order_sent, record_pnl
from backend.safety.regime import RiskEvent, record_trade_outcome
from backend.settings_store import RuntimeSettings
from backend.state import AppState
from backend.traders.base import Trader

if TYPE_CHECKING:
    from backend.ml.inference import Predictor

log = get_logger(__name__)


def _resolve_horizon_ms(spec: str) -> int:
    try:
        return parse_horizon_ms(spec)
    except Exception:
        log.warning("live: cannot parse trade_horizon=%r; defaulting to 5000ms", spec)
        return 5_000


@dataclass
class _LivePosition:
    """Open Binance live position tracked for automated exits."""

    symbol: str
    side: str  # "long" | "short"
    qty: float
    notional_usd: float
    entry_price: float
    ts_open_ms: int
    horizon_ms: int


class LiveTrader(Trader):
    """Market-taker live trader with strict pre-trade gates."""

    def __init__(
        self,
        state: AppState,
        *,
        predictor: Predictor | None = None,
        rest_client: BinanceRest | None = None,
        runtime_settings: RuntimeSettings | None = None,
        on_risk_event: Callable[[RiskEvent], Awaitable[None]] | None = None,
    ) -> None:
        self.state = state
        self.predictor = predictor
        self.rest = rest_client or get_client()
        self.runtime_settings = runtime_settings or RuntimeSettings()
        self._on_risk_event = on_risk_event
        self._running = False
        self._leveraged_symbols: set[str] = set()
        self._positions: dict[str, _LivePosition] = {}
        self._horizon_ms = _resolve_horizon_ms(settings.trade_horizon)
        self._stop_loss_bp = settings.trade_stop_loss_bp
        self._taker_fee_bp = settings.taker_fee_bp
        self._conf_thr = settings.trade_conf_threshold

    async def start(self) -> None:
        self._running = True
        log.warning("LIVE trader: started; real Binance orders are enabled for per-symbol live rows")

    async def stop(self) -> None:
        self._running = False
        log.info("live trader: stopping, cancelling and flattening managed symbols")
        await self.emergency_flatten()

    async def on_book_update(self, symbol: str) -> None:
        return

    async def cancel_all(self, symbol: str) -> None:
        try:
            await self.rest.cancel_all(symbol.upper())
        except Exception as e:
            log.warning("cancel_all(%s) failed: %s", symbol, e)

    async def flatten_symbol(self, symbol: str) -> None:
        """Cancel open orders and close any live Binance position for one symbol."""

        symbol = symbol.upper()
        await self.cancel_all(symbol)
        await self._flatten(symbol)

    async def emergency_flatten(self) -> None:
        """Best-effort flatten for every symbol managed by this trader."""

        for sym in self._managed_symbols():
            await self.flatten_symbol(sym)

    async def on_snapshot(self, symbol: str, snap: dict[str, Any]) -> None:
        """Evaluate a live entry (or exit) on the latest snapshot."""

        if not self._running:
            return
        symbol = symbol.upper()
        stats = self.state.symbols.get(symbol)
        if stats is None:
            return
        # Open-position exit checks run BEFORE the execution_mode / live_state
        # gate. If an operator disables a symbol whose position never made it
        # back to Binance (or hasn't been popped yet), we still need stop-loss
        # / timeout monitoring on the locally-tracked position; otherwise it
        # gets stranded the moment the gate flips.
        if symbol in self._positions:
            pred = self._prediction(symbol, snap) if self.predictor is not None else None
            await self._maybe_close(symbol, snap, pred)
            return

        if stats.execution_mode != "live" or stats.live_state not in {"probation_live", "active_live"}:
            return

        if not self._private_ws_ready():
            self._reject_symbol(symbol, "private ws not ready")
            return
        if self.predictor is None or not getattr(self.predictor, "enabled", False):
            self._reject_symbol(symbol, "predictor disabled")
            return

        pred = self._prediction(symbol, snap)
        if pred is None:
            return
        expected_gross_bp = self._expected_gross_bp(symbol, pred)
        stats.expected_gross_bp = expected_gross_bp
        if expected_gross_bp < self.runtime_settings.min_expected_gross_bp:
            stats.block_reason = (
                f"expected gross {expected_gross_bp:.2f}bp < "
                f"{self.runtime_settings.min_expected_gross_bp:.2f}bp"
            )
            return
        if abs(float(pred.confidence)) < settings.trade_conf_threshold:
            return

        side = "BUY" if pred.confidence > 0 else "SELL"
        entry_price = float(snap.get("best_ask" if side == "BUY" else "best_bid", 0.0))
        if entry_price <= 0:
            self._reject_symbol(symbol, "invalid entry price")
            return

        notional = float(stats.current_notional_usd or self.runtime_settings.probation_notional_usd)
        raw_qty = notional / entry_price
        try:
            filters = await self.rest.cached_symbol_filters(symbol)
        except Exception as e:
            self._reject_symbol(symbol, f"symbol filters unavailable: {e}")
            return
        qty = round_qty_down(raw_qty, filters)
        if qty <= 0:
            self._reject_symbol(
                symbol,
                f"qty {raw_qty:.10f} rounds to 0 at step={filters.step_size}",
            )
            return
        try:
            validate_notional(price=entry_price, qty=qty, filters=filters)
        except ValueError as e:
            self._reject_symbol(symbol, f"notional check: {e}")
            return
        intent = OrderIntent(
            symbol=symbol,
            side=side,
            qty=qty,
            price=entry_price,
            notional_usd=notional,
        )
        reason = check(self.state, intent, "live")
        if reason is not None:
            self._reject_symbol(symbol, reason)
            return

        await self._ensure_leverage(symbol)
        client_order_id = f"OPUS_{symbol}_{int(time.time() * 1000)}_{side}"
        try:
            await self.rest.place_order(
                symbol=symbol,
                side=side,
                order_type="MARKET",
                quantity=qty,
                reduce_only=False,
                client_order_id=client_order_id,
            )
        except Exception as e:
            self._reject_symbol(symbol, f"order failed: {e}")
            return

        ts_open_ms = int(snap.get("ts_ms", time.time() * 1000))
        self._positions[symbol] = _LivePosition(
            symbol=symbol,
            side="long" if side == "BUY" else "short",
            qty=qty,
            notional_usd=qty * entry_price,
            entry_price=entry_price,
            ts_open_ms=ts_open_ms,
            horizon_ms=self._horizon_ms,
        )
        record_order_sent(self.state.guards)
        stats.position_base = qty if side == "BUY" else -qty
        stats.position_entry = entry_price
        # ``fills_count`` is incremented in ``_maybe_close`` after the
        # round-trip completes, matching the PaperTrader semantics.
        log.info("live: OPEN %s %s qty=%.8f notional=$%.2f", side, symbol, qty, notional)

    async def _maybe_close(self, symbol: str, snap: dict[str, Any], pred: Any) -> None:
        """Close an open live position if any exit condition fires.

        Mirrors PaperTrader._maybe_close: stop-loss > opposing-signal >
        horizon timeout. Closing goes through ``_flatten`` which uses a
        reduce-only MARKET order.
        """

        pos = self._positions.get(symbol)
        if pos is None:
            return
        now_ms = int(snap.get("ts_ms", time.time() * 1000))
        held_ms = now_ms - pos.ts_open_ms
        # Long exits at bid, short exits at ask (taker exit).
        mark_price = float(snap.get("best_bid" if pos.side == "long" else "best_ask", 0.0))
        if mark_price <= 0:
            return

        gross_bp = self._gross_pnl_bp(pos, mark_price)
        exit_reason: str | None = None
        if gross_bp <= -self._stop_loss_bp:
            exit_reason = "stop_loss"
        elif pred is not None and (
            (pos.side == "long" and float(pred.confidence) < -self._conf_thr)
            or (pos.side == "short" and float(pred.confidence) > self._conf_thr)
        ):
            exit_reason = "opposing_signal"
        elif held_ms >= pos.horizon_ms:
            exit_reason = "timeout"

        if exit_reason is None:
            net_bp = gross_bp - 2.0 * self._taker_fee_bp
            stats = self.state.symbols.get(symbol)
            if stats is not None:
                stats.unrealized_pnl = pos.notional_usd * (net_bp / 10_000.0)
            return

        net_bp = gross_bp - 2.0 * self._taker_fee_bp
        pnl_usd = pos.notional_usd * (net_bp / 10_000.0)
        log.info(
            "live: CLOSE %s reason=%s gross=%.2fbp net=%.2fbp pnl=$%.4f held=%dms",
            symbol,
            exit_reason,
            gross_bp,
            net_bp,
            pnl_usd,
            held_ms,
        )

        # Pop the position from local tracking BEFORE attempting the
        # Binance REST flatten. ``_flatten`` swallows REST errors, so if
        # we left the entry in ``self._positions`` the next snapshot
        # would re-fire the same exit and double-count PnL on every
        # subsequent tick. The Binance position itself, if the REST call
        # failed, will be reconciled on next startup or via
        # ``emergency_flatten``.
        self._positions.pop(symbol, None)

        record_pnl(self.state.guards, pnl_usd, symbol=symbol)
        stats = self.state.symbols.get(symbol)
        if stats is not None:
            stats.realized_pnl += pnl_usd
            stats.fills_count += 1
        # Per-symbol regime guards (loss-streak pause, profit giveback,
        # rolling degradation) only fire if record_trade_outcome is
        # called with the realised PnL. Without it LIVE mode would have
        # none of these protections.
        event = record_trade_outcome(
            self.state,
            symbol=symbol,
            pnl_usd=pnl_usd,
            net_bp=net_bp,
            settings=self.runtime_settings,
        )
        if event is not None and self._on_risk_event is not None:
            await self._on_risk_event(event)

        await self._flatten(symbol)

    @staticmethod
    def _gross_pnl_bp(pos: _LivePosition, mark_price: float) -> float:
        if pos.entry_price <= 0:
            return 0.0
        sign = 1.0 if pos.side == "long" else -1.0
        return sign * (mark_price - pos.entry_price) / pos.entry_price * 10_000.0

    def _prediction(self, symbol: str, snap: dict[str, Any]):
        try:
            preds = self.predictor.predict(symbol, snap) if self.predictor is not None else None
        except Exception as e:
            self._reject_symbol(symbol, f"predict failed: {e}")
            return None
        if not preds:
            return None
        pred = preds.get(settings.trade_horizon)
        if pred is None and len(preds) == 1:
            return next(iter(preds.values()))
        return pred

    def _expected_gross_bp(self, symbol: str, pred: Any) -> float:
        latest_derived = getattr(self.predictor, "latest_derived", None)
        if latest_derived is None:
            return self.runtime_settings.min_expected_gross_bp
        derived = latest_derived(symbol)
        vol_bp = float(derived.get("vol_w120_bp") or 0.0)
        if not vol_bp > 0.0:
            return 0.0
        return abs(float(pred.confidence)) * vol_bp

    async def _ensure_leverage(self, symbol: str) -> None:
        if symbol in self._leveraged_symbols:
            return
        await self.rest.set_leverage(symbol, self.runtime_settings.leverage)
        self._leveraged_symbols.add(symbol)

    async def _flatten(self, symbol: str) -> None:
        # Local tracking is cleared by the caller (``_maybe_close``) or
        # by ``emergency_flatten`` so the REST call below cannot cause a
        # double-count if it fails. We still pop here defensively for
        # callers that bypass ``_maybe_close``.
        self._positions.pop(symbol, None)
        try:
            position_amt = await self.rest.get_open_position_amt(symbol)
            if position_amt:
                await self.rest.market_close_position(symbol, position_amt)
            stats = self.state.symbols.get(symbol)
            if stats is not None:
                stats.position_base = 0.0
                stats.position_entry = 0.0
                stats.unrealized_pnl = 0.0
        except Exception as e:
            log.warning("flatten(%s) failed: %s", symbol, e)

    def _managed_symbols(self) -> list[str]:
        return [
            sym
            for sym, stats in self.state.symbols.items()
            if stats.execution_mode == "live"
            and stats.live_state in {"probation_live", "active_live", "cooldown", "disabled"}
        ]

    # The Binance USER-DATA stream is event-driven (ORDER_TRADE_UPDATE /
    # ACCOUNT_UPDATE), not a continuous heartbeat like the public depth
    # feed. Between events the gap can be minutes, so reusing the public-WS
    # ``ws_stale_ms`` (default 1500ms) here would block every single live
    # entry. We therefore check connection state only and rely on the
    # websockets library's ping/pong + listenKey keepalive in
    # BinancePrivateWS to detect dead sockets.
    def _private_ws_ready(self) -> bool:
        return bool(self.state.binance_private_connected)

    def _reject_symbol(self, symbol: str, reason: str) -> None:
        stats = self.state.symbols.get(symbol)
        if stats is not None:
            stats.rejects_count += 1
            stats.block_reason = reason
        log.debug("live: reject %s: %s", symbol, reason)
