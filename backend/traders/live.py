"""Live trader for Binance USDT-M Futures.

This is the first live-capable implementation: market-taker entries, guarded
per-symbol live state, Binance private-WS health gate, leverage setup, and
emergency flatten for managed symbols.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from backend.config import settings
from backend.exchanges.binance_filters import round_price, round_qty_down, validate_notional
from backend.exchanges.binance_rest import BinanceRest, get_client
from backend.log import get_logger
from backend.ml.labels import parse_horizon_ms
from backend.safety.guards import OrderIntent, check, record_order_sent, record_pnl
from backend.safety.regime import RiskEvent, record_trade_outcome
from backend.settings_store import RuntimeSettings
from backend.state import AppState
from backend.traders.base import Trader

if TYPE_CHECKING:
    from backend.collector.live_trade_log import LiveTradeLogWriter
    from backend.ml.inference import Predictor

log = get_logger(__name__)

# Cap on the size of ``_seen_private_trade_ids``. Binance fill replays
# only happen within a WS reconnect window, so a few thousand entries
# is more than enough for dedupe; we evict the oldest above this cap to
# keep the live trader bounded on the 2 GB target VPS.
_MAX_SEEN_TRADE_IDS = 10_000


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
    pred_confidence: float = 0.0


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
        live_trade_log: LiveTradeLogWriter | None = None,
    ) -> None:
        self.state = state
        self.predictor = predictor
        self.rest = rest_client or get_client()
        self.runtime_settings = runtime_settings or RuntimeSettings()
        self._on_risk_event = on_risk_event
        self.live_trade_log = live_trade_log
        self._running = False
        # Map ``symbol -> last-applied leverage`` so an operator-initiated
        # leverage change via ``POST /api/settings`` (which updates
        # ``self.runtime_settings.leverage``) takes effect on the next
        # entry. A plain set would let the cached "already set" flag
        # short-circuit ``_ensure_leverage`` and keep the symbol at the
        # old higher leverage silently — a safety-critical mismatch.
        self._leveraged_symbols: dict[str, float] = {}
        self._positions: dict[str, _LivePosition] = {}
        self._horizon_ms = _resolve_horizon_ms(settings.trade_horizon)
        self._stop_loss_bp = settings.trade_stop_loss_bp
        self._taker_fee_bp = settings.taker_fee_bp
        self._conf_thr = settings.trade_conf_threshold
        self._take_profit_bp = float(settings.trade_take_profit_bp)
        # Dedupe key: (symbol, trade_id). Binance occasionally re-emits the
        # same fill event during reconnects; processing it twice would
        # double-count realized_pnl. Bounded with FIFO eviction so 24/7
        # operation cannot leak memory on the 2 GB VPS.
        self._seen_private_trade_ids: set[tuple[str, str]] = set()
        self._seen_private_trade_order: deque[tuple[str, str]] = deque()
        # Accumulator for partial fills on a reduce-only close: keyed by
        # (symbol, client_order_id) so we can flush a single
        # ``record_trade_outcome`` when the order finally hits FILLED.
        self._close_order_pnl: dict[tuple[str, str], float] = {}

    async def start(self) -> None:
        self._running = True
        await self.reconcile_open_positions()
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
        """Best-effort flatten for every symbol with potential exposure.

        Includes symbols still flagged ``execution_mode=="live"`` AND any
        symbol with a locally-tracked open position. The latter covers
        operator workflows where a symbol is switched live→paper while a
        real Binance position is still open: ``_managed_symbols`` would
        skip it, but the actual position is in ``self._positions`` and
        must still be closed.
        """

        symbols = set(self._managed_symbols()) | set(self._positions.keys())
        for sym in symbols:
            await self.flatten_symbol(sym)

    async def reconcile_open_positions(self) -> None:
        """Fail closed on startup when Binance already has open positions.

        If the bot was restarted with a position still on the exchange we
        cannot trust our local state for it (no ``_LivePosition`` entry,
        unknown entry/horizon, no protective orders we know about). The
        safe action is to disable the affected symbol so the operator
        decides manually whether to flatten or resume.
        """
        for symbol in self._managed_symbols():
            stats = self.state.symbols.get(symbol)
            if stats is None:
                continue
            try:
                positions = await self.rest.position_risk(symbol)
            except Exception as e:
                stats.live_state = "disabled"
                stats.block_reason = f"position reconciliation failed: {e}"
                log.warning("live: %s disabled; position reconciliation failed: %s", symbol, e)
                continue
            for pos in positions:
                amt = self._float(pos.get("positionAmt"))
                if amt and abs(amt) > 0.0:
                    stats.live_state = "disabled"
                    stats.block_reason = (
                        f"pre-existing Binance position {amt:+.8f} on startup; "
                        "operator must flatten before resuming live"
                    )
                    log.warning("live: %s disabled; pre-existing position %+.8f", symbol, amt)
                    break

    def on_order_update(self, data: dict[str, Any]) -> None:
        """Reconcile Binance ``ORDER_TRADE_UPDATE`` into local state.

        Reads realized PnL from the ``rp`` field, dedupes via trade IDs,
        accumulates partial-fill PnL per close order, and fires
        ``record_trade_outcome`` exactly once when a reduce-only order
        reaches ``FILLED``. Without this the live trader has no realised
        PnL feedback into the regime guards (loss-streak / giveback /
        rolling degradation) — those would all silently no-op.
        """
        order = data.get("o") or {}
        symbol = str(order.get("s", "")).upper()
        if not symbol:
            return
        stats = self.state.symbols.get(symbol)
        if stats is None:
            return

        trade_id = str(order.get("t", ""))
        if trade_id and trade_id != "0":
            dedupe_key = (symbol, trade_id)
            if dedupe_key in self._seen_private_trade_ids:
                return
            self._seen_private_trade_ids.add(dedupe_key)
            self._seen_private_trade_order.append(dedupe_key)
            while len(self._seen_private_trade_order) > _MAX_SEEN_TRADE_IDS:
                evicted = self._seen_private_trade_order.popleft()
                self._seen_private_trade_ids.discard(evicted)

        if self.live_trade_log is not None:
            try:
                self.live_trade_log.append_order_update(data)
            except Exception as e:
                log.error("live trade log append failed: %s", e)

        side = str(order.get("S", "")).upper()
        status = str(order.get("X", "")).upper()
        client_order_id = str(order.get("c", "") or "")
        reduce_only = self._truthy(order.get("R")) or client_order_id.startswith(
            ("OPUS_FLATTEN_", "OPUS_SL_", "OPUS_TP_")
        )
        filled_qty = self._float(order.get("z"))
        avg_price = self._float(order.get("ap")) or self._float(order.get("L"))
        realized_pnl = self._float(order.get("rp"))

        if realized_pnl:
            stats.realized_pnl += realized_pnl
            record_pnl(self.state.guards, realized_pnl, symbol=symbol)

        if reduce_only:
            # Only count actual trade events; ``NEW`` / ``CANCELLED`` /
            # ``EXPIRED`` ORDER_TRADE_UPDATE rows for the resting STOP/TP
            # would otherwise inflate ``fills_count`` (every armed
            # protective fires a ``NEW`` and a ``CANCELLED`` even when no
            # trade happened).
            exec_type = str(order.get("x", "")).upper()
            if exec_type == "TRADE" or status in {"PARTIALLY_FILLED", "FILLED"}:
                stats.fills_count += 1
            close_key = (symbol, client_order_id or str(order.get("i") or ""))
            if realized_pnl:
                self._close_order_pnl[close_key] = self._close_order_pnl.get(close_key, 0.0) + realized_pnl
            if status in {"CANCELED", "CANCELLED", "EXPIRED", "REJECTED"}:
                # A reduce-only STOP/TP that partially filled then got
                # cancelled/expired would leak a ``_close_order_pnl`` entry
                # forever without this cleanup. Partial-fill PnL was
                # already recorded individually above via ``record_pnl``,
                # so discarding the accumulator is the correct action.
                self._close_order_pnl.pop(close_key, None)
                return
            if status == "FILLED":
                close_pnl = self._close_order_pnl.pop(close_key, realized_pnl)
                # Always record the outcome on a FILLED reduce-only close,
                # even when ``close_pnl == 0.0`` (a break-even round-trip).
                # Skipping break-evens would leave them invisible to
                # loss-streak / rolling-degradation counters and undercount
                # the probation trade-count.
                event = record_trade_outcome(
                    self.state,
                    symbol=symbol,
                    pnl_usd=close_pnl,
                    net_bp=self._net_bp_from_realized(stats, close_pnl, filled_qty, avg_price),
                    settings=self.runtime_settings,
                )
                if event is not None and self._on_risk_event is not None:
                    # Risk-event sink is async (Telegram I/O); dispatch
                    # via the running loop without awaiting because
                    # ``on_order_update`` itself is sync (called from
                    # the private-WS receive loop).
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self._on_risk_event(event))
                    except RuntimeError:
                        log.warning("live: risk event %s dropped (no running loop)", event.reason)
                stats.position_base = 0.0
                stats.position_entry = 0.0
                stats.unrealized_pnl = 0.0
                self._positions.pop(symbol, None)
                # When STOP_MARKET or TAKE_PROFIT_MARKET fires on the
                # exchange, the OTHER protective is still resting on
                # Binance. Cancel everything for this symbol so the next
                # entry does not trip a stale SL/TP at a price level that
                # belonged to the previous position.
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.cancel_all(symbol))
                except RuntimeError:
                    log.warning("live: cancel_all(%s) dropped after close (no running loop)", symbol)
            return

        if filled_qty > 0.0 and avg_price > 0.0 and side in {"BUY", "SELL"}:
            signed_qty = filled_qty if side == "BUY" else -filled_qty
            stats.position_base = signed_qty
            stats.position_entry = avg_price
            stats.fills_count += 1
            pos = self._positions.get(symbol)
            if pos is not None:
                pos.qty = filled_qty
                pos.entry_price = avg_price
                pos.notional_usd = filled_qty * avg_price

    def on_account_update(self, data: dict[str, Any]) -> None:
        """Reconcile Binance ``ACCOUNT_UPDATE`` position fields into UI state."""
        account = data.get("a") or {}
        for position in account.get("P") or []:
            symbol = str(position.get("s", "")).upper()
            if not symbol:
                continue
            stats = self.state.symbols.get(symbol)
            if stats is None:
                continue
            position_amt = self._float(position.get("pa"))
            entry_price = self._float(position.get("ep"))
            unrealized_pnl = self._float(position.get("up"))
            stats.position_base = position_amt
            stats.position_entry = entry_price if position_amt else 0.0
            stats.unrealized_pnl = unrealized_pnl
            if not position_amt:
                self._positions.pop(symbol, None)

    def _warm_predictor_buffer(self, symbol: str, snap: dict[str, Any]) -> None:
        """Push a snapshot row into the predictor's history without predicting.

        Called from ``on_snapshot`` early-return paths (private-WS down,
        symbol switched to paper, etc.) so that lag returns / rolling
        volatility / OFI features stay continuous through outages. The
        normal trade path uses ``Predictor.predict`` which itself pushes
        the row — we must NOT call both for the same snapshot or every
        derived feature is corrupted by a duplicated row.
        """
        if self.predictor is None or not getattr(self.predictor, "enabled", False):
            return
        try:
            self.predictor.update(symbol, snap)
        except Exception as e:
            log.warning("live: predictor.update(%s) failed: %s", symbol, e)

    @staticmethod
    def _net_bp_from_realized(stats, pnl_usd: float, qty: float, price: float) -> float:
        notional = qty * price if qty > 0 and price > 0 else float(stats.current_notional_usd or 0.0)
        if notional <= 0.0:
            return 0.0
        return (pnl_usd / notional) * 10_000.0

    @staticmethod
    def _float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes"}
        return bool(value)

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
            # Even when this symbol is currently NOT trading we still need
            # to warm the predictor's history buffer; otherwise lag /
            # rolling / OFI features compute across a discontinuity the
            # moment the symbol is enabled. Same reasoning applies on
            # private-WS outage below.
            self._warm_predictor_buffer(symbol, snap)
            return

        if not self._private_ws_ready():
            self._warm_predictor_buffer(symbol, snap)
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
        if abs(float(pred.confidence)) < self._conf_thr:
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

        try:
            await self._ensure_leverage(symbol)
        except Exception as e:
            # A REST failure on ``set_leverage`` (network, 418 rate-limit,
            # 4xx) used to propagate up to the snapshot loop where it was
            # logged generically and silently swallowed. The symbol then
            # had no ``block_reason`` / ``rejects_count`` update, so the UI
            # showed nothing while every snapshot kept retrying. Treat it
            # the same as ``place_order`` failure: reject the symbol so
            # the operator gets actionable feedback.
            self._reject_symbol(symbol, f"set leverage failed: {e}")
            return
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
            pred_confidence=float(pred.confidence),
        )
        record_order_sent(self.state.guards)
        stats.position_base = qty if side == "BUY" else -qty
        stats.position_entry = entry_price
        # ``fills_count`` is incremented in ``_maybe_close`` after the
        # round-trip completes, matching the PaperTrader semantics.
        log.info("live: OPEN %s %s qty=%.8f notional=$%.2f", side, symbol, qty, notional)
        # Arm exchange-side STOP_MARKET / TAKE_PROFIT_MARKET reduce-only
        # orders so the position closes even if the bot or its private WS
        # is offline. Best-effort: failure here does NOT roll back the
        # entry — snapshot-loop monitoring still owns the exit.
        await self._arm_protective_orders(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=entry_price,
            filters=filters,
            ts_open_ms=ts_open_ms,
        )

    async def _arm_protective_orders(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        filters: Any,
        ts_open_ms: int,
    ) -> None:
        """Place reduce-only STOP_MARKET (+ optional TAKE_PROFIT_MARKET)."""

        close_side = "SELL" if side == "BUY" else "BUY"
        sign = 1.0 if side == "BUY" else -1.0
        # Stop-loss: price moves against us by ``stop_loss_bp``.
        if self._stop_loss_bp > 0.0:
            try:
                stop_price = round_price(
                    entry_price * (1.0 - sign * self._stop_loss_bp / 10_000.0),
                    filters,
                )
                if stop_price > 0.0:
                    await self.rest.place_order(
                        symbol=symbol,
                        side=close_side,
                        order_type="STOP_MARKET",
                        quantity=qty,
                        stop_price=stop_price,
                        reduce_only=True,
                        working_type="MARK_PRICE",
                        client_order_id=f"OPUS_SL_{symbol}_{ts_open_ms}",
                    )
            except Exception as e:
                log.warning("live: arm STOP_MARKET %s failed: %s", symbol, e)
        # Take-profit (optional): price moves in our favour by ``take_profit_bp``.
        if self._take_profit_bp > 0.0:
            try:
                tp_price = round_price(
                    entry_price * (1.0 + sign * self._take_profit_bp / 10_000.0),
                    filters,
                )
                if tp_price > 0.0:
                    await self.rest.place_order(
                        symbol=symbol,
                        side=close_side,
                        order_type="TAKE_PROFIT_MARKET",
                        quantity=qty,
                        stop_price=tp_price,
                        reduce_only=True,
                        working_type="MARK_PRICE",
                        client_order_id=f"OPUS_TP_{symbol}_{ts_open_ms}",
                    )
            except Exception as e:
                log.warning("live: arm TAKE_PROFIT_MARKET %s failed: %s", symbol, e)

    def stats_dict(self) -> dict:
        """Snapshot of live-trader state for the UI trader panel."""

        live_stats = [
            s
            for s in self.state.symbols.values()
            if s.execution_mode == "live"
            or s.live_state in {"probation_live", "active_live", "cooldown", "awaiting_operator"}
        ]
        n_trades = sum(s.live_trade_count for s in live_stats)
        wins = sum(s.live_wins for s in live_stats)
        losses = sum(s.live_losses for s in live_stats)
        win_rate = (wins / n_trades) if n_trades else 0.0
        return {
            "open_positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "qty": p.qty,
                    "notional_usd": p.notional_usd,
                    "entry_price": p.entry_price,
                    "ts_open_ms": p.ts_open_ms,
                    "horizon": f"{p.horizon_ms / 1000:g}s",
                    "confidence": p.pred_confidence,
                }
                for p in self._positions.values()
            ],
            "daily_trades": n_trades,
            "daily_wins": wins,
            "daily_losses": losses,
            "daily_win_rate": win_rate,
            "daily_pnl_usd": self.state.guards.daily_pnl,
            "horizon": settings.trade_horizon,
            "conf_threshold": settings.trade_conf_threshold,
            "notional_usd": self.runtime_settings.probation_notional_usd,
            "stop_loss_bp": self._stop_loss_bp,
            "take_profit_bp": self._take_profit_bp,
            "allowed_symbols": sorted(s.symbol for s in live_stats),
        }

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
        # would re-fire the same exit. The Binance position itself, if
        # the REST call failed, will be reconciled on next startup or
        # via ``emergency_flatten``.
        self._positions.pop(symbol, None)

        # PnL accounting flows EXCLUSIVELY through ``on_order_update``
        # using Binance's actual ``rp`` (realized_pnl) field. Recording
        # an estimated ``pnl_usd`` here in addition to the WS-driven
        # path would double-count: ``_flatten`` places a reduce-only
        # MARKET whose fill arrives back via ORDER_TRADE_UPDATE, and
        # ``on_order_update`` already calls ``record_pnl`` and
        # ``record_trade_outcome`` on the resulting realised PnL. If the
        # private WS is briefly offline, Binance replays events on
        # reconnect and our trade-id dedupe lets the missed fills land
        # exactly once. If the REST flatten itself failed, no fill was
        # generated and the missing PnL accurately reflects the
        # never-closed position; ``reconcile_open_positions`` will
        # disable the symbol on next start.
        #
        # Cancel the resting STOP_MARKET / TAKE_PROFIT_MARKET protective
        # orders BEFORE sending the reduce-only MARKET flatten. If we did
        # not, those orders would survive on Binance and could trigger
        # against the next position opened on this symbol at a stale
        # price level — an uncontrolled exit with garbage PnL.
        await self.cancel_all(symbol)
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
        desired = float(self.runtime_settings.leverage)
        current = self._leveraged_symbols.get(symbol)
        if current is not None and abs(current - desired) < 1e-9:
            return
        await self.rest.set_leverage(symbol, desired)
        self._leveraged_symbols[symbol] = desired

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
