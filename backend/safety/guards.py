"""Centralised pre-trade safety checks.

Every order — paper or live — must call ``Guards.check()`` first. If it
returns a non-empty reason, the order is rejected and that reason is
surfaced in the UI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

from backend.log import get_logger
from backend.state import AppState, Guards

log = get_logger(__name__)


@dataclass
class OrderIntent:
    symbol: str
    side: str  # "BUY" | "SELL"
    qty: float
    price: float | None
    notional_usd: float


def _today_utc_bucket() -> int:
    now = datetime.now(tz=UTC)
    return int(now.strftime("%Y%m%d"))


def _minute_bucket() -> int:
    return int(time.time() // 60)


def _prune_pnl_events(guards: Guards, now: float | None = None) -> None:
    now = time.time() if now is None else now
    cutoff = now - 12 * 3600.0
    guards.pnl_events = [event for event in guards.pnl_events if event[0] >= cutoff]


def pnl_12h(guards: Guards) -> float:
    _prune_pnl_events(guards)
    return sum(delta for _ts, _symbol, delta in guards.pnl_events)


def symbol_pnl_12h(guards: Guards, symbol: str) -> float:
    _prune_pnl_events(guards)
    target = symbol.upper()
    return sum(delta for _ts, sym, delta in guards.pnl_events if sym == target)


def _is_live_state(stats) -> bool:
    return getattr(stats, "execution_mode", "paper") == "live" and getattr(stats, "live_state", "paper") in {
        "probation_live",
        "active_live",
    }


def reset_if_new_day(guards: Guards) -> None:
    today = _today_utc_bucket()
    if guards.day_bucket_utc != today:
        if guards.day_bucket_utc != 0:
            log.info("guards: new UTC day, resetting daily counters")
        guards.day_bucket_utc = today
        guards.daily_pnl = 0.0
        guards.daily_orders = 0
        guards.risk_trade_count = 0
        guards.pnl_peak_usd = 0.0
        guards.pnl_drawdown_pct = 0.0
        guards.risk_events = []
        # NOTE: do NOT clear emergency_stopped here. Operator-initiated stops
        # (and other persistent halts) must survive the UTC day boundary and
        # only be cleared explicitly via /api/emergency/clear. Daily
        # loss-triggered emergencies are effectively re-evaluated every
        # check() because daily_pnl is reset to 0 above.


def check(state: AppState, intent: OrderIntent, mode: str) -> str | None:
    """Return None if the order is allowed, or a reason string if rejected."""
    g = state.guards
    reset_if_new_day(g)

    if g.emergency_stopped:
        return f"emergency stop active: {g.emergency_reason}"

    if mode == "live":
        if not state.binance_connected:
            return "binance not connected"
        age_ms = (time.time() - state.binance_last_msg_ts) * 1000.0
        if state.binance_last_msg_ts > 0 and age_ms > g.ws_stale_ms:
            return f"binance ws stale ({age_ms:.0f}ms > {g.ws_stale_ms}ms)"

    if intent.notional_usd > g.max_position_usd:
        return f"notional {intent.notional_usd:.2f} > max_position_usd {g.max_position_usd:.2f}"

    if g.daily_pnl <= -abs(g.daily_loss_limit_usd):
        trip_emergency(g, f"daily loss limit hit ({g.daily_pnl:.2f})")
        return g.emergency_reason

    loss_12h = pnl_12h(g)
    if loss_12h <= -abs(g.loss_12h_limit_usd):
        trip_emergency(g, f"12h loss limit hit ({loss_12h:.2f})")
        return g.emergency_reason

    sym_loss = symbol_pnl_12h(g, intent.symbol)
    if sym_loss <= -abs(g.symbol_loss_limit_usd):
        return f"symbol loss limit hit for {intent.symbol.upper()} ({sym_loss:.2f})"

    # Order rate limit (per minute, sliding bucket)
    mb = _minute_bucket()
    if mb != g.last_order_minute_bucket:
        g.last_order_minute_bucket = mb
        g.orders_this_minute = 0
    if g.orders_this_minute + 1 > g.max_orders_per_min:
        return f"order rate limit ({g.orders_this_minute}/{g.max_orders_per_min} per min)"

    if mode == "live":
        live_symbols = {s.symbol.upper() for s in state.symbols.values() if _is_live_state(s)}
        if intent.symbol.upper() not in live_symbols and len(live_symbols) >= g.max_live_symbols:
            return f"live symbol limit ({len(live_symbols)} >= {g.max_live_symbols})"

    return None


def record_order_sent(guards: Guards) -> None:
    mb = _minute_bucket()
    if mb != guards.last_order_minute_bucket:
        guards.last_order_minute_bucket = mb
        guards.orders_this_minute = 0
    guards.orders_this_minute += 1
    guards.daily_orders += 1


def record_pnl(guards: Guards, delta_usd: float, *, symbol: str = "", ts: float | None = None) -> None:
    reset_if_new_day(guards)
    guards.daily_pnl += delta_usd
    if symbol:
        guards.pnl_events.append((time.time() if ts is None else ts, symbol.upper(), delta_usd))
        _prune_pnl_events(guards)
    if guards.daily_pnl <= -abs(guards.daily_loss_limit_usd):
        trip_emergency(guards, f"daily loss limit hit ({guards.daily_pnl:.2f})")


def trip_emergency(guards: Guards, reason: str) -> None:
    if not guards.emergency_stopped:
        log.warning("guards: EMERGENCY STOP: %s", reason)
    guards.emergency_stopped = True
    guards.emergency_reason = reason


def clear_emergency(guards: Guards) -> None:
    log.info("guards: emergency stop cleared by operator")
    guards.emergency_stopped = False
    guards.emergency_reason = ""
