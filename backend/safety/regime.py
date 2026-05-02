"""Post-trade regime guards.

These guards protect accumulated edge: they run after a trade is closed and
pause trading when the recent trade stream stops behaving like the validated
paper/backtest regime.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from backend.log import get_logger
from backend.safety.guards import trip_emergency
from backend.settings_store import RuntimeSettings
from backend.state import AppState

log = get_logger(__name__)


@dataclass(frozen=True)
class RiskEvent:
    scope: str
    reason: str
    symbol: str = ""
    ts: float = 0.0
    pnl_usd: float = 0.0
    peak_usd: float = 0.0
    drawdown_pct: float = 0.0
    rolling_win_rate: float = 1.0
    rolling_sum_net_bp: float = 0.0
    consecutive_losses: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _pause_symbol(state: AppState, symbol: str, reason: str) -> None:
    stats = state.symbols.get(symbol)
    if stats is None:
        return
    stats.live_state = "awaiting_operator"
    stats.current_notional_usd = 0.0
    stats.block_reason = reason


def _append_event(state: AppState, event: RiskEvent) -> None:
    state.guards.risk_events.append(event.to_dict())
    state.guards.risk_events = state.guards.risk_events[-200:]
    log.warning("regime guard: %s %s", event.scope, event.reason)


def _drawdown_pct(pnl_usd: float, peak_usd: float) -> float:
    if peak_usd <= 0.0:
        return 0.0
    return max(0.0, (peak_usd - pnl_usd) / peak_usd)


def record_trade_outcome(
    state: AppState,
    *,
    symbol: str,
    pnl_usd: float,
    net_bp: float,
    settings: RuntimeSettings,
    ts: float | None = None,
) -> RiskEvent | None:
    """Update post-trade risk state and return a pause/stop event if triggered."""

    symbol = symbol.upper()
    stats = state.symbols.get(symbol)
    if stats is None:
        return None

    now = time.time() if ts is None else ts
    g = state.guards
    g.risk_trade_count += 1
    g.pnl_peak_usd = max(g.pnl_peak_usd, g.daily_pnl)
    g.pnl_drawdown_pct = _drawdown_pct(g.daily_pnl, g.pnl_peak_usd)

    stats.live_trade_count += 1
    if net_bp > 0:
        stats.live_wins += 1
        stats.consecutive_losses = 0
    else:
        stats.live_losses += 1
        stats.consecutive_losses += 1
    stats.symbol_realized_pnl_12h += pnl_usd
    stats.pnl_peak_usd = max(stats.pnl_peak_usd, stats.realized_pnl)
    stats.pnl_drawdown_pct = _drawdown_pct(stats.realized_pnl, stats.pnl_peak_usd)
    stats.recent_trade_net_bp.append(float(net_bp))
    stats.recent_trade_net_bp = stats.recent_trade_net_bp[-settings.rolling_guard_trades:]
    wins = sum(1 for x in stats.recent_trade_net_bp if x > 0.0)
    stats.rolling_win_rate = wins / len(stats.recent_trade_net_bp)
    stats.rolling_sum_net_bp = sum(stats.recent_trade_net_bp)

    if (
        g.risk_trade_count >= settings.global_guard_min_trades
        and g.pnl_peak_usd > 0.0
        and g.pnl_drawdown_pct >= settings.global_profit_giveback_pct
    ):
        reason = (
            f"global profit giveback {g.pnl_drawdown_pct * 100:.1f}% "
            f"from peak ${g.pnl_peak_usd:.4f}"
        )
        trip_emergency(g, reason)
        event = RiskEvent(
            scope="global",
            symbol="",
            reason=reason,
            ts=now,
            pnl_usd=g.daily_pnl,
            peak_usd=g.pnl_peak_usd,
            drawdown_pct=g.pnl_drawdown_pct,
        )
        _append_event(state, event)
        return event

    if (
        stats.live_trade_count >= settings.symbol_guard_min_trades
        and stats.pnl_peak_usd > 0.0
        and stats.pnl_drawdown_pct >= settings.symbol_profit_giveback_pct
    ):
        reason = (
            f"symbol profit giveback drawdown {stats.pnl_drawdown_pct * 100:.1f}% "
            f"from peak ${stats.pnl_peak_usd:.4f}"
        )
        _pause_symbol(state, symbol, reason)
        event = RiskEvent(
            scope="symbol",
            symbol=symbol,
            reason=reason,
            ts=now,
            pnl_usd=stats.realized_pnl,
            peak_usd=stats.pnl_peak_usd,
            drawdown_pct=stats.pnl_drawdown_pct,
            rolling_win_rate=stats.rolling_win_rate,
            rolling_sum_net_bp=stats.rolling_sum_net_bp,
            consecutive_losses=stats.consecutive_losses,
        )
        _append_event(state, event)
        return event

    if stats.consecutive_losses >= settings.loss_streak_limit:
        reason = f"loss streak {stats.consecutive_losses} trades"
        _pause_symbol(state, symbol, reason)
        event = RiskEvent(
            scope="symbol",
            symbol=symbol,
            reason=reason,
            ts=now,
            pnl_usd=stats.realized_pnl,
            peak_usd=stats.pnl_peak_usd,
            drawdown_pct=stats.pnl_drawdown_pct,
            rolling_win_rate=stats.rolling_win_rate,
            rolling_sum_net_bp=stats.rolling_sum_net_bp,
            consecutive_losses=stats.consecutive_losses,
        )
        _append_event(state, event)
        return event

    if (
        len(stats.recent_trade_net_bp) >= settings.rolling_guard_trades
        and stats.rolling_win_rate < settings.rolling_min_win_rate
        and stats.rolling_sum_net_bp < 0.0
        and (
            stats.rolling_sum_net_bp <= -abs(settings.rolling_min_loss_net_bp)
            or stats.pnl_drawdown_pct >= settings.rolling_min_drawdown_pct
        )
    ):
        reason = (
            f"rolling degradation win_rate={stats.rolling_win_rate * 100:.1f}% "
            f"sum_net_bp={stats.rolling_sum_net_bp:.2f} "
            f"drawdown={stats.pnl_drawdown_pct * 100:.1f}%"
        )
        _pause_symbol(state, symbol, reason)
        event = RiskEvent(
            scope="symbol",
            symbol=symbol,
            reason=reason,
            ts=now,
            pnl_usd=stats.realized_pnl,
            peak_usd=stats.pnl_peak_usd,
            drawdown_pct=stats.pnl_drawdown_pct,
            rolling_win_rate=stats.rolling_win_rate,
            rolling_sum_net_bp=stats.rolling_sum_net_bp,
            consecutive_losses=stats.consecutive_losses,
        )
        _append_event(state, event)
        return event

    return None
