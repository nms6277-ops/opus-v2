"""Global, process-wide runtime state.

Single source of truth for the FastAPI/UI to read and for workers to update.
Thread-safe under asyncio (no real threads are used in hot paths).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from backend.config import Mode, settings


@dataclass
class SymbolStats:
    """Per-symbol runtime stats shown in the UI."""

    symbol: str
    position_size_usd: float
    started_at: float
    added_at: float

    # Connection / data quality
    ws_last_msg_ts: float = 0.0
    ws_reconnects: int = 0
    ws_sequence_gaps: int = 0
    snapshots_written: int = 0
    last_snapshot_ts: float = 0.0

    # Market snapshot (kept up to date by collector for UI)
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread_bp: float = 0.0
    microprice: float = 0.0
    last_trade_price: float = 0.0
    last_trade_qty: float = 0.0

    # Trading stats (paper or live)
    position_base: float = 0.0
    position_entry: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fills_count: int = 0
    rejects_count: int = 0

    # Per-symbol execution state for VPS live mode.
    execution_mode: str = "paper"
    live_state: str = "paper"
    live_trade_count: int = 0
    live_wins: int = 0
    live_losses: int = 0
    consecutive_losses: int = 0
    recent_trade_net_bp: list[float] = field(default_factory=list)
    pnl_peak_usd: float = 0.0
    pnl_drawdown_pct: float = 0.0
    rolling_win_rate: float = 1.0
    rolling_sum_net_bp: float = 0.0
    symbol_realized_pnl_12h: float = 0.0
    current_notional_usd: float = 0.0
    expected_gross_bp: float = 0.0
    cooldown_until: float = 0.0
    block_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "position_size_usd": self.position_size_usd,
            "started_at": self.started_at,
            "added_at": self.added_at,
            "ws_last_msg_ts": self.ws_last_msg_ts,
            "ws_reconnects": self.ws_reconnects,
            "ws_sequence_gaps": self.ws_sequence_gaps,
            "snapshots_written": self.snapshots_written,
            "last_snapshot_ts": self.last_snapshot_ts,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread_bp": self.spread_bp,
            "microprice": self.microprice,
            "last_trade_price": self.last_trade_price,
            "last_trade_qty": self.last_trade_qty,
            "position_base": self.position_base,
            "position_entry": self.position_entry,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "fills_count": self.fills_count,
            "rejects_count": self.rejects_count,
            "execution_mode": self.execution_mode,
            "live_state": self.live_state,
            "live_trade_count": self.live_trade_count,
            "live_wins": self.live_wins,
            "live_losses": self.live_losses,
            "consecutive_losses": self.consecutive_losses,
            "pnl_peak_usd": self.pnl_peak_usd,
            "pnl_drawdown_pct": self.pnl_drawdown_pct,
            "rolling_win_rate": self.rolling_win_rate,
            "rolling_sum_net_bp": self.rolling_sum_net_bp,
            "symbol_realized_pnl_12h": self.symbol_realized_pnl_12h,
            "current_notional_usd": self.current_notional_usd,
            "expected_gross_bp": self.expected_gross_bp,
            "cooldown_until": self.cooldown_until,
            "block_reason": self.block_reason,
        }


@dataclass
class Guards:
    """Live-editable safety guards. Loaded from settings on startup."""

    daily_loss_limit_usd: float = settings.daily_loss_limit_usd
    max_position_usd: float = settings.max_position_usd
    max_live_symbols: int = settings.max_live_symbols
    max_orders_per_min: int = settings.max_orders_per_min
    ws_stale_ms: int = settings.ws_stale_ms
    loss_12h_limit_usd: float = settings.hard_12h_loss_usd
    symbol_loss_limit_usd: float = settings.hard_symbol_loss_usd

    # Runtime counters (reset daily)
    daily_pnl: float = 0.0
    daily_orders: int = 0
    last_order_minute_bucket: int = 0
    orders_this_minute: int = 0
    day_bucket_utc: int = 0
    emergency_stopped: bool = False
    emergency_reason: str = ""
    pnl_events: list[tuple[float, str, float]] = field(default_factory=list)
    risk_trade_count: int = 0
    pnl_peak_usd: float = 0.0
    pnl_drawdown_pct: float = 0.0
    risk_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AppState:
    """Single app-wide state object. Locked per mutation."""

    mode: Mode = settings.mode
    started_at: float = field(default_factory=time.time)

    symbols: dict[str, SymbolStats] = field(default_factory=dict)
    guards: Guards = field(default_factory=Guards)

    # Connection health
    binance_connected: bool = False
    binance_last_msg_ts: float = 0.0
    binance_private_connected: bool = False
    binance_private_last_msg_ts: float = 0.0
    bybit_connected: bool = False
    bybit_last_msg_ts: float = 0.0

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def snapshot(self) -> dict[str, Any]:
        """Serializable snapshot for the UI."""
        return {
            "mode": self.mode.value,
            "started_at": self.started_at,
            "now": time.time(),
            "binance_connected": self.binance_connected,
            "binance_last_msg_ts": self.binance_last_msg_ts,
            "binance_private_connected": self.binance_private_connected,
            "binance_private_last_msg_ts": self.binance_private_last_msg_ts,
            "bybit_connected": self.bybit_connected,
            "bybit_last_msg_ts": self.bybit_last_msg_ts,
            "guards": {
                "daily_loss_limit_usd": self.guards.daily_loss_limit_usd,
                "max_position_usd": self.guards.max_position_usd,
                "max_live_symbols": self.guards.max_live_symbols,
                "max_orders_per_min": self.guards.max_orders_per_min,
                "ws_stale_ms": self.guards.ws_stale_ms,
                "loss_12h_limit_usd": self.guards.loss_12h_limit_usd,
                "symbol_loss_limit_usd": self.guards.symbol_loss_limit_usd,
                "daily_pnl": self.guards.daily_pnl,
                "pnl_12h": sum(e[2] for e in self.guards.pnl_events),
                "risk_trade_count": self.guards.risk_trade_count,
                "pnl_peak_usd": self.guards.pnl_peak_usd,
                "pnl_drawdown_pct": self.guards.pnl_drawdown_pct,
                "daily_orders": self.guards.daily_orders,
                "emergency_stopped": self.guards.emergency_stopped,
                "emergency_reason": self.guards.emergency_reason,
                "risk_events": self.guards.risk_events[-20:],
            },
            "symbols": [s.to_dict() for s in self.symbols.values()],
        }


# Module-level singleton
app_state = AppState()
