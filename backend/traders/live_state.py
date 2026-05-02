"""Per-symbol live execution state machine."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from backend.settings_store import RuntimeSettings


class ExecutionMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class LiveSymbolState(str, Enum):
    PAPER = "paper"
    WATCH_ONLY = "watch_only"
    PROBATION_LIVE = "probation_live"
    ACTIVE_LIVE = "active_live"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"


@dataclass
class LiveSymbolRuntime:
    """Mutable live state for one symbol."""

    symbol: str
    execution_mode: ExecutionMode = ExecutionMode.PAPER
    live_state: LiveSymbolState = LiveSymbolState.PAPER
    live_trade_count: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    realized_pnl_usd: float = 0.0
    sum_net_bp: float = 0.0
    cooldown_until: float = 0.0
    current_notional_usd: float = 0.0
    block_reason: str = ""

    def arm_live(self, settings: RuntimeSettings) -> None:
        """Switch the symbol into live probation."""

        self.execution_mode = ExecutionMode.LIVE
        self.live_state = LiveSymbolState.PROBATION_LIVE
        self.live_trade_count = 0
        self.wins = 0
        self.losses = 0
        self.consecutive_losses = 0
        self.realized_pnl_usd = 0.0
        self.sum_net_bp = 0.0
        self.cooldown_until = 0.0
        self.current_notional_usd = settings.probation_notional_usd
        self.block_reason = ""

    def set_paper(self) -> None:
        self.execution_mode = ExecutionMode.PAPER
        self.live_state = LiveSymbolState.PAPER
        self.current_notional_usd = 0.0
        self.block_reason = ""

    def disable(self, reason: str) -> None:
        self.execution_mode = ExecutionMode.PAPER
        self.live_state = LiveSymbolState.DISABLED
        self.current_notional_usd = 0.0
        self.block_reason = reason

    def _cooldown(self, settings: RuntimeSettings, reason: str) -> None:
        self.live_state = LiveSymbolState.COOLDOWN
        self.current_notional_usd = 0.0
        self.cooldown_until = time.time() + settings.cooldown_hours * 3600.0
        self.block_reason = reason

    def _promote(self, settings: RuntimeSettings) -> None:
        self.live_state = LiveSymbolState.ACTIVE_LIVE
        self.current_notional_usd = settings.active_notional_usd
        self.block_reason = ""

    def record_closed_trade(self, *, net_bp: float, pnl_usd: float, settings: RuntimeSettings) -> None:
        """Update probation/active counters after a closed live trade."""

        if self.live_state not in {LiveSymbolState.PROBATION_LIVE, LiveSymbolState.ACTIVE_LIVE}:
            return

        self.live_trade_count += 1
        self.sum_net_bp += net_bp
        self.realized_pnl_usd += pnl_usd
        if pnl_usd > 0:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1

        if self.realized_pnl_usd <= -abs(settings.symbol_loss_limit_usd):
            self._cooldown(settings, f"symbol loss limit hit ({self.realized_pnl_usd:.4f})")
            return

        if self.consecutive_losses >= 4:
            self._cooldown(settings, "consecutive losses limit hit")
            return

        if self.live_state == LiveSymbolState.PROBATION_LIVE and self.live_trade_count >= settings.probation_trades:
            probation_ok = (
                self.wins >= 4
                and self.sum_net_bp > 0.0
                and self.consecutive_losses < 3
                and self.realized_pnl_usd > -0.10
            )
            if probation_ok:
                self._promote(settings)
            else:
                self._cooldown(
                    settings,
                    (
                        "probation failed "
                        f"(wins={self.wins}, sum_net_bp={self.sum_net_bp:.2f}, "
                        f"pnl={self.realized_pnl_usd:.4f})"
                    ),
                )

    def can_trade(self, *, now: float, tradeability_ok: bool) -> tuple[bool, str]:
        """Return whether new live entries are allowed for this symbol."""

        if self.execution_mode != ExecutionMode.LIVE:
            return False, "symbol is not live"
        if self.live_state == LiveSymbolState.COOLDOWN:
            if self.cooldown_until > now:
                return False, "symbol in cooldown"
            self.live_state = LiveSymbolState.WATCH_ONLY
            self.block_reason = "cooldown expired"
            return False, self.block_reason
        if self.live_state not in {LiveSymbolState.PROBATION_LIVE, LiveSymbolState.ACTIVE_LIVE}:
            return False, self.block_reason or f"symbol state is {self.live_state.value}"
        if not tradeability_ok:
            return False, "tradeability filter blocked"
        return True, ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "execution_mode": self.execution_mode.value,
            "live_state": self.live_state.value,
            "live_trade_count": self.live_trade_count,
            "live_wins": self.wins,
            "live_losses": self.losses,
            "consecutive_losses": self.consecutive_losses,
            "realized_pnl_usd": self.realized_pnl_usd,
            "sum_net_bp": self.sum_net_bp,
            "cooldown_until": self.cooldown_until,
            "current_notional_usd": self.current_notional_usd,
            "block_reason": self.block_reason,
        }
