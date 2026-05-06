"""Persisted UI-editable runtime trading settings.

The `.env` file owns secrets and hard maximums. This module owns live settings
that operators can change from the UI, while clamping them to those hard caps.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeSettings:
    """Runtime trading parameters editable from the UI."""

    leverage: int = 10
    max_live_symbols: int = 5
    daily_loss_limit_usd: float = 2.0
    loss_12h_limit_usd: float = 2.0
    symbol_loss_limit_usd: float = 0.30
    probation_notional_usd: float = 6.0
    active_notional_usd: float = 20.0
    min_expected_gross_bp: float = 12.0
    global_profit_giveback_pct: float = 0.30
    symbol_profit_giveback_pct: float = 0.30
    loss_streak_limit: int = 4
    rolling_guard_trades: int = 15
    rolling_min_win_rate: float = 0.35
    rolling_min_loss_net_bp: float = 50.0
    rolling_min_drawdown_pct: float = 0.10
    global_guard_min_trades: int = 20
    symbol_guard_min_trades: int = 10
    probation_trades: int = 7
    cooldown_hours: int = 12


class SettingsLimitError(ValueError):
    """Raised when UI runtime settings exceed environment hard caps."""


@dataclass(frozen=True)
class SettingsHardCaps:
    """Hard upper bounds loaded from environment/config."""

    hard_max_leverage: int = 10
    hard_max_live_symbols: int = 5
    hard_daily_loss_usd: float = 2.0
    hard_12h_loss_usd: float = 2.0
    hard_symbol_loss_usd: float = 0.30
    hard_notional_usd: float = 20.0


def _coerce_settings(data: dict[str, Any]) -> RuntimeSettings:
    defaults = asdict(RuntimeSettings())
    known = {k: data.get(k, v) for k, v in defaults.items()}
    return RuntimeSettings(
        leverage=max(1, int(known["leverage"])),
        max_live_symbols=max(1, int(known["max_live_symbols"])),
        daily_loss_limit_usd=max(0.0, float(known["daily_loss_limit_usd"])),
        loss_12h_limit_usd=max(0.0, float(known["loss_12h_limit_usd"])),
        symbol_loss_limit_usd=max(0.0, float(known["symbol_loss_limit_usd"])),
        probation_notional_usd=max(0.0, float(known["probation_notional_usd"])),
        active_notional_usd=max(0.0, float(known["active_notional_usd"])),
        min_expected_gross_bp=max(0.0, float(known["min_expected_gross_bp"])),
        global_profit_giveback_pct=max(0.0, float(known["global_profit_giveback_pct"])),
        symbol_profit_giveback_pct=max(0.0, float(known["symbol_profit_giveback_pct"])),
        loss_streak_limit=max(1, int(known["loss_streak_limit"])),
        rolling_guard_trades=max(1, int(known["rolling_guard_trades"])),
        rolling_min_win_rate=max(0.0, min(1.0, float(known["rolling_min_win_rate"]))),
        rolling_min_loss_net_bp=max(0.0, float(known["rolling_min_loss_net_bp"])),
        rolling_min_drawdown_pct=max(0.0, min(1.0, float(known["rolling_min_drawdown_pct"]))),
        global_guard_min_trades=max(1, int(known["global_guard_min_trades"])),
        symbol_guard_min_trades=max(1, int(known["symbol_guard_min_trades"])),
        probation_trades=max(1, int(known["probation_trades"])),
        cooldown_hours=max(1, int(known["cooldown_hours"])),
    )


def _limit_errors(runtime_settings: RuntimeSettings, hard_caps: SettingsHardCaps) -> list[str]:
    checks = (
        ("leverage", runtime_settings.leverage, "OPUS_HARD_MAX_LEVERAGE", hard_caps.hard_max_leverage),
        (
            "max_live_symbols",
            runtime_settings.max_live_symbols,
            "OPUS_HARD_MAX_LIVE_SYMBOLS",
            hard_caps.hard_max_live_symbols,
        ),
        (
            "daily_loss_limit_usd",
            runtime_settings.daily_loss_limit_usd,
            "OPUS_HARD_DAILY_LOSS_USD",
            hard_caps.hard_daily_loss_usd,
        ),
        (
            "loss_12h_limit_usd",
            runtime_settings.loss_12h_limit_usd,
            "OPUS_HARD_12H_LOSS_USD",
            hard_caps.hard_12h_loss_usd,
        ),
        (
            "symbol_loss_limit_usd",
            runtime_settings.symbol_loss_limit_usd,
            "OPUS_HARD_SYMBOL_LOSS_USD",
            hard_caps.hard_symbol_loss_usd,
        ),
        (
            "probation_notional_usd",
            runtime_settings.probation_notional_usd,
            "OPUS_HARD_MAX_NOTIONAL_USD",
            hard_caps.hard_notional_usd,
        ),
        (
            "active_notional_usd",
            runtime_settings.active_notional_usd,
            "OPUS_HARD_MAX_NOTIONAL_USD",
            hard_caps.hard_notional_usd,
        ),
    )
    return [
        f"{field}={value} exceeds {env_name}={limit}"
        for field, value, env_name, limit in checks
        if value > limit
    ]


def _clamp_to_hard_caps(
    runtime_settings: RuntimeSettings,
    hard_caps: SettingsHardCaps,
) -> RuntimeSettings:
    """Clamp settings to environment-level hard caps."""

    return replace(
        runtime_settings,
        leverage=min(runtime_settings.leverage, hard_caps.hard_max_leverage),
        max_live_symbols=min(runtime_settings.max_live_symbols, hard_caps.hard_max_live_symbols),
        daily_loss_limit_usd=min(runtime_settings.daily_loss_limit_usd, hard_caps.hard_daily_loss_usd),
        loss_12h_limit_usd=min(runtime_settings.loss_12h_limit_usd, hard_caps.hard_12h_loss_usd),
        symbol_loss_limit_usd=min(runtime_settings.symbol_loss_limit_usd, hard_caps.hard_symbol_loss_usd),
        probation_notional_usd=min(runtime_settings.probation_notional_usd, hard_caps.hard_notional_usd),
        active_notional_usd=min(runtime_settings.active_notional_usd, hard_caps.hard_notional_usd),
    )


class SettingsStore:
    """JSON-backed runtime settings store."""

    def __init__(self, path: Path, hard_caps: SettingsHardCaps) -> None:
        self.path = Path(path)
        self.hard_caps = hard_caps
        self._settings: RuntimeSettings | None = None

    @staticmethod
    def clamp_to_hard_caps(
        runtime_settings: RuntimeSettings,
        hard_caps: SettingsHardCaps,
    ) -> RuntimeSettings:
        return _clamp_to_hard_caps(runtime_settings, hard_caps)

    @staticmethod
    def validate_hard_caps(
        runtime_settings: RuntimeSettings,
        hard_caps: SettingsHardCaps,
    ) -> None:
        errors = _limit_errors(runtime_settings, hard_caps)
        if errors:
            raise SettingsLimitError("; ".join(errors))

    def load(self) -> RuntimeSettings:
        """Load settings from disk, writing safe defaults if missing/corrupt."""

        if self._settings is not None:
            return self._settings
        runtime_settings = RuntimeSettings()
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    runtime_settings = _coerce_settings(data)
            except (OSError, ValueError, TypeError):
                runtime_settings = RuntimeSettings()
        self._settings = self.clamp_to_hard_caps(runtime_settings, self.hard_caps)
        if not self.path.exists():
            self.save(self._settings)
        return self._settings

    def save(self, runtime_settings: RuntimeSettings) -> RuntimeSettings:
        """Persist validated settings and return the stored value."""

        self.validate_hard_caps(runtime_settings, self.hard_caps)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(runtime_settings), indent=2), encoding="utf-8")
        tmp.replace(self.path)
        self._settings = runtime_settings
        return runtime_settings

    def snapshot(self) -> dict[str, Any]:
        return asdict(self.load())
