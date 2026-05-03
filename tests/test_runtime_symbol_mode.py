"""Runtime.set_symbol_execution_mode must flatten live positions on live->paper."""

import time

import pytest

from backend.config import Mode
from backend.runtime import Runtime
from backend.state import AppState, SymbolStats


class _RecordingLiveTrader:
    def __init__(self) -> None:
        self.flatten_calls: list[str] = []
        self.cancel_calls: list[str] = []
        self.runtime_settings = None

    async def flatten_symbol(self, symbol: str) -> None:
        self.flatten_calls.append(symbol)

    async def cancel_all(self, symbol: str) -> None:
        self.cancel_calls.append(symbol)


def _state_with(symbol: str = "UBUSDT") -> AppState:
    state = AppState()
    state.mode = Mode.LIVE
    stats = SymbolStats(
        symbol=symbol,
        position_size_usd=6.0,
        started_at=time.time(),
        added_at=time.time(),
    )
    stats.execution_mode = "live"
    stats.live_state = "active_live"
    stats.current_notional_usd = 6.0
    state.symbols[symbol] = stats
    return state


@pytest.mark.asyncio
async def test_set_symbol_execution_mode_paper_flattens_live_position(monkeypatch):
    state = _state_with("UBUSDT")
    rt = Runtime.__new__(Runtime)
    rt.state = state

    from backend import runtime as runtime_mod

    fake_live = _RecordingLiveTrader()
    monkeypatch.setattr(runtime_mod, "LiveTrader", _RecordingLiveTrader)
    rt._trader = fake_live

    result = await rt.set_symbol_execution_mode("UBUSDT", "paper")

    assert result["execution_mode"] == "paper"
    assert state.symbols["UBUSDT"].execution_mode == "paper"
    assert state.symbols["UBUSDT"].live_state == "paper"
    assert fake_live.flatten_calls == ["UBUSDT"], (
        "live->paper switch must flatten the real Binance position"
    )


@pytest.mark.asyncio
async def test_set_symbol_execution_mode_live_does_not_call_flatten(monkeypatch):
    """Promoting paper -> live must NOT call flatten (no position to close)."""

    state = AppState()
    state.mode = Mode.LIVE
    stats = SymbolStats(
        symbol="UBUSDT",
        position_size_usd=6.0,
        started_at=time.time(),
        added_at=time.time(),
    )
    stats.execution_mode = "paper"
    stats.live_state = "paper"
    state.symbols["UBUSDT"] = stats

    rt = Runtime.__new__(Runtime)
    rt.state = state
    from backend import runtime as runtime_mod
    from backend.settings_store import RuntimeSettings

    fake_live = _RecordingLiveTrader()
    monkeypatch.setattr(runtime_mod, "LiveTrader", _RecordingLiveTrader)
    rt._trader = fake_live
    rt.runtime_settings = RuntimeSettings()

    await rt.set_symbol_execution_mode("UBUSDT", "live")

    assert state.symbols["UBUSDT"].execution_mode == "live"
    assert fake_live.flatten_calls == []
    assert fake_live.cancel_calls == []
