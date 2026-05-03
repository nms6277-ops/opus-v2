"""WS watchdog must flatten live positions, not just cancel orders."""

import asyncio
import time

import pytest

from backend.config import Mode
from backend.runtime import Runtime
from backend.state import AppState


class _RecordingLiveTrader:
    """Mimics LiveTrader's interface for the watchdog branch."""

    def __init__(self) -> None:
        self.cancel_calls: list[str] = []
        self.emergency_flatten_calls = 0

    async def cancel_all(self, symbol: str) -> None:
        self.cancel_calls.append(symbol)

    async def emergency_flatten(self) -> None:
        self.emergency_flatten_calls += 1


class _RecordingPaperTrader:
    """Paper traders should still get cancel_all per symbol."""

    def __init__(self) -> None:
        self.cancel_calls: list[str] = []

    async def cancel_all(self, symbol: str) -> None:
        self.cancel_calls.append(symbol)


@pytest.mark.asyncio
async def test_ws_watchdog_emergency_flattens_live_positions(monkeypatch):
    state = AppState()
    state.mode = Mode.LIVE
    state.binance_last_msg_ts = time.time() - 60.0  # very stale
    state.guards.ws_stale_ms = 1500.0

    rt = Runtime.__new__(Runtime)  # bypass __init__ side-effects
    rt.state = state
    rt._stopped = False
    rt.symbols = {"UBUSDT": object(), "AIOTUSDT": object()}

    # runtime.py imports LiveTrader at module top, so isinstance() in the
    # watchdog checks against backend.runtime.LiveTrader specifically.
    from backend import runtime as runtime_mod

    fake_live = _RecordingLiveTrader()
    monkeypatch.setattr(runtime_mod, "LiveTrader", _RecordingLiveTrader)
    rt._trader = fake_live

    async def _runner():
        task = asyncio.create_task(rt._ws_watchdog())
        await asyncio.sleep(0.7)
        rt._stopped = True
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    await _runner()

    assert fake_live.emergency_flatten_calls >= 1, "watchdog must call emergency_flatten on live trader"
    assert fake_live.cancel_calls == [], "live path must NOT fall through to plain cancel_all"
    assert state.guards.emergency_stopped is True


@pytest.mark.asyncio
async def test_ws_watchdog_paper_trader_only_cancels_orders(monkeypatch):
    state = AppState()
    state.mode = Mode.LIVE
    state.binance_last_msg_ts = time.time() - 60.0
    state.guards.ws_stale_ms = 1500.0

    rt = Runtime.__new__(Runtime)
    rt.state = state
    rt._stopped = False
    rt.symbols = {"UBUSDT": object(), "AIOTUSDT": object()}

    from backend import runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "LiveTrader", _RecordingLiveTrader)
    paper = _RecordingPaperTrader()
    rt._trader = paper

    async def _runner():
        task = asyncio.create_task(rt._ws_watchdog())
        await asyncio.sleep(0.7)
        rt._stopped = True
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    await _runner()

    assert sorted(set(paper.cancel_calls)) == ["AIOTUSDT", "UBUSDT"]
