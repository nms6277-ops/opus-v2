import pytest

from backend.config import settings
from backend.runtime import Runtime, SymbolCtx
from backend.state import AppState, SymbolStats


class FakeBook:
    pass


class FakeWriter:
    async def close(self):
        return None


class FakeWS:
    def __init__(self):
        self.started = []
        self.stopped = False
        self.added = []
        self.removed = []

    async def start(self, symbols):
        self.started.append(list(symbols))

    async def stop(self):
        self.stopped = True

    async def add_symbols(self, symbols):
        self.added.append(list(symbols))

    async def remove_symbols(self, symbols):
        self.removed.append(list(symbols))


@pytest.mark.asyncio
async def test_maybe_start_bybit_skips_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_bybit", False)
    rt = Runtime(AppState())
    rt.bybit_ws = FakeWS()

    await rt._maybe_start_bybit()

    assert rt.bybit_ws.started == []


@pytest.mark.asyncio
async def test_maybe_add_remove_bybit_symbols_skips_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_bybit", False)
    rt = Runtime(AppState())
    rt.bybit_ws = FakeWS()

    await rt._maybe_add_bybit_symbols(["UBUSDT"])
    await rt._maybe_remove_bybit_symbols(["UBUSDT"])

    assert rt.bybit_ws.added == []
    assert rt.bybit_ws.removed == []


@pytest.mark.asyncio
async def test_remove_symbol_does_not_unsubscribe_bybit_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_bybit", False)
    rt = Runtime(AppState())
    rt.binance_ws = FakeWS()
    rt.bybit_ws = FakeWS()
    rt.symbols["UBUSDT"] = SymbolCtx(symbol="UBUSDT", ob=FakeBook(), writer=FakeWriter())
    rt.state.symbols["UBUSDT"] = SymbolStats(
        symbol="UBUSDT",
        position_size_usd=6.0,
        started_at=1.0,
        added_at=1.0,
    )

    await rt.remove_symbol("UBUSDT")

    assert rt.binance_ws.removed == [["UBUSDT"]]
    assert rt.bybit_ws.removed == []
