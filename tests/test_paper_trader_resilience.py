"""Resilience tests for PaperTrader exit logic under predictor failure."""

from types import SimpleNamespace

import pytest

from backend.config import settings
from backend.settings_store import RuntimeSettings
from backend.state import AppState, SymbolStats
from backend.traders.paper import PaperTrader


class _ExplodingPredictor:
    enabled = True

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, symbol, snap):
        self.calls += 1
        raise RuntimeError("model corrupted")


class _ConstantPredictor:
    enabled = True

    def __init__(self, confidence: float) -> None:
        self.confidence = confidence

    def predict(self, symbol, snap):
        return {
            settings.trade_horizon: SimpleNamespace(
                confidence=self.confidence,
                p_up=0.7 if self.confidence > 0 else 0.1,
                p_flat=0.2,
                p_down=0.1 if self.confidence > 0 else 0.7,
            )
        }


def _state_with(symbol: str = "UBUSDT") -> AppState:
    import time as _time

    state = AppState()
    stats = SymbolStats(
        symbol=symbol,
        position_size_usd=6.0,
        started_at=_time.time(),
        added_at=_time.time(),
    )
    stats.best_bid = 0.99
    stats.best_ask = 1.01
    state.symbols[symbol] = stats
    return state


def _snap(ts_ms: int = 1_000_000) -> dict:
    return {
        "ts_ms": ts_ms,
        "best_bid": 0.99,
        "best_ask": 1.01,
        "mid": 1.00,
    }


@pytest.mark.asyncio
async def test_paper_trader_runs_exit_checks_when_predictor_fails():
    """If predictor.predict raises, _maybe_close must still run for open positions."""

    state = _state_with("UBUSDT")
    pred = _ConstantPredictor(confidence=0.9)
    trader = PaperTrader(state, predictor=pred, runtime_settings=RuntimeSettings())
    await trader.start()
    # Allow the trader to take any allowed symbol.
    trader._allowed = set()

    # Open via a working predictor.
    await trader.on_snapshot("UBUSDT", _snap(ts_ms=1_000_000))
    assert "UBUSDT" in trader._positions
    pos = trader._positions["UBUSDT"]

    # Swap in a predictor that explodes on every call.
    boom = _ExplodingPredictor()
    trader.predictor = boom

    # Snapshot well past the horizon — timeout path inside _maybe_close
    # must still fire even though the predictor raised.
    snap = _snap(ts_ms=pos.ts_open_ms + pos.horizon_ms + 5_000)
    await trader.on_snapshot("UBUSDT", snap)

    assert boom.calls == 1, "predictor must still be called once per snapshot"
    assert "UBUSDT" not in trader._positions, (
        "open position must be closed by horizon timeout even when predictor raises"
    )
