import time
from types import SimpleNamespace

import pytest

from backend.settings_store import RuntimeSettings
from backend.state import AppState, SymbolStats
from backend.traders.live import LiveTrader


class FakePredictor:
    enabled = True

    def __init__(self, confidence=0.6):
        self.confidence = confidence

    def predict(self, symbol, snap):
        return {
            "5s": SimpleNamespace(
                confidence=self.confidence,
                p_up=0.70 if self.confidence > 0 else 0.10,
                p_flat=0.20,
                p_down=0.10 if self.confidence > 0 else 0.70,
            )
        }

    def latest_derived(self, symbol):
        return {"vol_w120_bp": 30.0}


class FakeRest:
    def __init__(self):
        self.leverage_calls = []
        self.orders = []
        self.cancelled = []
        self.positions = {}
        self.flattened = []

    async def set_leverage(self, symbol, leverage):
        self.leverage_calls.append((symbol, leverage))
        return {"symbol": symbol, "leverage": leverage}

    async def cached_symbol_filters(self, symbol):
        # Use loose filters so tests focus on the logic, not exchange limits.
        from backend.exchanges.binance_filters import SymbolFilters

        return SymbolFilters(
            tick_size=1e-9,
            step_size=1e-9,
            min_qty=0.0,
            min_notional=0.0,
            price_precision=9,
            quantity_precision=9,
        )

    async def place_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"orderId": len(self.orders), "clientOrderId": kwargs.get("client_order_id")}

    async def cancel_all(self, symbol):
        self.cancelled.append(symbol)
        return {"symbol": symbol}

    async def get_open_position_amt(self, symbol):
        return self.positions.get(symbol, 0.0)

    async def market_close_position(self, symbol, position_amt):
        self.flattened.append((symbol, position_amt))
        return {"symbol": symbol, "positionAmt": position_amt}


def _state_with_symbol(*, live=True):
    state = AppState()
    state.mode = "live"
    state.binance_connected = True
    state.binance_last_msg_ts = time.time()
    state.binance_private_connected = True
    state.binance_private_last_msg_ts = time.time()
    stats = SymbolStats(symbol="UBUSDT", position_size_usd=6.0, started_at=time.time(), added_at=time.time())
    if live:
        stats.execution_mode = "live"
        stats.live_state = "probation_live"
        stats.current_notional_usd = 6.0
    state.symbols["UBUSDT"] = stats
    return state


def _snap():
    return {
        "ts_ms": int(time.time() * 1000),
        "symbol": "UBUSDT",
        "best_bid": 0.999,
        "best_ask": 1.001,
        "mid": 1.0,
    }


@pytest.mark.asyncio
async def test_live_trader_rejects_when_private_ws_stale():
    state = _state_with_symbol(live=True)
    state.binance_private_connected = False
    rest = FakeRest()
    trader = LiveTrader(
        state,
        predictor=FakePredictor(),
        rest_client=rest,
        runtime_settings=RuntimeSettings(),
    )
    await trader.start()

    await trader.on_snapshot("UBUSDT", _snap())

    assert rest.orders == []
    assert state.symbols["UBUSDT"].rejects_count == 1
    assert "private ws" in state.symbols["UBUSDT"].block_reason


@pytest.mark.asyncio
async def test_live_trader_ignores_paper_symbol():
    state = _state_with_symbol(live=False)
    rest = FakeRest()
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    await trader.start()

    await trader.on_snapshot("UBUSDT", _snap())

    assert rest.orders == []


@pytest.mark.asyncio
async def test_live_trader_sets_leverage_and_places_opus_market_order():
    state = _state_with_symbol(live=True)
    rest = FakeRest()
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    await trader.start()

    await trader.on_snapshot("UBUSDT", _snap())

    assert rest.leverage_calls == [("UBUSDT", 10)]
    assert len(rest.orders) == 1
    order = rest.orders[0]
    assert order["symbol"] == "UBUSDT"
    assert order["side"] == "BUY"
    assert order["order_type"] == "MARKET"
    assert order["quantity"] == pytest.approx(6.0 / 1.001)
    assert order["client_order_id"].startswith("OPUS_UBUSDT_")
    assert state.symbols["UBUSDT"].position_base == pytest.approx(6.0 / 1.001)


@pytest.mark.asyncio
async def test_live_trader_blocks_low_expected_gross_symbol():
    state = _state_with_symbol(live=True)
    rest = FakeRest()
    trader = LiveTrader(
        state,
        predictor=FakePredictor(confidence=0.2),
        rest_client=rest,
        runtime_settings=RuntimeSettings(min_expected_gross_bp=12.0),
    )
    await trader.start()

    await trader.on_snapshot("UBUSDT", _snap())

    assert rest.orders == []
    assert state.symbols["UBUSDT"].expected_gross_bp == pytest.approx(6.0)
    assert "expected gross" in state.symbols["UBUSDT"].block_reason


@pytest.mark.asyncio
async def test_live_trader_closes_open_position_on_horizon_timeout():
    state = _state_with_symbol(live=True)
    rest = FakeRest()
    rest.positions["UBUSDT"] = 6.0
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    await trader.start()

    # Open a position via the normal entry path.
    await trader.on_snapshot("UBUSDT", _snap())
    assert "UBUSDT" in trader._positions
    pos = trader._positions["UBUSDT"]

    # Snapshot well past the horizon -> timeout exit fires.
    snap = _snap()
    snap["ts_ms"] = pos.ts_open_ms + pos.horizon_ms + 1000
    await trader.on_snapshot("UBUSDT", snap)

    assert "UBUSDT" not in trader._positions
    assert rest.flattened, "expected reduce-only close via market_close_position"


@pytest.mark.asyncio
async def test_live_trader_closes_open_position_on_opposing_signal():
    state = _state_with_symbol(live=True)
    rest = FakeRest()
    rest.positions["UBUSDT"] = 6.0
    pred = FakePredictor(confidence=0.6)
    trader = LiveTrader(state, predictor=pred, rest_client=rest, runtime_settings=RuntimeSettings())
    await trader.start()

    await trader.on_snapshot("UBUSDT", _snap())
    assert "UBUSDT" in trader._positions

    # Predictor flips to strong opposing signal.
    pred.confidence = -0.6
    snap = _snap()
    snap["ts_ms"] = trader._positions["UBUSDT"].ts_open_ms + 100
    await trader.on_snapshot("UBUSDT", snap)

    assert "UBUSDT" not in trader._positions
    assert rest.flattened


@pytest.mark.asyncio
async def test_live_trader_records_pnl_and_realized_exactly_once_even_if_flatten_fails():
    """Even if the Binance REST flatten errors, PnL must record exactly once."""

    state = _state_with_symbol(live=True)
    rest = FakeRest()
    # Make market_close_position blow up to simulate a Binance / network error.
    rest.positions["UBUSDT"] = 6.0

    async def boom(symbol, position_amt):
        raise RuntimeError("simulated REST failure")

    rest.market_close_position = boom

    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    await trader.start()

    await trader.on_snapshot("UBUSDT", _snap())
    assert "UBUSDT" in trader._positions
    pos = trader._positions["UBUSDT"]

    # First close attempt — flatten will raise but be swallowed.
    snap = _snap()
    snap["ts_ms"] = pos.ts_open_ms + pos.horizon_ms + 1000
    await trader.on_snapshot("UBUSDT", snap)

    pnl_after_first = state.symbols["UBUSDT"].realized_pnl
    daily_after_first = state.guards.daily_pnl
    fills_after_first = state.symbols["UBUSDT"].fills_count
    assert "UBUSDT" not in trader._positions, "position must be popped before flatten"

    # Subsequent snapshots must NOT re-record PnL since the position is gone.
    snap2 = dict(snap)
    snap2["ts_ms"] += 1000
    await trader.on_snapshot("UBUSDT", snap2)

    assert state.symbols["UBUSDT"].realized_pnl == pnl_after_first
    assert state.guards.daily_pnl == daily_after_first
    assert state.symbols["UBUSDT"].fills_count == fills_after_first


@pytest.mark.asyncio
async def test_live_trader_stop_cancels_and_flattens_managed_symbols():
    state = _state_with_symbol(live=True)
    rest = FakeRest()
    rest.positions["UBUSDT"] = 3.0
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    await trader.start()

    await trader.stop()

    assert rest.cancelled == ["UBUSDT"]
    assert rest.flattened == [("UBUSDT", 3.0)]
