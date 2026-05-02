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
    trader = LiveTrader(state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings())
    await trader.start()

    await trader.on_snapshot("UBUSDT", _snap())

    assert rest.orders == []


@pytest.mark.asyncio
async def test_live_trader_sets_leverage_and_places_opus_market_order():
    state = _state_with_symbol(live=True)
    rest = FakeRest()
    trader = LiveTrader(state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings())
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
async def test_live_trader_stop_cancels_and_flattens_managed_symbols():
    state = _state_with_symbol(live=True)
    rest = FakeRest()
    rest.positions["UBUSDT"] = 3.0
    trader = LiveTrader(state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings())
    await trader.start()

    await trader.stop()

    assert rest.cancelled == ["UBUSDT"]
    assert rest.flattened == [("UBUSDT", 3.0)]
