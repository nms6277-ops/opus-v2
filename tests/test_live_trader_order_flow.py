import asyncio
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

    async def position_risk(self, symbol=None):
        # Mirror the live REST shape but with no pre-existing positions.
        target = (symbol or "").upper()
        return [{"symbol": target, "positionAmt": "0"}]


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
    # Entry MARKET + reduce-only STOP_MARKET (default trade_stop_loss_bp=50).
    market_orders = [o for o in rest.orders if o["order_type"] == "MARKET"]
    stop_orders = [o for o in rest.orders if o["order_type"] == "STOP_MARKET"]
    assert len(market_orders) == 1, rest.orders
    assert len(stop_orders) == 1, rest.orders
    entry = market_orders[0]
    assert entry["symbol"] == "UBUSDT"
    assert entry["side"] == "BUY"
    assert entry["quantity"] == pytest.approx(6.0 / 1.001)
    assert entry["client_order_id"].startswith("OPUS_UBUSDT_")
    stop = stop_orders[0]
    assert stop["side"] == "SELL"  # close-side
    assert stop["reduce_only"] is True
    assert stop["working_type"] == "MARK_PRICE"
    assert stop["client_order_id"].startswith("OPUS_SL_UBUSDT_")
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
async def test_emergency_flatten_includes_positions_after_live_to_paper_switch():
    """A position must still be flattened even if the symbol was switched to paper."""

    state = _state_with_symbol(live=True)
    rest = FakeRest()
    rest.positions["UBUSDT"] = 6.0
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    await trader.start()

    # Open a real position, then drop the symbol to paper without flattening
    # (simulates a stale process state where flatten failed earlier).
    await trader.on_snapshot("UBUSDT", _snap())
    assert "UBUSDT" in trader._positions
    state.symbols["UBUSDT"].execution_mode = "paper"
    state.symbols["UBUSDT"].live_state = "paper"

    # _managed_symbols would now return [] — but emergency_flatten must
    # still cover the open position via _positions.
    assert "UBUSDT" not in trader._managed_symbols()
    await trader.emergency_flatten()
    assert any(sym == "UBUSDT" for sym, _ in rest.flattened)
    assert "UBUSDT" not in trader._positions


@pytest.mark.asyncio
async def test_live_trader_runs_exit_checks_even_after_symbol_disabled():
    """Stranded positions must still get stop-loss/timeout exits if mode flips."""

    state = _state_with_symbol(live=True)
    rest = FakeRest()
    rest.positions["UBUSDT"] = 6.0
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    await trader.start()

    # Open a real position, then have the operator disable the symbol — but
    # simulate the path where the position remains in trader._positions
    # (e.g. mode flipped externally without flatten_symbol).
    await trader.on_snapshot("UBUSDT", _snap())
    assert "UBUSDT" in trader._positions
    pos = trader._positions["UBUSDT"]
    state.symbols["UBUSDT"].execution_mode = "paper"
    state.symbols["UBUSDT"].live_state = "disabled"

    snap = _snap()
    snap["ts_ms"] = pos.ts_open_ms + pos.horizon_ms + 1000
    await trader.on_snapshot("UBUSDT", snap)

    assert "UBUSDT" not in trader._positions, "exit checks must run even when symbol no longer live"
    assert rest.flattened


@pytest.mark.asyncio
async def test_live_trader_close_does_not_locally_record_pnl():
    """``_maybe_close`` only sends the flatten; PnL accounting is owned by
    ``on_order_update`` using Binance's actual ``rp`` field. Recording an
    estimated PnL locally in addition would double-count when the WS fill
    arrives back from the reduce-only MARKET ``_flatten`` placed.
    """

    state = _state_with_symbol(live=True)
    rest = FakeRest()
    rest.positions["UBUSDT"] = 6.0
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    await trader.start()

    await trader.on_snapshot("UBUSDT", _snap())
    assert "UBUSDT" in trader._positions
    pos = trader._positions["UBUSDT"]

    # Trigger horizon-timeout exit. ``_maybe_close`` must NOT record PnL —
    # the flatten REST is sent and the actual realised PnL will arrive
    # later via the private-WS ORDER_TRADE_UPDATE event.
    snap = _snap()
    snap["ts_ms"] = pos.ts_open_ms + pos.horizon_ms + 1000
    await trader.on_snapshot("UBUSDT", snap)

    assert "UBUSDT" not in trader._positions, "position must be popped after exit decision"
    assert rest.flattened, "flatten REST must have been called"
    # Crucially: no local PnL accounting happened in _maybe_close.
    assert state.symbols["UBUSDT"].realized_pnl == 0.0
    assert state.guards.daily_pnl == 0.0
    assert state.symbols["UBUSDT"].fills_count == 0
    # Subsequent snapshots must not re-fire any accounting either.
    snap2 = dict(snap)
    snap2["ts_ms"] += 1000
    await trader.on_snapshot("UBUSDT", snap2)
    assert state.symbols["UBUSDT"].realized_pnl == 0.0
    assert state.guards.daily_pnl == 0.0


@pytest.mark.asyncio
async def test_live_trader_close_records_pnl_when_ws_fill_arrives():
    """Full close cycle: _maybe_close → flatten REST → ORDER_TRADE_UPDATE
    arrives via on_order_update → PnL recorded EXACTLY ONCE from Binance's
    ``rp`` field (no double-counting with _maybe_close estimate).
    """

    state = _state_with_symbol(live=True)
    rest = FakeRest()
    rest.positions["UBUSDT"] = 6.0
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    await trader.start()
    await trader.on_snapshot("UBUSDT", _snap())
    pos = trader._positions["UBUSDT"]

    # Horizon timeout fires the flatten.
    snap = _snap()
    snap["ts_ms"] = pos.ts_open_ms + pos.horizon_ms + 1000
    await trader.on_snapshot("UBUSDT", snap)

    # Simulate the Binance ORDER_TRADE_UPDATE that arrives AFTER the
    # reduce-only MARKET _flatten placed.
    fill = {
        "e": "ORDER_TRADE_UPDATE",
        "o": {
            "s": "UBUSDT",
            "S": "SELL",
            "X": "FILLED",
            "c": "OPUS_FLATTEN_UBUSDT_99",
            "i": 5555,
            "t": 7777,
            "z": "5.994005994",
            "ap": "1.0005",
            "rp": "-0.030",
            "R": True,
        },
    }
    trader.on_order_update(fill)
    # Re-deliver same trade-id (Binance reconnect replay) — must dedupe.
    trader.on_order_update(fill)

    assert state.symbols["UBUSDT"].realized_pnl == pytest.approx(-0.030)
    assert state.guards.daily_pnl == pytest.approx(-0.030)
    assert state.symbols["UBUSDT"].fills_count == 1


@pytest.mark.asyncio
async def test_seen_private_trade_ids_evicts_oldest_above_cap(monkeypatch):
    """``_seen_private_trade_ids`` must stay bounded under 24/7 operation."""
    from backend.traders import live as live_mod

    monkeypatch.setattr(live_mod, "_MAX_SEEN_TRADE_IDS", 5)

    state = _state_with_symbol(live=True)
    rest = FakeRest()
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )

    for tid in range(20):
        trader.on_order_update(
            {
                "e": "ORDER_TRADE_UPDATE",
                "o": {
                    "s": "UBUSDT",
                    "S": "BUY",
                    "X": "FILLED",
                    "c": f"OPUS_test_{tid}",
                    "i": tid,
                    "t": tid,
                    "z": "1.0",
                    "ap": "1.0",
                    "rp": "0.0",
                    "R": False,
                },
            }
        )

    # Set must be capped to the configured max.
    assert len(trader._seen_private_trade_ids) <= 5
    assert len(trader._seen_private_trade_order) <= 5
    # Oldest IDs evicted; newest retained.
    assert ("UBUSDT", "0") not in trader._seen_private_trade_ids
    assert ("UBUSDT", "19") in trader._seen_private_trade_ids


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


# ---------------------------------------------------------------------------
# Tests for v3-ported live-trading features:
#   - reconcile_open_positions (fail-closed startup)
#   - on_order_update (real fill tracking via private WS)
#   - on_account_update (position sync)
#   - _arm_protective_orders (STOP/TP on Binance)
#   - stats_dict (UI panel data)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_open_positions_disables_symbol_with_existing_position():
    """If Binance already has a position on startup, the symbol must be disabled."""
    state = _state_with_symbol(live=True)
    rest = FakeRest()

    async def position_risk(symbol=None):  # pre-existing position!
        return [{"symbol": (symbol or "").upper(), "positionAmt": "3.5"}]

    rest.position_risk = position_risk
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    await trader.start()

    stats = state.symbols["UBUSDT"]
    assert stats.live_state == "disabled"
    assert "pre-existing Binance position" in (stats.block_reason or "")


@pytest.mark.asyncio
async def test_reconcile_open_positions_disables_on_rest_error():
    """REST failure during reconciliation must fail closed (disable symbol)."""
    state = _state_with_symbol(live=True)
    rest = FakeRest()

    async def position_risk(symbol=None):
        raise RuntimeError("Binance 502")

    rest.position_risk = position_risk
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    await trader.start()

    stats = state.symbols["UBUSDT"]
    assert stats.live_state == "disabled"
    assert "position reconciliation failed" in (stats.block_reason or "")


@pytest.mark.asyncio
async def test_arm_protective_orders_places_stop_when_take_profit_disabled():
    """Default config has trade_take_profit_bp=0 — only STOP_MARKET is armed."""
    state = _state_with_symbol(live=True)
    rest = FakeRest()
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    await trader.start()

    await trader.on_snapshot("UBUSDT", _snap())

    stops = [o for o in rest.orders if o["order_type"] == "STOP_MARKET"]
    tps = [o for o in rest.orders if o["order_type"] == "TAKE_PROFIT_MARKET"]
    assert len(stops) == 1
    assert len(tps) == 0
    stop = stops[0]
    assert stop["reduce_only"] is True
    assert stop["working_type"] == "MARK_PRICE"
    # Long entry @ 1.001, stop_loss_bp=50 → stop ≈ 1.001 * (1 - 0.005) = 0.99600
    assert stop["stop_price"] == pytest.approx(1.001 * (1 - 50.0 / 10_000.0), rel=1e-6)


@pytest.mark.asyncio
async def test_arm_protective_orders_places_take_profit_when_enabled(monkeypatch):
    """trade_take_profit_bp > 0 also arms TAKE_PROFIT_MARKET."""
    from backend.config import settings as cfg

    monkeypatch.setattr(cfg, "trade_take_profit_bp", 80.0)
    state = _state_with_symbol(live=True)
    rest = FakeRest()
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    await trader.start()

    await trader.on_snapshot("UBUSDT", _snap())

    tps = [o for o in rest.orders if o["order_type"] == "TAKE_PROFIT_MARKET"]
    assert len(tps) == 1
    tp = tps[0]
    assert tp["reduce_only"] is True
    assert tp["side"] == "SELL"
    # Long entry @ 1.001, take_profit_bp=80 → tp ≈ 1.001 * (1 + 0.008)
    assert tp["stop_price"] == pytest.approx(1.001 * (1 + 80.0 / 10_000.0), rel=1e-6)
    assert tp["client_order_id"].startswith("OPUS_TP_UBUSDT_")


@pytest.mark.asyncio
async def test_arm_protective_orders_failure_does_not_rollback_entry(caplog):
    """If STOP/TP placement fails the entry stays — bot still monitors via snapshot loop."""
    state = _state_with_symbol(live=True)
    rest = FakeRest()
    real_place = rest.place_order

    async def place_order(**kwargs):
        if kwargs.get("order_type") == "STOP_MARKET":
            raise RuntimeError("Binance -2021 stop would trigger immediately")
        return await real_place(**kwargs)

    rest.place_order = place_order
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    await trader.start()
    await trader.on_snapshot("UBUSDT", _snap())

    # Entry MARKET succeeded and local position exists despite STOP failure.
    market_orders = [o for o in rest.orders if o["order_type"] == "MARKET"]
    assert len(market_orders) == 1
    assert "UBUSDT" in trader._positions


@pytest.mark.asyncio
async def test_on_order_update_records_realized_pnl_exactly_once_with_dedupe():
    state = _state_with_symbol(live=True)
    rest = FakeRest()
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    # No start() — we're testing the handler in isolation, so reconcile is bypassed.

    fill_event = {
        "e": "ORDER_TRADE_UPDATE",
        "o": {
            "s": "UBUSDT",
            "S": "SELL",
            "X": "FILLED",
            "c": "OPUS_FLATTEN_UBUSDT_1",
            "i": 12345,
            "t": 999,
            "z": "5.0",
            "ap": "1.005",
            "rp": "0.42",
            "R": True,
        },
    }
    trader.on_order_update(fill_event)
    # Re-deliver same trade-id (Binance reconnect replay) — must NOT double-count.
    trader.on_order_update(fill_event)

    stats = state.symbols["UBUSDT"]
    assert stats.realized_pnl == pytest.approx(0.42)
    assert state.guards.daily_pnl == pytest.approx(0.42)
    assert stats.fills_count == 1


@pytest.mark.asyncio
async def test_on_account_update_syncs_position_fields_and_clears_when_flat():
    state = _state_with_symbol(live=True)
    rest = FakeRest()
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    trader.on_account_update(
        {
            "e": "ACCOUNT_UPDATE",
            "a": {
                "P": [
                    {"s": "UBUSDT", "pa": "5.0", "ep": "1.0", "up": "0.10"},
                ]
            },
        }
    )
    stats = state.symbols["UBUSDT"]
    assert stats.position_base == pytest.approx(5.0)
    assert stats.position_entry == pytest.approx(1.0)
    assert stats.unrealized_pnl == pytest.approx(0.10)

    # Position closed externally → bot must clean up its own tracking.
    trader._positions["UBUSDT"] = SimpleNamespace()  # type: ignore[assignment]
    trader.on_account_update(
        {
            "e": "ACCOUNT_UPDATE",
            "a": {"P": [{"s": "UBUSDT", "pa": "0", "ep": "0", "up": "0"}]},
        }
    )
    assert state.symbols["UBUSDT"].position_base == 0.0
    assert state.symbols["UBUSDT"].position_entry == 0.0
    assert "UBUSDT" not in trader._positions


@pytest.mark.asyncio
async def test_stats_dict_returns_open_positions_and_daily_aggregates():
    state = _state_with_symbol(live=True)
    state.symbols["UBUSDT"].live_trade_count = 3
    state.symbols["UBUSDT"].live_wins = 2
    state.symbols["UBUSDT"].live_losses = 1
    rest = FakeRest()
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    await trader.start()
    await trader.on_snapshot("UBUSDT", _snap())
    state.guards.daily_pnl = 1.23  # set AFTER start (which clears via reset_if_new_day)

    out = trader.stats_dict()
    assert out["daily_trades"] == 3
    assert out["daily_wins"] == 2
    assert out["daily_losses"] == 1
    assert out["daily_win_rate"] == pytest.approx(2 / 3)
    assert out["daily_pnl_usd"] == pytest.approx(1.23)
    assert "UBUSDT" in out["allowed_symbols"]
    assert len(out["open_positions"]) == 1
    pos = out["open_positions"][0]
    assert pos["symbol"] == "UBUSDT"
    assert pos["side"] == "long"


@pytest.mark.asyncio
async def test_break_even_fill_still_records_trade_outcome():
    """A reduce-only FILLED with rp=0.0 must still feed the regime guards."""
    state = _state_with_symbol(live=True)
    rest = FakeRest()
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    initial = state.symbols["UBUSDT"].live_trade_count

    trader.on_order_update(
        {
            "e": "ORDER_TRADE_UPDATE",
            "o": {
                "s": "UBUSDT",
                "S": "SELL",
                "X": "FILLED",
                "c": "OPUS_FLATTEN_UBUSDT_77",
                "i": 4242,
                "t": 4242,
                "z": "5.0",
                "ap": "1.001",
                "rp": "0.0",
                "R": True,
            },
        }
    )

    # Probation / loss-streak counters must reflect the round-trip even at
    # break-even. Without this the trade is invisible to all regime guards.
    assert state.symbols["UBUSDT"].live_trade_count == initial + 1


@pytest.mark.asyncio
async def test_maybe_close_cancels_protective_orders_before_flatten():
    """``_maybe_close`` must cancel the resting STOP/TP before sending the
    flatten MARKET; otherwise the unrelated protective survives on Binance
    and can trigger against the next position."""
    state = _state_with_symbol(live=True)
    rest = FakeRest()
    rest.positions["UBUSDT"] = 6.0
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    await trader.start()
    await trader.on_snapshot("UBUSDT", _snap())
    pos = trader._positions["UBUSDT"]

    snap = _snap()
    snap["ts_ms"] = pos.ts_open_ms + pos.horizon_ms + 1000
    await trader.on_snapshot("UBUSDT", snap)

    # cancel_all must have been called for the symbol, BEFORE the flatten.
    assert rest.cancelled, "cancel_all(symbol) must run during _maybe_close"
    assert "UBUSDT" in [s.upper() for s in rest.cancelled]
    assert rest.flattened, "flatten must still execute"


@pytest.mark.asyncio
async def test_on_order_update_cancels_protective_orders_after_exchange_close():
    """When STOP_MARKET fires on the exchange, the resting TAKE_PROFIT_MARKET
    must be cancelled so it does not trigger against the next entry."""
    state = _state_with_symbol(live=True)
    rest = FakeRest()
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )

    trader.on_order_update(
        {
            "e": "ORDER_TRADE_UPDATE",
            "o": {
                "s": "UBUSDT",
                "S": "SELL",
                "X": "FILLED",
                "c": "OPUS_SL_UBUSDT_999",
                "i": 8888,
                "t": 8888,
                "z": "5.0",
                "ap": "0.996",
                "rp": "-0.025",
                "R": True,
            },
        }
    )
    # Allow the create_task'd cancel_all to run on the test loop.
    await asyncio.sleep(0)
    assert "UBUSDT" in [s.upper() for s in rest.cancelled]


@pytest.mark.asyncio
async def test_on_order_update_does_not_count_new_or_cancelled_as_fills():
    """``NEW`` / ``CANCELLED`` events for resting STOP/TP must not bump
    ``fills_count`` \u2014 only actual trade executions count."""
    state = _state_with_symbol(live=True)
    rest = FakeRest()
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    initial_fills = state.symbols["UBUSDT"].fills_count

    for status, exec_type in [("NEW", "NEW"), ("CANCELLED", "CANCELED"), ("EXPIRED", "EXPIRED")]:
        trader.on_order_update(
            {
                "e": "ORDER_TRADE_UPDATE",
                "o": {
                    "s": "UBUSDT",
                    "S": "SELL",
                    "X": status,
                    "x": exec_type,
                    "c": f"OPUS_SL_UBUSDT_{status}",
                    "i": 1000 + hash(status) % 1000,
                    "t": "0",
                    "z": "0",
                    "ap": "0",
                    "rp": "0",
                    "R": True,
                },
            }
        )

    assert state.symbols["UBUSDT"].fills_count == initial_fills, (
        "non-trade ORDER_TRADE_UPDATE events must not bump fills_count"
    )

    # Sanity: a real TRADE event still counts.
    trader.on_order_update(
        {
            "e": "ORDER_TRADE_UPDATE",
            "o": {
                "s": "UBUSDT",
                "S": "SELL",
                "X": "PARTIALLY_FILLED",
                "x": "TRADE",
                "c": "OPUS_FLATTEN_UBUSDT_42",
                "i": 4242,
                "t": 4242,
                "z": "1.0",
                "ap": "1.0",
                "rp": "0.0",
                "R": True,
            },
        }
    )
    assert state.symbols["UBUSDT"].fills_count == initial_fills + 1


@pytest.mark.asyncio
async def test_on_snapshot_warms_predictor_when_private_ws_down():
    """Even when private WS is stale and entries are blocked, the predictor's
    history buffer must keep receiving snapshots so that lag / rolling
    features stay continuous when the WS recovers."""

    class CountingPredictor:
        enabled = True

        def __init__(self):
            self.update_calls = 0
            self.predict_calls = 0

        def update(self, symbol, snap):
            self.update_calls += 1

        def predict(self, symbol, snap):
            self.predict_calls += 1
            self.update(symbol, snap)
            return None

    state = _state_with_symbol(live=True)
    state.binance_private_connected = False  # private WS down
    rest = FakeRest()
    pred = CountingPredictor()
    trader = LiveTrader(state, predictor=pred, rest_client=rest, runtime_settings=RuntimeSettings())
    await trader.start()

    await trader.on_snapshot("UBUSDT", _snap())
    await trader.on_snapshot("UBUSDT", _snap())

    # Both snapshots must have been pushed exactly once (no double-push).
    assert pred.update_calls == 2
    # And no entry was attempted.
    assert rest.orders == []


@pytest.mark.asyncio
async def test_on_snapshot_warms_predictor_when_symbol_paper():
    """Symbol switched to paper while LIVE mode is active: predictor buffer
    must keep warming so it is ready when the symbol is re-enabled."""

    class CountingPredictor:
        enabled = True

        def __init__(self):
            self.update_calls = 0

        def update(self, symbol, snap):
            self.update_calls += 1

        def predict(self, symbol, snap):
            self.update(symbol, snap)
            return None

    state = _state_with_symbol(live=False)  # execution_mode == "paper"
    rest = FakeRest()
    pred = CountingPredictor()
    trader = LiveTrader(state, predictor=pred, rest_client=rest, runtime_settings=RuntimeSettings())
    await trader.start()

    await trader.on_snapshot("UBUSDT", _snap())
    assert pred.update_calls == 1
    assert rest.orders == []


@pytest.mark.asyncio
async def test_live_trader_reapplies_leverage_after_runtime_setting_change():
    """Operator lowers leverage in the UI; the LiveTrader must push the new
    value to Binance instead of silently short-circuiting because the symbol
    is already in the cache."""

    state = _state_with_symbol(live=True)
    rest = FakeRest()
    settings = RuntimeSettings(leverage=10)
    trader = LiveTrader(state, predictor=FakePredictor(), rest_client=rest, runtime_settings=settings)
    await trader.start()

    await trader.on_snapshot("UBUSDT", _snap())
    assert rest.leverage_calls == [("UBUSDT", 10.0)]

    # Close the open position so the next snapshot takes the entry path again.
    trader._positions.pop("UBUSDT", None)
    state.symbols["UBUSDT"].position_base = 0.0

    # Operator reduces leverage via the UI.
    trader.runtime_settings = RuntimeSettings(leverage=3)
    await trader.on_snapshot("UBUSDT", _snap())

    assert rest.leverage_calls[-1] == ("UBUSDT", 3.0), rest.leverage_calls

    # Same leverage on a second snapshot must NOT resubmit.
    trader._positions.pop("UBUSDT", None)
    state.symbols["UBUSDT"].position_base = 0.0
    calls_before = len(rest.leverage_calls)
    await trader.on_snapshot("UBUSDT", _snap())
    assert len(rest.leverage_calls) == calls_before


@pytest.mark.asyncio
async def test_close_order_pnl_cleared_on_canceled_reduce_only():
    """Partial-filled reduce-only order that is later CANCELED must not
    leak a ``_close_order_pnl`` entry — otherwise 24/7 operation drifts
    unbounded against the 2 GB VPS memory budget."""

    state = _state_with_symbol(live=True)
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=FakeRest(), runtime_settings=RuntimeSettings()
    )

    # Partial fill on a reduce-only STOP order.
    trader.on_order_update(
        {
            "o": {
                "s": "UBUSDT",
                "S": "SELL",
                "X": "PARTIALLY_FILLED",
                "x": "TRADE",
                "c": "OPUS_SL_UBUSDT_1",
                "i": 111,
                "t": "T1",
                "R": True,
                "z": 1.0,
                "ap": 1.0,
                "rp": -0.01,
            }
        }
    )
    assert ("UBUSDT", "OPUS_SL_UBUSDT_1") in trader._close_order_pnl

    # Order then gets CANCELED before it finishes.
    trader.on_order_update(
        {
            "o": {
                "s": "UBUSDT",
                "S": "SELL",
                "X": "CANCELED",
                "x": "CANCELED",
                "c": "OPUS_SL_UBUSDT_1",
                "i": 111,
                "t": "T2",
                "R": True,
                "z": 1.0,
                "ap": 1.0,
                "rp": 0.0,
            }
        }
    )
    assert ("UBUSDT", "OPUS_SL_UBUSDT_1") not in trader._close_order_pnl


@pytest.mark.asyncio
async def test_set_leverage_failure_rejects_symbol_not_crash():
    """A REST failure on set_leverage must reject the symbol cleanly with a
    block_reason / rejects_count update, NOT propagate up to the snapshot
    loop where it surfaces as an opaque generic error."""

    state = _state_with_symbol(live=True)

    class _BoomRest(FakeRest):
        async def set_leverage(self, symbol, leverage):
            raise RuntimeError("Binance POST /fapi/v1/leverage failed 418: rate limited")

    rest = _BoomRest()
    trader = LiveTrader(
        state, predictor=FakePredictor(), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    await trader.start()

    await trader.on_snapshot("UBUSDT", _snap())

    # No order should have been placed.
    assert rest.orders == []
    # Symbol should have a clear, actionable block_reason in the UI.
    assert state.symbols["UBUSDT"].rejects_count == 1
    assert "set leverage failed" in state.symbols["UBUSDT"].block_reason
    assert "rate limited" in state.symbols["UBUSDT"].block_reason


@pytest.mark.asyncio
async def test_entry_uses_instance_conf_threshold_not_module_settings():
    """Entry gate must read self._conf_thr (snapshot of settings at __init__)
    instead of the module-level ``settings.trade_conf_threshold``. Otherwise
    a hot-reload / monkeypatch of settings would only take effect in the
    exit gate, splitting entry vs exit thresholds."""

    state = _state_with_symbol(live=True)
    rest = FakeRest()
    trader = LiveTrader(
        state, predictor=FakePredictor(confidence=0.05), rest_client=rest, runtime_settings=RuntimeSettings()
    )
    # Force the instance threshold high enough that the prediction (0.05)
    # cannot pass. If the entry gate were still reading module-level
    # settings, this monkey-patch would be invisible and an order would
    # still go through.
    trader._conf_thr = 0.5
    await trader.start()

    await trader.on_snapshot("UBUSDT", _snap())

    assert rest.orders == [], "instance threshold must gate the entry decision"
