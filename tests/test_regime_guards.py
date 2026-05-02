from backend.safety.guards import record_pnl
from backend.safety.regime import record_trade_outcome
from backend.settings_store import RuntimeSettings
from backend.state import AppState, SymbolStats


def _settings(**overrides):
    base = {
        "global_profit_giveback_pct": 0.30,
        "symbol_profit_giveback_pct": 0.30,
        "loss_streak_limit": 4,
        "rolling_guard_trades": 15,
        "rolling_min_win_rate": 0.35,
        "rolling_min_loss_net_bp": 50.0,
        "rolling_min_drawdown_pct": 0.10,
        "global_guard_min_trades": 20,
        "symbol_guard_min_trades": 10,
    }
    base.update(overrides)
    return RuntimeSettings(**base)


def _state(symbol="UBUSDT"):
    state = AppState()
    state.symbols[symbol] = SymbolStats(symbol=symbol, position_size_usd=6.0, started_at=1.0, added_at=1.0)
    return state


def _close(state, symbol, pnl_usd, net_bp, settings):
    stats = state.symbols[symbol]
    stats.realized_pnl += pnl_usd
    record_pnl(state.guards, pnl_usd, symbol=symbol, ts=1.0 + stats.live_trade_count)
    return record_trade_outcome(state, symbol=symbol, pnl_usd=pnl_usd, net_bp=net_bp, settings=settings)


def test_global_profit_giveback_trips_emergency_after_min_sample():
    state = _state("UBUSDT")
    settings = _settings()

    for _ in range(20):
        assert _close(state, "UBUSDT", 0.10, 10.0, settings) is None
    event = _close(state, "UBUSDT", -0.61, -61.0, settings)

    assert event is not None
    assert event.scope == "global"
    assert "global profit giveback" in event.reason
    assert state.guards.emergency_stopped is True
    assert state.guards.pnl_drawdown_pct > 0.30


def test_symbol_profit_giveback_pauses_symbol_for_operator_decision():
    state = _state("UBUSDT")
    state.symbols["UBUSDT"].execution_mode = "live"
    state.symbols["UBUSDT"].live_state = "active_live"
    settings = _settings(global_guard_min_trades=999)

    for _ in range(10):
        assert _close(state, "UBUSDT", 0.01, 10.0, settings) is None
    event = _close(state, "UBUSDT", -0.031, -31.0, settings)

    stats = state.symbols["UBUSDT"]
    assert event is not None
    assert event.scope == "symbol"
    assert event.symbol == "UBUSDT"
    assert "symbol profit giveback" in event.reason
    assert stats.live_state == "awaiting_operator"
    assert "drawdown" in stats.block_reason


def test_four_trade_loss_streak_pauses_symbol():
    state = _state("UBUSDT")
    settings = _settings(global_guard_min_trades=999, symbol_guard_min_trades=999)

    events = [_close(state, "UBUSDT", -0.001, -1.0, settings) for _ in range(4)]

    assert events[-1] is not None
    assert "loss streak" in events[-1].reason
    assert state.symbols["UBUSDT"].live_state == "awaiting_operator"
    assert state.symbols["UBUSDT"].consecutive_losses == 4


def test_rolling_degradation_ignores_shallow_negative_noise():
    state = _state("UBUSDT")
    settings = _settings(global_guard_min_trades=999, symbol_guard_min_trades=999)
    for _ in range(10):
        assert _close(state, "UBUSDT", 0.10, 10.0, settings) is None
    pattern = [1, -2, -2, -2, 1, -2, -2, -2, 1, -2, -2, -2, 1, -2, -2]

    event = None
    for net_bp in pattern:
        event = _close(state, "UBUSDT", net_bp / 1000.0, float(net_bp), settings)

    stats = state.symbols["UBUSDT"]
    assert event is None
    assert stats.rolling_win_rate < 0.35
    assert stats.rolling_sum_net_bp < 0.0
    assert stats.rolling_sum_net_bp > -50.0
    assert stats.live_state != "awaiting_operator"


def test_rolling_degradation_triggers_on_low_winrate_and_large_negative_sum_net():
    state = _state("UBUSDT")
    settings = _settings(global_guard_min_trades=999, symbol_guard_min_trades=999)
    pattern = [1, -6, -6, -6, 1, -6, -6, -6, 1, -6, -6, -6, 1, -6, -6]

    event = None
    for net_bp in pattern:
        event = _close(state, "UBUSDT", net_bp / 1000.0, float(net_bp), settings)

    stats = state.symbols["UBUSDT"]
    assert event is not None
    assert "rolling degradation" in event.reason
    assert stats.rolling_win_rate < 0.35
    assert stats.rolling_sum_net_bp <= -50.0
    assert stats.live_state == "awaiting_operator"


def test_rolling_degradation_triggers_on_low_winrate_and_symbol_drawdown():
    state = _state("UBUSDT")
    settings = _settings(
        global_guard_min_trades=999,
        symbol_guard_min_trades=999,
        rolling_min_loss_net_bp=999.0,
        rolling_min_drawdown_pct=0.10,
    )
    for _ in range(10):
        assert _close(state, "UBUSDT", 0.01, 10.0, settings) is None
    pattern = [1, -2, -2, -2, 1, -2, -2, -2, 1, -2, -2, -2, 1, -2, -2]

    event = None
    for net_bp in pattern:
        event = _close(state, "UBUSDT", net_bp / 1000.0, float(net_bp), settings)

    stats = state.symbols["UBUSDT"]
    assert event is not None
    assert "rolling degradation" in event.reason
    assert stats.rolling_win_rate < 0.35
    assert stats.rolling_sum_net_bp > -50.0
    assert stats.pnl_drawdown_pct >= 0.10
    assert stats.live_state == "awaiting_operator"
