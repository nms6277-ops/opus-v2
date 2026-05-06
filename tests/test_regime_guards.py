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


def test_break_even_trades_do_not_count_as_losses_in_streak():
    state = _state("UBUSDT")
    settings = _settings(global_guard_min_trades=999, symbol_guard_min_trades=999)

    # 3 real losses, then 3 exact break-evens. Old logic would count all 6
    # as consecutive losses and pause the symbol; correct logic keeps the
    # streak at 3 (break-evens are neither wins nor losses).
    for _ in range(3):
        _close(state, "UBUSDT", -0.001, -1.0, settings)
    for _ in range(3):
        _close(state, "UBUSDT", 0.0, 0.0, settings)

    stats = state.symbols["UBUSDT"]
    assert stats.consecutive_losses == 3
    assert stats.live_losses == 3
    assert stats.live_wins == 0
    assert stats.live_state != "awaiting_operator"


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


def test_symbol_realized_pnl_12h_computed_from_pnl_events_not_accumulated():
    """The UI's per-symbol 12h PnL must match what the loss-limit guard sees:
    both are derived from ``guards.pnl_events`` (timestamp-pruned). A parallel
    lifetime accumulator would silently drift and mislead operators."""

    import time as _t

    state = _state("UBUSDT")
    state.symbols["BIOUSDT"] = SymbolStats(
        symbol="BIOUSDT", position_size_usd=6.0, started_at=1.0, added_at=1.0
    )
    settings = _settings()

    # Close a losing trade on UBUSDT a few seconds ago -> counted.
    record_pnl(state.guards, -0.50, symbol="UBUSDT")
    record_trade_outcome(state, symbol="UBUSDT", pnl_usd=-0.50, net_bp=-50.0, settings=settings)

    # Close a winning trade on BIOUSDT -> counted, separate symbol.
    record_pnl(state.guards, 0.30, symbol="BIOUSDT")
    record_trade_outcome(state, symbol="BIOUSDT", pnl_usd=0.30, net_bp=30.0, settings=settings)

    # Forge a stale event on UBUSDT from 24h ago -> must NOT show in 12h value.
    state.guards.pnl_events.append((_t.time() - 86400.0, "UBUSDT", -999.0))

    snapshot = state.snapshot()
    by_symbol = {s["symbol"]: s for s in snapshot["symbols"]}

    # UBUSDT has one in-window loss (-0.50); the 24h-old forged event is ignored.
    assert by_symbol["UBUSDT"]["symbol_realized_pnl_12h"] == -0.50
    assert by_symbol["BIOUSDT"]["symbol_realized_pnl_12h"] == 0.30

    # The dataclass field must be gone so nothing can resurrect the
    # lifetime-accumulator bug by writing to it.
    assert not hasattr(state.symbols["UBUSDT"], "symbol_realized_pnl_12h")
