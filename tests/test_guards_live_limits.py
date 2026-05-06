import time

from backend.safety.guards import OrderIntent, check, record_pnl
from backend.state import AppState, SymbolStats


def _intent(symbol: str = "UBUSDT") -> OrderIntent:
    return OrderIntent(symbol=symbol, side="BUY", qty=1.0, price=1.0, notional_usd=6.0)


def _live_symbol(symbol: str) -> SymbolStats:
    stats = SymbolStats(symbol=symbol, position_size_usd=6.0, started_at=time.time(), added_at=time.time())
    stats.execution_mode = "live"
    stats.live_state = "active_live"
    return stats


def test_max_live_symbols_counts_explicit_live_state():
    state = AppState()
    state.binance_connected = True
    state.binance_last_msg_ts = time.time()
    state.guards.max_live_symbols = 5
    for idx in range(5):
        state.symbols[f"S{idx}USDT"] = _live_symbol(f"S{idx}USDT")

    reason = check(state, _intent("SIXTHUSDT"), "live")

    assert reason is not None
    assert "live symbol limit" in reason


def test_existing_live_symbol_is_not_counted_as_new_symbol():
    state = AppState()
    state.binance_connected = True
    state.binance_last_msg_ts = time.time()
    state.guards.max_live_symbols = 1
    state.symbols["UBUSDT"] = _live_symbol("UBUSDT")

    assert check(state, _intent("UBUSDT"), "live") is None


def test_daily_loss_cap_rejects_orders():
    state = AppState()
    state.guards.daily_loss_limit_usd = 2.0
    record_pnl(state.guards, -2.01, symbol="UBUSDT")

    reason = check(state, _intent(), "paper")

    assert reason is not None
    assert "daily loss limit" in reason


def test_12h_loss_cap_rejects_all_symbols():
    state = AppState()
    state.guards.daily_loss_limit_usd = 999.0
    state.guards.loss_12h_limit_usd = 2.0
    record_pnl(state.guards, -2.01, symbol="UBUSDT")

    reason = check(state, _intent("MEGAUSDT"), "paper")

    assert reason is not None
    assert "12h loss limit" in reason


def test_symbol_loss_cap_rejects_only_that_symbol():
    state = AppState()
    state.guards.daily_loss_limit_usd = 999.0
    state.guards.loss_12h_limit_usd = 999.0
    state.guards.symbol_loss_limit_usd = 0.30
    record_pnl(state.guards, -0.31, symbol="UBUSDT")

    blocked = check(state, _intent("UBUSDT"), "paper")
    allowed = check(state, _intent("MEGAUSDT"), "paper")

    assert blocked is not None
    assert "symbol loss limit" in blocked
    assert allowed is None


def test_operator_emergency_stop_survives_utc_day_rollover():
    """Operator-initiated stops must persist across UTC midnight."""
    from backend.safety.guards import reset_if_new_day, trip_emergency

    state = AppState()
    trip_emergency(state.guards, "operator")
    assert state.guards.emergency_stopped is True
    assert state.guards.emergency_reason == "operator"

    # Force a "new day" by making the bucket stale.
    state.guards.day_bucket_utc = 19000101
    state.guards.daily_pnl = -1.23
    reset_if_new_day(state.guards)

    # Daily counters reset, but the emergency stop is preserved.
    assert state.guards.daily_pnl == 0.0
    assert state.guards.emergency_stopped is True
    assert state.guards.emergency_reason == "operator"

    # And `check()` keeps blocking until /api/emergency/clear is called.
    blocked = check(state, _intent("UBUSDT"), "paper")
    assert blocked is not None
    assert "emergency stop" in blocked


def test_guards_defaults_are_clamped_to_hard_caps():
    """Initial Guards must respect env hard caps even before the operator
    hits ``POST /api/settings``. Otherwise the bot would tolerate losses
    up to the (unbounded) soft env defaults on first startup, even when
    OPUS_HARD_DAILY_LOSS_USD was explicitly pinned lower."""

    from backend.config import settings
    from backend.state import Guards

    g = Guards()
    assert g.daily_loss_limit_usd <= settings.hard_daily_loss_usd
    assert g.max_position_usd <= settings.hard_max_notional_usd
    assert g.max_live_symbols <= settings.hard_max_live_symbols
    assert g.max_orders_per_min <= settings.hard_max_orders_per_min
