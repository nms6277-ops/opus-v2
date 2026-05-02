import time

from backend.settings_store import RuntimeSettings
from backend.traders.live_state import ExecutionMode, LiveSymbolRuntime, LiveSymbolState


def test_new_symbol_starts_in_paper():
    state = LiveSymbolRuntime(symbol="UBUSDT")
    assert state.execution_mode == ExecutionMode.PAPER
    assert state.live_state == LiveSymbolState.PAPER
    assert state.current_notional_usd == 0.0


def test_arm_live_starts_probation_with_probation_notional():
    state = LiveSymbolRuntime(symbol="UBUSDT")
    state.arm_live(RuntimeSettings(probation_notional_usd=6.0))
    assert state.execution_mode == ExecutionMode.LIVE
    assert state.live_state == LiveSymbolState.PROBATION_LIVE
    assert state.current_notional_usd == 6.0


def test_probation_promotes_after_seven_positive_trade_sample():
    settings = RuntimeSettings(probation_trades=7, active_notional_usd=20.0)
    state = LiveSymbolRuntime(symbol="UBUSDT")
    state.arm_live(settings)

    for net_bp in [5, 6, -2, 4, 8, -1, 3]:
        state.record_closed_trade(net_bp=net_bp, pnl_usd=0.01 if net_bp > 0 else -0.002, settings=settings)

    assert state.live_state == LiveSymbolState.ACTIVE_LIVE
    assert state.current_notional_usd == 20.0
    assert state.wins == 5
    assert state.losses == 2


def test_probation_cools_down_on_bad_first_sample():
    settings = RuntimeSettings(probation_trades=7, cooldown_hours=12)
    state = LiveSymbolRuntime(symbol="RIVERUSDT")
    state.arm_live(settings)

    for net_bp in [5, -10, 4, -10, 3, -10, -10]:
        state.record_closed_trade(net_bp=net_bp, pnl_usd=0.003 if net_bp > 0 else -0.01, settings=settings)

    assert state.live_state == LiveSymbolState.COOLDOWN
    assert state.cooldown_until > time.time()
    assert "probation failed" in state.block_reason


def test_consecutive_losses_cool_down_before_probation_finishes():
    settings = RuntimeSettings(probation_trades=7)
    state = LiveSymbolRuntime(symbol="RIVERUSDT")
    state.arm_live(settings)

    for _ in range(4):
        state.record_closed_trade(net_bp=-3.0, pnl_usd=-0.003, settings=settings)

    assert state.live_state == LiveSymbolState.COOLDOWN
    assert "consecutive losses" in state.block_reason


def test_symbol_loss_cap_cools_down_symbol():
    settings = RuntimeSettings(symbol_loss_limit_usd=0.30)
    state = LiveSymbolRuntime(symbol="RIVERUSDT")
    state.arm_live(settings)

    state.record_closed_trade(net_bp=-40.0, pnl_usd=-0.31, settings=settings)

    assert state.live_state == LiveSymbolState.COOLDOWN
    assert "symbol loss limit" in state.block_reason


def test_can_trade_requires_live_state_and_tradeability():
    settings = RuntimeSettings()
    state = LiveSymbolRuntime(symbol="UBUSDT")
    state.arm_live(settings)

    allowed, reason = state.can_trade(now=time.time(), tradeability_ok=True)
    assert allowed is True
    assert reason == ""

    allowed, reason = state.can_trade(now=time.time(), tradeability_ok=False)
    assert allowed is False
    assert reason == "tradeability filter blocked"
