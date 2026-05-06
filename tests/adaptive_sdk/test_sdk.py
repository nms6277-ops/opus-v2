"""Integration tests for AdaptiveAnalyticsSDK."""

from __future__ import annotations

import pytest

from backend.adaptive_sdk.sdk import AdaptiveAnalyticsSDK, _has_required_price_excursion
from backend.adaptive_sdk.types import (
    BookSnapshot,
    BucketFillMode,
    BucketSizePolicy,
    ClassificationMode,
    ExhaustionType,
    GlobalConfig,
    Outcome,
    SymbolConfig,
    TradeTick,
)


def _tick(symbol: str, price: float, qty: float, buyer_aggr: bool, ts: float) -> TradeTick:
    return TradeTick(
        symbol=symbol,
        price=price,
        quantity=qty,
        is_buyer_maker=not buyer_aggr,
        timestamp=ts,
    )


def _fast_cfg(**overrides) -> SymbolConfig:
    """Config with aggressive warmup so tests don't need to run thousands of ticks."""
    base = dict(
        classification_mode=ClassificationMode.TRADE_SIGN,
        bucket_fill_mode=BucketFillMode.EXACT_FILL,
        bucket_policy=BucketSizePolicy.FIXED,
        # Bucket sized so that a single phase-2 burst does NOT close it -- VPIN
        # stays well below vpin_high during the test burst.
        fixed_bucket_size=10.0,
        vpin_window=10,
        min_buckets_for_vpin=5,
        min_ticks_for_z=10,
        price_extrema_window_ms=1000,
        flow_window_ms=200,
        flow_lambda=0.3,
        vpin_mid=0.3,
        vpin_high=0.9,
        z_thresholds=(0.5, 1.0, 1.5),
        z_scale=0.3,
    )
    base.update(overrides)
    return SymbolConfig(**base)


# ---------------------------------------------------------------------------
# Lifecycle & warmup
# ---------------------------------------------------------------------------


def test_unregistered_symbol_raises_on_trade():
    sdk = AdaptiveAnalyticsSDK(GlobalConfig())
    with pytest.raises(ValueError):
        sdk.on_trade(_tick("X", 100.0, 1.0, True, 0.0))


def test_register_symbol_is_idempotent():
    sdk = AdaptiveAnalyticsSDK(GlobalConfig())
    sdk.register_symbol("X")
    sdk.register_symbol("X")  # no-op
    state = sdk.get_state("X")
    assert state.is_ready is False


def test_no_signals_emitted_during_warmup():
    sdk = AdaptiveAnalyticsSDK(GlobalConfig())
    sdk.register_symbol("X", _fast_cfg())
    # Before warmup: feed a few ticks, no signal should ever fire.
    for i in range(3):
        sig = sdk.on_trade(_tick("X", 100.0, 2.0, True, float(i) * 0.01))
        assert sig is None
    state = sdk.get_state("X")
    assert state.is_ready is False


# ---------------------------------------------------------------------------
# Toxicity gate
# ---------------------------------------------------------------------------


def test_toxicity_gate_blocks_signals_when_vpin_high():
    cfg = _fast_cfg(vpin_high=0.01)  # practically always tripped
    sdk = AdaptiveAnalyticsSDK(GlobalConfig())
    sdk.register_symbol("X", cfg)
    # Drive warmup with strong one-sided flow to guarantee VPIN > 0.01.
    # All-buy trades => every bucket is fully buy => VPIN == 1.0.
    for i in range(50):
        sdk.on_trade(_tick("X", 100.0 + i * 0.01, 1.0, True, float(i) * 0.01))
    state = sdk.get_state("X")
    assert state.is_ready
    assert state.vpin > cfg.vpin_high
    # Another tick must not fire even if z-exhaustion triggers would be met.
    sig = sdk.on_trade(_tick("X", 100.5, 50.0, False, 100.0))
    assert sig is None


def test_sell_exhaustion_confidence_prefers_bid_pressure_for_bounce():
    cfg = _fast_cfg(vpin_mid=0.2, vpin_high=0.9)

    bid_pressure = AdaptiveAnalyticsSDK._compute_confidence(
        z_relevant=2.0,
        z_threshold=1.0,
        cfg=cfg,
        vpin=0.3,
        obi=0.8,
        has_book=True,
        exhaustion_type=ExhaustionType.SELL_EXHAUSTION,
    )
    ask_pressure = AdaptiveAnalyticsSDK._compute_confidence(
        z_relevant=2.0,
        z_threshold=1.0,
        cfg=cfg,
        vpin=0.3,
        obi=-0.8,
        has_book=True,
        exhaustion_type=ExhaustionType.SELL_EXHAUSTION,
    )

    assert bid_pressure > ask_pressure


def test_buy_exhaustion_confidence_prefers_ask_pressure_for_fade():
    cfg = _fast_cfg(vpin_mid=0.2, vpin_high=0.9)

    ask_pressure = AdaptiveAnalyticsSDK._compute_confidence(
        z_relevant=2.0,
        z_threshold=1.0,
        cfg=cfg,
        vpin=0.3,
        obi=-0.8,
        has_book=True,
        exhaustion_type=ExhaustionType.BUY_EXHAUSTION,
    )
    bid_pressure = AdaptiveAnalyticsSDK._compute_confidence(
        z_relevant=2.0,
        z_threshold=1.0,
        cfg=cfg,
        vpin=0.3,
        obi=0.8,
        has_book=True,
        exhaustion_type=ExhaustionType.BUY_EXHAUSTION,
    )

    assert ask_pressure > bid_pressure


def test_sell_exhaustion_requires_prior_down_move_when_enabled():
    cfg = _fast_cfg(
        min_price_excursion_bps=2.0,
        min_price_excursion_vol_multiplier=0.5,
    )

    assert not _has_required_price_excursion(
        cfg=cfg,
        exhaustion_type=ExhaustionType.SELL_EXHAUSTION,
        price=100.0,
        local_high=100.02,
        local_low=99.98,
        realized_vol=0.002,
    )

    assert _has_required_price_excursion(
        cfg=cfg,
        exhaustion_type=ExhaustionType.SELL_EXHAUSTION,
        price=99.80,
        local_high=100.20,
        local_low=99.70,
        realized_vol=0.002,
    )


def test_buy_exhaustion_requires_prior_up_move_when_enabled():
    cfg = _fast_cfg(
        min_price_excursion_bps=2.0,
        min_price_excursion_vol_multiplier=0.5,
    )

    assert not _has_required_price_excursion(
        cfg=cfg,
        exhaustion_type=ExhaustionType.BUY_EXHAUSTION,
        price=100.0,
        local_high=100.02,
        local_low=99.98,
        realized_vol=0.002,
    )

    assert _has_required_price_excursion(
        cfg=cfg,
        exhaustion_type=ExhaustionType.BUY_EXHAUSTION,
        price=100.20,
        local_high=100.30,
        local_low=99.80,
        realized_vol=0.002,
    )


# ---------------------------------------------------------------------------
# Signal emission end-to-end
# ---------------------------------------------------------------------------


def _drive_warmup_and_force_sell_exhaustion(sdk, symbol: str = "X"):
    """Force a SELL_EXHAUSTION signal with deterministic randomness.

    Returns the first emitted :class:`Signal` (asserts one exists).
    """
    import random

    random.seed(11)
    ts = 0.0

    # Phase 1: warmup -- small mixed-side trades in a narrow price range with
    # JITTERED volumes so EWVar on flow becomes non-zero.
    for i in range(200):
        side = i % 2 == 0
        price = 100.0 + (i % 5) * 0.01 + random.uniform(-0.005, 0.005)
        qty = 0.25 + random.uniform(0.0, 0.2)
        sdk.on_trade(_tick(symbol, price, qty, side, ts))
        ts += 0.02

    # Phase 2: dump sustained sells at a price above local_low (~100.00).
    # Each trade is a huge sell burst; Z-score spikes, price stays above lows.
    for _ in range(60):
        sig = sdk.on_trade(_tick(symbol, 100.10, 5.0, False, ts))
        ts += 0.01
        if sig is not None:
            return sig
    raise AssertionError("expected a SELL_EXHAUSTION signal from sustained burst")


def test_sell_exhaustion_signal_fires():
    cfg = _fast_cfg(
        vpin_window=5,
        min_buckets_for_vpin=5,
        min_ticks_for_z=50,
        flow_window_ms=200,
        price_extrema_window_ms=3000,
    )
    sdk = AdaptiveAnalyticsSDK(GlobalConfig())
    sdk.register_symbol("X", cfg)
    sig = _drive_warmup_and_force_sell_exhaustion(sdk, "X")
    assert sig.symbol == "X"
    assert 0.0 <= sig.confidence <= 1.0
    assert sig.metrics.vpin >= 0.0
    assert sig.arm_id in (0, 1, 2)


# ---------------------------------------------------------------------------
# Zone of interest
# ---------------------------------------------------------------------------


def test_zone_of_interest_blocks_signals_outside_range():
    import random

    random.seed(13)
    cfg = _fast_cfg(min_ticks_for_z=50, price_extrema_window_ms=3000)
    sdk = AdaptiveAnalyticsSDK(GlobalConfig())
    sdk.register_symbol("X", cfg)

    ts = 0.0
    for i in range(200):
        side = i % 2 == 0
        price = 100.0 + (i % 5) * 0.01 + random.uniform(-0.005, 0.005)
        qty = 0.25 + random.uniform(0.0, 0.2)
        sdk.on_trade(_tick("X", price, qty, side, ts))
        ts += 0.02
    assert sdk.get_state("X").is_ready

    # Zone far from current price => all signals blocked even under heavy flow.
    sdk.set_zone_of_interest("X", upper_bound=300.0, lower_bound=200.0)
    for _ in range(60):
        sig = sdk.on_trade(_tick("X", 100.10, 5.0, False, ts))
        assert sig is None
        ts += 0.01


def test_zone_of_interest_bad_bounds_raises():
    sdk = AdaptiveAnalyticsSDK(GlobalConfig())
    sdk.register_symbol("X")
    with pytest.raises(ValueError):
        sdk.set_zone_of_interest("X", upper_bound=1.0, lower_bound=2.0)


# ---------------------------------------------------------------------------
# report_outcome idempotency & MAB feedback
# ---------------------------------------------------------------------------


def test_report_outcome_unknown_signal_returns_false():
    sdk = AdaptiveAnalyticsSDK(GlobalConfig())
    sdk.register_symbol("X")
    outcome = Outcome(
        signal_id="does-not-exist",
        pnl=1.0,
        mfe=2.0,
        mae=0.5,
        fees=0.1,
        slippage=0.05,
        holding_time_ms=1234.0,
    )
    assert sdk.report_outcome("X", outcome) is False


def test_report_outcome_is_idempotent_and_updates_arm():
    cfg = _fast_cfg(min_ticks_for_z=50, price_extrema_window_ms=3000)
    sdk = AdaptiveAnalyticsSDK(GlobalConfig())
    sdk.register_symbol("X", cfg)
    sig = _drive_warmup_and_force_sell_exhaustion(sdk, "X")

    # Capture MAB arm state before.
    ctx = sdk._contexts["X"]
    before = (
        ctx.mab.arms[sig.arm_id].mu,
        ctx.mab.arms[sig.arm_id].kappa,
    )

    outcome = Outcome(
        signal_id=sig.signal_id,
        pnl=10.0,
        mfe=15.0,
        mae=1.0,
        fees=0.5,
        slippage=0.2,
        holding_time_ms=2500.0,
    )
    assert sdk.report_outcome("X", outcome) is True

    after_first = (
        ctx.mab.arms[sig.arm_id].mu,
        ctx.mab.arms[sig.arm_id].kappa,
    )
    assert after_first != before  # arm state moved
    assert after_first[1] == before[1] + 1.0  # kappa increments by 1

    # Second call with same id: must NOT update arm state again.
    assert sdk.report_outcome("X", outcome) is False
    after_second = (
        ctx.mab.arms[sig.arm_id].mu,
        ctx.mab.arms[sig.arm_id].kappa,
    )
    assert after_second == after_first


def test_report_outcome_wrong_symbol_returns_false():
    cfg = _fast_cfg(min_ticks_for_z=50, price_extrema_window_ms=3000)
    sdk = AdaptiveAnalyticsSDK(GlobalConfig())
    sdk.register_symbol("X", cfg)
    sdk.register_symbol("Y", cfg)
    sig = _drive_warmup_and_force_sell_exhaustion(sdk, "X")

    outcome = Outcome(
        signal_id=sig.signal_id,
        pnl=1.0,
        mfe=2.0,
        mae=1.0,
        fees=0.1,
        slippage=0.1,
        holding_time_ms=1000.0,
    )
    assert sdk.report_outcome("Y", outcome) is False  # wrong symbol


def test_custom_reward_transform_is_used():
    calls: list[Outcome] = []

    def scorer(o: Outcome) -> float:
        calls.append(o)
        return 42.0

    cfg = _fast_cfg(
        min_ticks_for_z=50,
        price_extrema_window_ms=3000,
        reward_transform=scorer,
    )
    sdk = AdaptiveAnalyticsSDK(GlobalConfig())
    sdk.register_symbol("X", cfg)
    sig = _drive_warmup_and_force_sell_exhaustion(sdk, "X")

    outcome = Outcome(
        signal_id=sig.signal_id,
        pnl=-999.0,
        mfe=0.0,
        mae=0.0,
        fees=0.0,
        slippage=0.0,
        holding_time_ms=0.0,
    )
    assert sdk.report_outcome("X", outcome) is True
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Pending signals: TTL & size cap
# ---------------------------------------------------------------------------


def test_pending_signals_size_cap_evicts_oldest():
    cfg = _fast_cfg(pending_signals_max=3, pending_cleanup_interval_ticks=10_000)
    sdk = AdaptiveAnalyticsSDK(GlobalConfig())
    # Simulate directly at the private API level for determinism.
    sdk._register_pending("a", "X", 0, ts=0.0, max_size=cfg.pending_signals_max)
    sdk._register_pending("b", "X", 0, ts=1.0, max_size=cfg.pending_signals_max)
    sdk._register_pending("c", "X", 0, ts=2.0, max_size=cfg.pending_signals_max)
    sdk._register_pending("d", "X", 0, ts=3.0, max_size=cfg.pending_signals_max)
    assert "a" not in sdk._pending
    assert set(sdk._pending.keys()) == {"b", "c", "d"}


def test_pending_signals_ttl_cleanup():
    sdk = AdaptiveAnalyticsSDK(GlobalConfig())
    sdk._register_pending("old", "X", 0, ts=0.0, max_size=100)
    sdk._register_pending("new", "X", 0, ts=180.0, max_size=100)
    # At now=200, ttl=50 -> cutoff=150. "old" (ts=0) is expired; "new" (ts=180) is kept.
    sdk._cleanup_pending(now=200.0, ttl=50.0)
    assert "old" not in sdk._pending
    assert "new" in sdk._pending


# ---------------------------------------------------------------------------
# OBI / on_book_update
# ---------------------------------------------------------------------------


def test_on_book_update_sets_obi():
    sdk = AdaptiveAnalyticsSDK(GlobalConfig())
    sdk.register_symbol("X")
    sdk.on_book_update(
        BookSnapshot(
            symbol="X",
            best_bid=100.0,
            best_ask=100.1,
            bid_vol=30.0,
            ask_vol=10.0,
            timestamp=0.0,
        )
    )
    ctx = sdk._contexts["X"]
    assert ctx.has_book_data is True
    assert ctx.obi == pytest.approx((30.0 - 10.0) / 40.0)


def test_empty_book_is_ignored():
    sdk = AdaptiveAnalyticsSDK(GlobalConfig())
    sdk.register_symbol("X")
    sdk.on_book_update(
        BookSnapshot(
            symbol="X",
            best_bid=0.0,
            best_ask=0.0,
            bid_vol=0.0,
            ask_vol=0.0,
            timestamp=0.0,
        )
    )
    ctx = sdk._contexts["X"]
    assert ctx.has_book_data is False


# ---------------------------------------------------------------------------
# Cross-symbol isolation
# ---------------------------------------------------------------------------


def test_symbols_are_isolated():
    sdk = AdaptiveAnalyticsSDK(GlobalConfig())
    sdk.register_symbol("X", _fast_cfg())
    sdk.register_symbol("Y", _fast_cfg())

    # Drive X heavily; Y receives nothing.
    ts = 0.0
    for _i in range(40):
        sdk.on_trade(_tick("X", 100.0, 1.0, True, ts))
        ts += 0.02

    state_x = sdk.get_state("X")
    state_y = sdk.get_state("Y")
    assert state_x.buckets_filled > 0
    assert state_y.buckets_filled == 0
    assert state_y.is_ready is False
