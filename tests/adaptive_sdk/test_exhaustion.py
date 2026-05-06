"""Tests for EWMA/EWVar, RollingExtrema, and ExhaustionDetector."""

from __future__ import annotations

import pytest

from backend.adaptive_sdk.exhaustion import (
    EWMAEWVar,
    ExhaustionDetector,
    RollingExtrema,
    RollingRealizedVolatility,
)
from backend.adaptive_sdk.types import SymbolConfig, TradeTick

# ---------------------------------------------------------------------------
# EWMAEWVar
# ---------------------------------------------------------------------------


def test_ewma_ewvar_single_observation_is_zero_var():
    stats = EWMAEWVar(lambda_=0.1)
    stats.update(5.0)
    assert stats.mean == 5.0
    assert stats.var == 0.0
    assert stats.z_score(10.0) == 0.0  # undefined while var == 0


def test_ewma_ewvar_converges_to_constant_stream():
    stats = EWMAEWVar(lambda_=0.2)
    for _ in range(500):
        stats.update(7.0)
    assert stats.mean == pytest.approx(7.0, abs=1e-9)
    assert stats.var == pytest.approx(0.0, abs=1e-9)


def test_ewma_ewvar_detects_outliers():
    stats = EWMAEWVar(lambda_=0.1)
    # Feed noisy observations around 10 with std ~1.
    import random

    random.seed(42)
    for _ in range(200):
        stats.update(10.0 + random.gauss(0.0, 1.0))
    # A 10-sigma outlier should score huge.
    z = stats.z_score(20.0)
    assert z > 5.0


# ---------------------------------------------------------------------------
# RollingExtrema (time-windowed)
# ---------------------------------------------------------------------------


def test_rolling_extrema_basic_min_max():
    ex = RollingExtrema(window_ms=1000)
    ex.update(0.0, 100.0)
    ex.update(0.1, 105.0)
    ex.update(0.2, 95.0)
    assert ex.local_max == 105.0
    assert ex.local_min == 95.0


def test_rolling_extrema_evicts_old_entries():
    # window_s = 0.5 -- at ts=0.6 cutoff is 0.1, so only the 0.0 entry expires.
    ex = RollingExtrema(window_ms=500)
    ex.update(0.0, 50.0)  # will expire at ts=0.6 (cutoff=0.1)
    ex.update(0.2, 200.0)  # survives
    ex.update(0.4, 100.0)  # survives
    ex.update(0.6, 110.0)
    assert ex.local_min == 100.0
    assert ex.local_max == 200.0
    # Jump further forward: at ts=1.0, cutoff=0.5 -> only (0.6, 110) survives.
    ex.update(1.0, 110.0)
    assert ex.local_min == 110.0
    assert ex.local_max == 110.0


def test_rolling_extrema_monotonic_deque_property():
    """After many random updates the queried min/max always equal the
    literal min/max of entries inside the time window."""
    import random

    random.seed(7)
    window_ms = 500
    ex = RollingExtrema(window_ms=window_ms)
    history: list[tuple[float, float]] = []
    for i in range(1000):
        ts = i * 0.01  # 10 ms apart
        price = 100.0 + random.gauss(0.0, 5.0)
        ex.update(ts, price)
        history.append((ts, price))
        # Brute-force reference.
        cutoff = ts - window_ms / 1000.0
        valid = [p for (t, p) in history if t >= cutoff]
        assert ex.local_min == pytest.approx(min(valid))
        assert ex.local_max == pytest.approx(max(valid))


# ---------------------------------------------------------------------------
# Rolling realized volatility
# ---------------------------------------------------------------------------


def test_rolling_realized_volatility_is_zero_for_flat_price():
    rv = RollingRealizedVolatility(window_ms=60_000)

    rv.update(0.0, 100.0)
    rv.update(1.0, 100.0)
    rv.update(2.0, 100.0)

    assert rv.value == pytest.approx(0.0)


def test_rolling_realized_volatility_evicts_old_returns():
    rv = RollingRealizedVolatility(window_ms=1_000)

    rv.update(0.0, 100.0)
    rv.update(0.1, 90.0)
    high_vol = rv.value
    rv.update(2.0, 90.0)

    assert high_vol > 0.0
    assert rv.value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ExhaustionDetector integration
# ---------------------------------------------------------------------------


def _tick(price: float, qty: float, buyer_aggr: bool, ts: float) -> TradeTick:
    return TradeTick(
        symbol="X",
        price=price,
        quantity=qty,
        is_buyer_maker=not buyer_aggr,
        timestamp=ts,
    )


def test_exhaustion_detector_flow_z_score_reacts_to_burst():
    cfg = SymbolConfig(
        flow_window_ms=1000,
        price_extrema_window_ms=3000,
        flow_lambda=0.2,
    )
    det = ExhaustionDetector(cfg)
    # Warm up with 60 seconds of varying baseline sells so EWVar becomes non-zero.
    import random

    random.seed(7)
    for i in range(60):
        qty = 0.1 + 0.05 * random.random()  # ~0.1 - 0.15 per second
        det.on_trade(_tick(price=100.0, qty=qty, buyer_aggr=False, ts=float(i) + 0.01))
    # Roll one more window so the last partial one is committed.
    det.on_trade(_tick(price=100.0, qty=0.11, buyer_aggr=False, ts=60.01))
    baseline_z = det.z_sell_flow
    # Now open a new window and dump huge sell volume mid-window.
    det.on_trade(_tick(price=100.0, qty=50.0, buyer_aggr=False, ts=60.5))
    # Mid-window z should be dramatically higher than baseline.
    assert det.z_sell_flow > baseline_z + 5.0


def test_exhaustion_detector_tracks_price_extrema():
    cfg = SymbolConfig(price_extrema_window_ms=1000, flow_window_ms=1000)
    det = ExhaustionDetector(cfg)
    det.on_trade(_tick(price=100.0, qty=1.0, buyer_aggr=True, ts=0.0))
    det.on_trade(_tick(price=105.0, qty=1.0, buyer_aggr=True, ts=0.1))
    det.on_trade(_tick(price=98.0, qty=1.0, buyer_aggr=False, ts=0.2))
    assert det.local_high == 105.0
    assert det.local_low == 98.0


def test_buyer_maker_routes_to_sell_aggressor():
    """is_buyer_maker=True means seller is aggressor -> should go to curr_sell."""
    cfg = SymbolConfig(flow_window_ms=1000, price_extrema_window_ms=1000)
    det = ExhaustionDetector(cfg)
    det.on_trade(_tick(price=100.0, qty=2.0, buyer_aggr=False, ts=0.0))  # seller aggressor
    # Internal check via Z-score of current window vs EWMA (not initialized yet, returns 0).
    # Use internal state directly.
    assert det._curr_sell == pytest.approx(2.0)
    assert det._curr_buy == pytest.approx(0.0)
