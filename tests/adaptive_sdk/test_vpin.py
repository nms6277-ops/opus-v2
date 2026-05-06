"""Tests for TrueVPINEngine.

Covers:
* volume conservation under EXACT_FILL trade splitting,
* single-trade overflow across multiple buckets,
* BVC classification math (Phi(0)=0.5, monotonicity, symmetry),
* BVC warmup: no buckets emitted until sigma tracker is ready,
* MINIMUM_FILL does not split trades,
* rolling-daily bucket sizing.
"""

from __future__ import annotations

import pytest

from backend.adaptive_sdk.types import (
    BucketFillMode,
    BucketSizePolicy,
    ClassificationMode,
    SymbolConfig,
    TradeTick,
)
from backend.adaptive_sdk.vpin import TrueVPINEngine, _phi


def _make_tick(qty: float, price: float = 100.0, buyer_aggr: bool = True, ts: float = 0.0) -> TradeTick:
    # buyer_aggr=True means buyer is aggressor, i.e. is_buyer_maker=False.
    return TradeTick(
        symbol="X",
        price=price,
        quantity=qty,
        is_buyer_maker=not buyer_aggr,
        timestamp=ts,
    )


# ---------------------------------------------------------------------------
# TRADE_SIGN + EXACT_FILL
# ---------------------------------------------------------------------------


def test_exact_fill_conserves_volume_across_splits():
    cfg = SymbolConfig(
        classification_mode=ClassificationMode.TRADE_SIGN,
        bucket_fill_mode=BucketFillMode.EXACT_FILL,
        fixed_bucket_size=1.0,
        vpin_window=100,
    )
    eng = TrueVPINEngine(cfg)
    # One giant trade of qty=10 into bucket_size=1 => 10 closed buckets.
    eng.on_trade(_make_tick(qty=10.0))
    assert eng.buckets_filled == 10
    # All buys (buyer_aggr=True) => every bucket is fully buy-classified.
    for _ in range(eng.buckets_filled):
        pass  # iteration below
    buckets = list(eng._closed)  # internal but useful here
    assert sum(b.v_buy for b in buckets) == pytest.approx(10.0)
    assert sum(b.v_sell for b in buckets) == pytest.approx(0.0)
    assert sum(b.v_total for b in buckets) == pytest.approx(10.0)


def test_exact_fill_preserves_total_across_many_ticks():
    """Conservation: sum(V_buy + V_sell) over closed buckets equals ingested volume."""
    cfg = SymbolConfig(
        classification_mode=ClassificationMode.TRADE_SIGN,
        bucket_fill_mode=BucketFillMode.EXACT_FILL,
        fixed_bucket_size=3.7,
        vpin_window=1000,
    )
    eng = TrueVPINEngine(cfg)
    rng_qtys = [0.5, 2.0, 1.3, 7.2, 0.1, 9.9, 4.4, 2.5, 8.8, 0.9]
    sides = [True, False, True, False, True, True, False, False, True, False]
    total = 0.0
    for q, s in zip(rng_qtys, sides, strict=True):
        total += q
        eng.on_trade(_make_tick(qty=q, buyer_aggr=s))
    closed_total = sum(b.v_total for b in eng._closed)
    # The *current* open bucket holds residual; account for it.
    residual = eng._curr.v_total if eng._curr is not None else 0.0
    assert closed_total + residual == pytest.approx(total, rel=1e-10)


def test_exact_fill_overflows_into_multiple_buckets():
    cfg = SymbolConfig(
        classification_mode=ClassificationMode.TRADE_SIGN,
        bucket_fill_mode=BucketFillMode.EXACT_FILL,
        fixed_bucket_size=2.0,
        vpin_window=100,
    )
    eng = TrueVPINEngine(cfg)
    # First trade: qty=5 (fills 2 whole buckets, leaves 1 in current).
    eng.on_trade(_make_tick(qty=5.0))
    assert eng.buckets_filled == 2
    # Each closed bucket has v_total == 2.0.
    for b in eng._closed:
        assert b.v_total == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# TRADE_SIGN + MINIMUM_FILL
# ---------------------------------------------------------------------------


def test_minimum_fill_does_not_split():
    cfg = SymbolConfig(
        classification_mode=ClassificationMode.TRADE_SIGN,
        bucket_fill_mode=BucketFillMode.MINIMUM_FILL,
        fixed_bucket_size=1.0,
        vpin_window=100,
    )
    eng = TrueVPINEngine(cfg)
    eng.on_trade(_make_tick(qty=5.0))
    # One trade => one overflow => exactly one bucket closed, with v_total=5.
    assert eng.buckets_filled == 1
    b = eng._closed[0]
    assert b.v_total == pytest.approx(5.0)
    assert b.v_buy == pytest.approx(5.0)  # buyer aggressor
    assert b.v_sell == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BVC math
# ---------------------------------------------------------------------------


def test_phi_of_zero_is_half():
    assert _phi(0.0) == pytest.approx(0.5, abs=1e-12)


def test_phi_monotonic_in_r():
    # Monotone increasing in its argument.
    a, b, c = _phi(-1.0), _phi(0.0), _phi(1.0)
    assert a < b < c
    assert a == pytest.approx(1.0 - c, abs=1e-12)  # symmetry: Phi(-x) = 1 - Phi(x)


def test_bvc_classifies_flat_bucket_as_fifty_fifty():
    """With P_open == P_close, Phi(0) == 0.5 => V_buy == V_sell."""
    cfg = SymbolConfig(
        classification_mode=ClassificationMode.BVC,
        bucket_fill_mode=BucketFillMode.MINIMUM_FILL,
        fixed_bucket_size=1.0,
        min_buckets_for_bvc_sigma=1,  # warm up fast for test
        vpin_window=100,
    )
    eng = TrueVPINEngine(cfg)
    # Prime sigma with a varying close first.
    eng.on_trade(_make_tick(qty=1.5, price=100.0))  # closes bucket, seeds tracker
    eng.on_trade(_make_tick(qty=1.5, price=101.0))  # r_1 produced
    # Now feed a flat bucket at price=101.
    eng.on_trade(_make_tick(qty=1.5, price=101.0))
    last = eng._closed[-1]
    # Most recent bucket: p_open == p_close == 101 => 50/50.
    assert last.v_buy == pytest.approx(last.v_sell, rel=1e-9)
    assert last.v_buy + last.v_sell == pytest.approx(last.v_total)


def test_bvc_upward_move_classifies_mostly_buy():
    # Use EXACT_FILL so we can get multiple trades per bucket and a real intra-bar swing.
    cfg = SymbolConfig(
        classification_mode=ClassificationMode.BVC,
        bucket_fill_mode=BucketFillMode.EXACT_FILL,
        fixed_bucket_size=2.0,
        min_buckets_for_bvc_sigma=2,
        vpin_window=100,
    )
    eng = TrueVPINEngine(cfg)
    # Warm sigma tracker: 4 buckets produce 3 returns, >= min_buckets_for_bvc_sigma=2.
    eng.on_trade(_make_tick(qty=2.0, price=100.0))  # bucket closes: p_open=p_close=100
    eng.on_trade(_make_tick(qty=2.0, price=100.5))  # closes at 100.5
    eng.on_trade(_make_tick(qty=2.0, price=101.0))  # closes at 101.0
    assert eng.bvc_ready
    # Now assemble a bucket with a BIG intra-bar upward swing.
    # Each trade below is 1.0 unit, so the bucket fills after two trades.
    eng.on_trade(_make_tick(qty=1.0, price=101.0))  # starts new bucket at 101
    eng.on_trade(_make_tick(qty=1.0, price=105.0))  # closes it at 105
    last = eng._closed[-1]
    assert last.p_open == pytest.approx(101.0)
    assert last.p_close == pytest.approx(105.0)
    assert last.v_buy > last.v_sell
    assert last.v_buy > 0.9 * last.v_total  # strong buy bias


def test_bvc_does_not_emit_during_sigma_warmup():
    cfg = SymbolConfig(
        classification_mode=ClassificationMode.BVC,
        bucket_fill_mode=BucketFillMode.MINIMUM_FILL,
        fixed_bucket_size=1.0,
        min_buckets_for_bvc_sigma=5,  # need 5 cross-bucket returns to classify
        vpin_window=100,
    )
    eng = TrueVPINEngine(cfg)
    # 5 bucket closes produce only 4 returns (first close has no prior). Not yet ready.
    for i in range(5):
        eng.on_trade(_make_tick(qty=1.5, price=100.0 + i))
    assert eng.buckets_filled == 0, "no classified buckets before sigma warmup"
    assert not eng.bvc_ready
    # 6th close -> 5 returns, tracker becomes ready AFTER this close.
    eng.on_trade(_make_tick(qty=1.5, price=105.0))
    assert eng.bvc_ready
    # This close was evaluated BEFORE the tracker became ready, so still 0 classified.
    assert eng.buckets_filled == 0
    # 7th close -> tracker ready at evaluation time; first classified bucket appears.
    eng.on_trade(_make_tick(qty=1.5, price=106.0))
    assert eng.buckets_filled == 1


# ---------------------------------------------------------------------------
# BVC + EXACT_FILL (split without signing)
# ---------------------------------------------------------------------------


def test_bvc_exact_fill_splits_volume_unsigned():
    cfg = SymbolConfig(
        classification_mode=ClassificationMode.BVC,
        bucket_fill_mode=BucketFillMode.EXACT_FILL,
        fixed_bucket_size=1.0,
        min_buckets_for_bvc_sigma=1,
        vpin_window=100,
    )
    eng = TrueVPINEngine(cfg)
    # Warmup
    eng.on_trade(_make_tick(qty=1.0, price=100.0))
    eng.on_trade(_make_tick(qty=1.0, price=101.0))
    buckets_before = eng.buckets_filled
    # Feed 5 units at price 102 => 5 new buckets (each of size 1).
    eng.on_trade(_make_tick(qty=5.0, price=102.0))
    assert eng.buckets_filled - buckets_before == 5


# ---------------------------------------------------------------------------
# ROLLING_DAILY bucket sizing
# ---------------------------------------------------------------------------


def test_rolling_daily_recomputes_bucket_size():
    cfg = SymbolConfig(
        classification_mode=ClassificationMode.TRADE_SIGN,
        bucket_fill_mode=BucketFillMode.EXACT_FILL,
        bucket_policy=BucketSizePolicy.ROLLING_DAILY,
        fixed_bucket_size=1.0,
        target_buckets_per_day=10,
        min_bucket_size=0.01,
        bucket_size_recompute_interval_s=0.0,  # recompute on every tick
        vpin_window=1000,
    )
    eng = TrueVPINEngine(cfg)
    # Feed 1000 units of volume over 1 hour => bucket_size should target 100.
    for i in range(100):
        eng.on_trade(_make_tick(qty=10.0, price=100.0, ts=float(i) * 36.0))
    # After many ticks the size should be roughly rolling_vol / 10.
    # We fed 1000 units total; target_buckets_per_day=10 => size=100.
    assert eng.bucket_size == pytest.approx(100.0, rel=0.01)


def test_rolling_daily_respects_min_bucket_size():
    cfg = SymbolConfig(
        classification_mode=ClassificationMode.TRADE_SIGN,
        bucket_fill_mode=BucketFillMode.EXACT_FILL,
        bucket_policy=BucketSizePolicy.ROLLING_DAILY,
        fixed_bucket_size=1.0,
        target_buckets_per_day=10_000_000,  # absurdly large => would collapse size
        min_bucket_size=5.0,
        bucket_size_recompute_interval_s=0.0,
        vpin_window=1000,
    )
    eng = TrueVPINEngine(cfg)
    for i in range(10):
        eng.on_trade(_make_tick(qty=1.0, price=100.0, ts=float(i)))
    assert eng.bucket_size >= 5.0


def test_rolling_daily_can_use_configured_one_minute_window():
    cfg = SymbolConfig(
        classification_mode=ClassificationMode.TRADE_SIGN,
        bucket_fill_mode=BucketFillMode.EXACT_FILL,
        bucket_policy=BucketSizePolicy.ROLLING_DAILY,
        fixed_bucket_size=1.0,
        rolling_volume_window_s=60.0,
        target_buckets_per_window=10,
        min_bucket_size=0.01,
        bucket_size_recompute_interval_s=0.0,
        vpin_window=1000,
    )
    eng = TrueVPINEngine(cfg)

    eng.on_trade(_make_tick(qty=100.0, price=100.0, ts=0.0))
    assert eng.bucket_size == pytest.approx(10.0)

    eng.on_trade(_make_tick(qty=100.0, price=100.0, ts=120.0))

    assert eng.bucket_size == pytest.approx(10.0)
