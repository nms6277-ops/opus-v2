"""v2 spread-aware labels (taker round-trip).

The v1 mid-to-mid labelling logic ignored the bid-ask spread completely;
on micro-cap altcoins (AIOT, BSB, ...) where the spread is 3-8 bp and
predicted moves are 1-5 bp, this produced large UP/DOWN training sets
that were never actually profitable to taker-trade. The v2 ``cost_mode=
"taker"`` mode replaces the FLAT band with a per-symbol threshold equal
to ``mean(spread_bp) + 8 bp commission + 3 bp safety margin``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from backend.ml.labels import (
    DEFAULT_HORIZONS,
    LBL_DOWN,
    LBL_FLAT,
    LBL_UP,
    HorizonSpec,
    _classify_taker,
    add_labels,
)


def _synthetic_frame(
    *,
    n: int = 200,
    spread_bp: float = 4.0,
    mid_after_50: float | None = None,
    mid_after_150: float | None = None,
    test_split: int = 150,
) -> pl.DataFrame:
    """Build a synthetic snapshot frame at 100ms cadence with controlled mid drift."""
    ts = np.arange(n) * 100
    base_mid = 100.0
    half = base_mid * spread_bp / (2 * 10_000.0)
    bb = np.full(n, base_mid - half)
    ba = np.full(n, base_mid + half)
    if mid_after_50 is not None:
        bb[50:] = mid_after_50 - half
        ba[50:] = mid_after_50 + half
    if mid_after_150 is not None:
        bb[150:] = mid_after_150 - half
        ba[150:] = mid_after_150 + half
    return pl.DataFrame(
        {
            "symbol": ["AIOTUSDT"] * n,
            "ts_ms": ts,
            "best_bid": bb,
            "best_ask": ba,
            "mid": (bb + ba) / 2,
            "spread_bp": (ba - bb) / ((bb + ba) / 2) * 10_000.0,
            "part": ["train"] * test_split + ["test"] * (n - test_split),
        }
    )


def test_classify_taker_long_only():
    long = np.array([10.0, 20.0, 30.0])
    short = np.array([-5.0, -10.0, -15.0])
    out = _classify_taker(long, short, threshold_bp=15.0)
    np.testing.assert_array_equal(out, np.array([LBL_FLAT, LBL_UP, LBL_UP], dtype=np.int8))


def test_classify_taker_short_only():
    long = np.array([-10.0, -20.0, -30.0])
    short = np.array([10.0, 25.0, 40.0])
    out = _classify_taker(long, short, threshold_bp=15.0)
    np.testing.assert_array_equal(out, np.array([LBL_FLAT, LBL_DOWN, LBL_DOWN], dtype=np.int8))


def test_classify_taker_both_pick_larger():
    long = np.array([20.0, 25.0])
    short = np.array([18.0, 30.0])
    out = _classify_taker(long, short, threshold_bp=15.0)
    np.testing.assert_array_equal(out, np.array([LBL_UP, LBL_DOWN], dtype=np.int8))


def test_taker_threshold_includes_spread_fees_and_margin():
    """Spread=4bp, RT fee=8bp, margin=3bp -> threshold should be ~15bp."""
    df = _synthetic_frame()
    _, thresh = add_labels(df, horizons=DEFAULT_HORIZONS, cost_mode="taker")
    for (sym, _h), t in thresh.items():
        assert sym == "AIOTUSDT"
        assert 14.5 < t < 15.5, f"threshold {t} should be near 15bp"


def test_taker_mode_drops_small_moves_to_flat():
    """A 4bp move at 5s would be UP under cost_mode=mid but FLAT under taker:
    ``gross_long ≈ 4 - 4 = 0 bp`` and threshold is ~15 bp so the row collapses
    to FLAT — the spread eats the edge."""
    df = _synthetic_frame(mid_after_50=100.04)  # +4bp move starting at idx 50
    df_lab, _ = add_labels(df, horizons=DEFAULT_HORIZONS, cost_mode="taker")
    # No row should be UP when the long round-trip never clears 15bp threshold.
    assert (df_lab["y_5s"] == LBL_UP).sum() == 0
    assert (df_lab["y_5s"] == LBL_DOWN).sum() == 0


def test_taker_mode_keeps_large_moves_as_up():
    """A 32bp jump is large enough that the long round-trip beats threshold
    even after deducting the entry-side spread cost."""
    df = _synthetic_frame(mid_after_50=100.32)  # +32bp move at idx 50
    df_lab, thresh = add_labels(df, horizons=DEFAULT_HORIZONS, cost_mode="taker")
    h2 = HorizonSpec("2s", 2_000, 500)
    assert thresh[("AIOTUSDT", h2.name)] < 16.0
    # Rows in (50 - 20, 50) — mid jumps within the 2s horizon — should be UP.
    pre_jump_up = df_lab.filter((pl.col("ts_ms") >= 3000) & (pl.col("ts_ms") < 5000))[
        "y_2s"
    ].to_numpy()
    assert (pre_jump_up == LBL_UP).any()


def test_mid_mode_back_compat_still_works():
    """Legacy v1 mode shouldn't have broken — same call signature, just a
    different ``cost_mode``."""
    df = _synthetic_frame(mid_after_50=100.04)
    df_lab, thresh = add_labels(df, horizons=DEFAULT_HORIZONS, cost_mode="mid")
    # Mid mode picks a small FLAT band and labels small drifts; we just need
    # the call to succeed and produce *some* non-FLAT labels.
    assert any(t > 0.0 for t in thresh.values())
    assert df_lab.height == df.height


def test_invalid_cost_mode_raises():
    df = _synthetic_frame()
    try:
        add_labels(df, horizons=DEFAULT_HORIZONS, cost_mode="bogus")
    except ValueError as e:
        assert "cost_mode" in str(e)
    else:
        raise AssertionError("expected ValueError for cost_mode='bogus'")


def test_labels_emit_long_short_columns():
    df = _synthetic_frame(mid_after_50=100.10)
    df_lab, _ = add_labels(df, horizons=DEFAULT_HORIZONS, cost_mode="taker")
    for h in DEFAULT_HORIZONS:
        assert f"gross_long_{h.name}_bp" in df_lab.columns
        assert f"gross_short_{h.name}_bp" in df_lab.columns
        assert f"ret_{h.name}_bp" in df_lab.columns
        assert f"y_{h.name}_valid" in df_lab.columns
        assert f"y_{h.name}" in df_lab.columns
