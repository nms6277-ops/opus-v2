"""v2 wide-band bucket coverage in :mod:`backend.features.snapshot`.

The v2 schema partitions the order book into fixed-width slices in bp around
mid (default +/- 50 bp every 5 bp), independent of tick size. This guarantees
that an AIOT book with a 4 bp spread populates the same column grid as a BTC
book with a 0.5 bp spread, so a model trained on one book width generalises.
"""

from __future__ import annotations

import time

import pytest

from backend.collector.lob import OrderBook, Trade
from backend.features.snapshot import _band_edges, build


def test_band_edges_default():
    assert _band_edges(50, 5) == (
        (0, 5),
        (5, 10),
        (10, 15),
        (15, 20),
        (20, 25),
        (25, 30),
        (30, 35),
        (35, 40),
        (40, 45),
        (45, 50),
    )


def test_band_edges_uneven():
    # Last band gets clipped to band_bps_max if step doesn't divide it evenly.
    assert _band_edges(12, 5) == ((0, 5), (5, 10), (10, 12))


def test_band_edges_disabled():
    assert _band_edges(0, 5) == ()
    assert _band_edges(50, 0) == ()
    assert _band_edges(5, 50) == ()


@pytest.fixture
def synthetic_book():
    """Order book around mid=100 with explicit qty at known bp offsets."""
    ob = OrderBook("AIOTUSDT")
    mid = 100.0

    def bid_at(bp: float) -> float:
        return mid * (1.0 - bp / 10_000.0)

    def ask_at(bp: float) -> float:
        return mid * (1.0 + bp / 10_000.0)

    ob.bids = {
        bid_at(1.0): 10.0,
        bid_at(7.0): 20.0,
        bid_at(22.0): 30.0,
        bid_at(55.0): 999.0,
    }
    ob.asks = {
        ask_at(3.0): 15.0,
        ask_at(12.0): 25.0,
        ask_at(47.0): 35.0,
        ask_at(60.0): 999.0,
    }
    ob.last_update_id = 1
    ob.ready = True
    ob.add_trade(Trade(ts_ms=int(time.time() * 1000), price=mid, qty=1.0, is_buyer_maker=False))
    return ob


def test_band_buckets_route_qty_to_correct_slice(synthetic_book):
    snap = build(
        synthetic_book,
        depth=5,
        trade_window_ms=2000,
        now_ms=int(time.time() * 1000),
        band_bps_max=50,
        band_bps_step=5,
    )
    assert snap is not None
    row = snap.to_row()
    assert row["band_bid_qty_00_05bp"] == 10.0
    assert row["band_bid_qty_05_10bp"] == 20.0
    assert row["band_bid_qty_10_15bp"] == 0.0
    assert row["band_bid_qty_20_25bp"] == 30.0
    assert row["band_bid_qty_45_50bp"] == 0.0
    assert row["band_ask_qty_00_05bp"] == 15.0
    assert row["band_ask_qty_05_10bp"] == 0.0
    assert row["band_ask_qty_10_15bp"] == 25.0
    assert row["band_ask_qty_30_35bp"] == 0.0
    assert row["band_ask_qty_45_50bp"] == 35.0


def test_band_buckets_drop_levels_outside_max_bp(synthetic_book):
    """Levels deeper than ``band_bps_max`` are silently dropped — so the same
    feature columns make sense for thin (3 bp spread) and thick (50 bp spread)
    books without exploding the schema. The 55 bp bid and 60 bp ask above MUST
    NOT contribute to any band."""
    snap = build(synthetic_book, band_bps_max=50, band_bps_step=5)
    assert snap is not None
    row = snap.to_row()
    total_bid_band = sum(v for k, v in row.items() if k.startswith("band_bid_qty_"))
    total_ask_band = sum(v for k, v in row.items() if k.startswith("band_ask_qty_"))
    assert total_bid_band == 10.0 + 20.0 + 30.0
    assert total_ask_band == 15.0 + 25.0 + 35.0


def test_legacy_columns_still_emitted(synthetic_book):
    """Backward compat: existing v1 columns survive alongside the new bands."""
    snap = build(synthetic_book, band_bps_max=50, band_bps_step=5)
    assert snap is not None
    row = snap.to_row()
    # Legacy cumulative buckets
    assert "bid_bkt_qty_05bp" in row
    assert "ask_bkt_qty_50bp" in row
    # Legacy top-N raw price/qty columns
    assert "bid_p_00" in row
    assert "ask_q_00" in row
    # Always-on aggregates
    assert "spread_bp" in row
    assert "imbalance_top1" in row


def test_sdk_columns_passed_through(synthetic_book):
    snap = build(
        synthetic_book,
        sdk_columns={"sdk_vpin": 0.42, "sdk_buy_flow_z": 1.7},
    )
    assert snap is not None
    row = snap.to_row()
    assert row["sdk_vpin"] == 0.42
    assert row["sdk_buy_flow_z"] == 1.7


def test_band_buckets_disabled_when_max_zero(synthetic_book):
    snap = build(synthetic_book, band_bps_max=0, band_bps_step=5)
    assert snap is not None
    row = snap.to_row()
    assert not any(k.startswith("band_bid_qty_") for k in row)
    assert not any(k.startswith("band_ask_qty_") for k in row)
