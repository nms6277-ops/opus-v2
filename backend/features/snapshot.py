"""Compact feature snapshots written to disk every N ms.

Each snapshot is a single row describing the book + trade state at that
instant. We keep the row small (~100 scalar columns) so a day of 250ms
snapshots across a few symbols still fits on disk after zstd compression.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from backend.collector.lob import OrderBook


@dataclass
class Snapshot:
    """One feature snapshot row."""

    ts_ms: int
    symbol: str

    best_bid: float
    best_ask: float
    best_bid_qty: float
    best_ask_qty: float

    # Aggregate book features
    spread: float  # absolute (quote ccy)
    spread_bp: float  # basis points relative to mid
    mid: float
    microprice: float
    imbalance_top1: float  # (bb_qty - ba_qty) / (bb_qty + ba_qty)
    imbalance_top5: float
    imbalance_top20: float

    # Depth totals
    bid_vol_top5: float
    ask_vol_top5: float
    bid_vol_top20: float
    ask_vol_top20: float

    # Book-slope proxies
    bid_weighted_depth_20: float  # sum of (price-distance * size)
    ask_weighted_depth_20: float

    # Rolling trade features (window = trade_window_ms arg passed to build())
    buy_volume_win: float
    sell_volume_win: float
    trade_count_win: int
    vwap_win: float

    # Top-N price and size arrays (optional, compact)
    bids_top: list[tuple[float, float]]
    asks_top: list[tuple[float, float]]

    # Price-bucket aggregations (tick-size independent):
    # cumulative size within +/-N bp of the mid price. Keys are the bp values.
    bid_bucket_qty: dict[int, float]
    ask_bucket_qty: dict[int, float]

    def to_row(self) -> dict:
        """Flatten into a dict suitable for pyarrow/polars."""
        row = {
            "ts_ms": self.ts_ms,
            "symbol": self.symbol,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "best_bid_qty": self.best_bid_qty,
            "best_ask_qty": self.best_ask_qty,
            "spread": self.spread,
            "spread_bp": self.spread_bp,
            "mid": self.mid,
            "microprice": self.microprice,
            "imbalance_top1": self.imbalance_top1,
            "imbalance_top5": self.imbalance_top5,
            "imbalance_top20": self.imbalance_top20,
            "bid_vol_top5": self.bid_vol_top5,
            "ask_vol_top5": self.ask_vol_top5,
            "bid_vol_top20": self.bid_vol_top20,
            "ask_vol_top20": self.ask_vol_top20,
            "bid_weighted_depth_20": self.bid_weighted_depth_20,
            "ask_weighted_depth_20": self.ask_weighted_depth_20,
            "buy_volume_win": self.buy_volume_win,
            "sell_volume_win": self.sell_volume_win,
            "trade_count_win": self.trade_count_win,
            "vwap_win": self.vwap_win,
        }
        # Flatten book levels as separate columns bid_p_00, bid_q_00, ...
        depth = len(self.bids_top) if self.bids_top else 0
        depth = max(depth, len(self.asks_top) if self.asks_top else 0)
        for i in range(depth):
            if i < len(self.bids_top):
                bp, bq = self.bids_top[i]
            else:
                bp, bq = 0.0, 0.0
            if i < len(self.asks_top):
                ap, aq = self.asks_top[i]
            else:
                ap, aq = 0.0, 0.0
            row[f"bid_p_{i:02d}"] = bp
            row[f"bid_q_{i:02d}"] = bq
            row[f"ask_p_{i:02d}"] = ap
            row[f"ask_q_{i:02d}"] = aq
        # Price-bucket columns: bid_bkt_qty_05bp, ask_bkt_qty_10bp, ...
        for bp_val, q in self.bid_bucket_qty.items():
            row[f"bid_bkt_qty_{bp_val:02d}bp"] = q
        for bp_val, q in self.ask_bucket_qty.items():
            row[f"ask_bkt_qty_{bp_val:02d}bp"] = q
        return row


def build(
    ob: OrderBook,
    *,
    depth: int = 20,
    trade_window_ms: int = 1000,
    now_ms: int | None = None,
    bucket_bps: tuple[int, ...] = (5, 10, 25, 50),
) -> Snapshot | None:
    """Compute a Snapshot from the current order book state."""
    if not ob.ready:
        return None
    bids, asks = ob.top(depth)
    if not bids or not asks:
        return None

    best_bid, best_bid_qty = bids[0]
    best_ask, best_ask_qty = asks[0]
    if best_bid <= 0.0 or best_ask <= 0.0 or best_ask < best_bid:
        return None

    mid = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid
    spread_bp = (spread / mid) * 10_000.0 if mid > 0 else 0.0

    tot_top1 = best_bid_qty + best_ask_qty
    imb_top1 = (best_bid_qty - best_ask_qty) / tot_top1 if tot_top1 > 0 else 0.0

    bid5 = sum(q for _, q in bids[:5])
    ask5 = sum(q for _, q in asks[:5])
    tot5 = bid5 + ask5
    imb5 = (bid5 - ask5) / tot5 if tot5 > 0 else 0.0

    bid20 = sum(q for _, q in bids[:20])
    ask20 = sum(q for _, q in asks[:20])
    tot20 = bid20 + ask20
    imb20 = (bid20 - ask20) / tot20 if tot20 > 0 else 0.0

    # Microprice = qty-weighted mid (classic Gatheral form)
    if tot_top1 > 0:
        microprice = (best_bid * best_ask_qty + best_ask * best_bid_qty) / tot_top1
    else:
        microprice = mid

    # Book-slope proxies: sum over top-20 of (distance * size). Cheap, robust.
    bid_slope = sum(abs(p - best_bid) * q for p, q in bids[:20])
    ask_slope = sum(abs(p - best_ask) * q for p, q in asks[:20])

    # Price-bucket aggregations: cumulative qty within +/-N bp of mid. Use the
    # full in-memory book (not truncated to `depth`) so deeper levels count too.
    bid_bucket_qty: dict[int, float] = {}
    ask_bucket_qty: dict[int, float] = {}
    for bp_val in bucket_bps:
        threshold = mid * bp_val / 10_000.0
        lo = mid - threshold
        hi = mid + threshold
        bid_bucket_qty[bp_val] = sum(q for p, q in ob.bids.items() if p >= lo)
        ask_bucket_qty[bp_val] = sum(q for p, q in ob.asks.items() if p <= hi)

    if now_ms is None:
        now_ms = int(time.time() * 1000)
    window_start = now_ms - trade_window_ms

    buy_vol = 0.0
    sell_vol = 0.0
    notional = 0.0
    n_trades = 0
    for t in reversed(ob.recent_trades):
        if t.ts_ms < window_start:
            break
        if t.is_buyer_maker:
            sell_vol += t.qty  # sell aggressor
        else:
            buy_vol += t.qty
        notional += t.qty * t.price
        n_trades += 1
    tot_qty = buy_vol + sell_vol
    vwap = notional / tot_qty if tot_qty > 0 else 0.0

    return Snapshot(
        ts_ms=now_ms,
        symbol=ob.symbol,
        best_bid=best_bid,
        best_ask=best_ask,
        best_bid_qty=best_bid_qty,
        best_ask_qty=best_ask_qty,
        spread=spread,
        spread_bp=spread_bp,
        mid=mid,
        microprice=microprice,
        imbalance_top1=imb_top1,
        imbalance_top5=imb5,
        imbalance_top20=imb20,
        bid_vol_top5=bid5,
        ask_vol_top5=ask5,
        bid_vol_top20=bid20,
        ask_vol_top20=ask20,
        bid_weighted_depth_20=bid_slope,
        ask_weighted_depth_20=ask_slope,
        buy_volume_win=buy_vol,
        sell_volume_win=sell_vol,
        trade_count_win=n_trades,
        vwap_win=vwap,
        bids_top=bids,
        asks_top=asks,
        bid_bucket_qty=bid_bucket_qty,
        ask_bucket_qty=ask_bucket_qty,
    )
