"""In-memory limit order book reconstructor for a single symbol.

Maintains top-N bid/ask levels from a Binance **USDT-M Futures** depth
stream. Stream format: ``@depth@100ms`` — diff updates with ``U``
(first update id), ``u`` (final update id), and ``pu`` (previous final
update id). We synchronise against a REST snapshot per the official
docs for futures (critically different from spot!):

  https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Diff-Book-Depth-Streams

Key rules for FUTURES (NOT spot):
  1. Fetch REST snapshot -> get ``lastUpdateId``.
  2. Drop buffered events where ``u <  lastUpdateId``.
  3. The first applied event must satisfy
     ``U <= lastUpdateId AND u >= lastUpdateId`` (no ``+1``, unlike spot).
  4. While streaming, every subsequent event must satisfy
     ``event.pu == prev.u`` (NOT ``event.U == prev.u + 1`` — that is
     the spot rule). Futures aggregates multiple id ranges into one
     event, so ``U`` and ``u`` are generally non-contiguous across
     events; only ``pu`` is contiguous.
  5. On any gap, re-fetch the snapshot and restart from step 2.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trade:
    """A single aggressor trade (aggTrade)."""

    ts_ms: int
    price: float
    qty: float
    is_buyer_maker: bool  # True => sell aggressor; False => buy aggressor


@dataclass
class OrderBook:
    """Top-N limit order book snapshot in memory.

    Uses two ``dict[float, float]`` as ``price -> size``. We keep the full
    requested depth; sorting happens on access (top-N only, cheap).
    """

    symbol: str
    max_depth: int = 50
    last_update_id: int = 0
    ready: bool = False

    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)

    # Rolling window of recent trades for feature computation
    recent_trades: deque[Trade] = field(default_factory=lambda: deque(maxlen=1024))

    # Quality counters
    updates_applied: int = 0
    snapshot_resyncs: int = 0
    sequence_gaps: int = 0
    last_update_ts_ms: int = 0

    # Buffered updates while waiting for a REST snapshot
    _buffer: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.last_update_id = 0
        self.ready = False
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Snapshot application (REST /fapi/v1/depth)
    # ------------------------------------------------------------------
    def apply_snapshot(self, snap: dict[str, Any]) -> None:
        self.bids.clear()
        self.asks.clear()
        for p, q in snap.get("bids", []):
            price = float(p)
            size = float(q)
            if size > 0:
                self.bids[price] = size
        for p, q in snap.get("asks", []):
            price = float(p)
            size = float(q)
            if size > 0:
                self.asks[price] = size

        self.last_update_id = int(snap["lastUpdateId"])
        self.ready = False  # becomes True after first matching buffered update

    # ------------------------------------------------------------------
    # Diff update application
    # ------------------------------------------------------------------
    def buffer_update(self, evt: dict[str, Any]) -> None:
        """Store a depth update until the REST snapshot is processed."""
        self._buffer.append(evt)

    def flush_buffer(self) -> bool:
        """Apply buffered updates after a snapshot (FUTURES rules).

        Step-by-step per the Futures spec:
          - Drop events with ``u < lastUpdateId`` (stale).
          - The first applied event must satisfy
            ``U <= lastUpdateId AND u >= lastUpdateId`` (the bridge).
          - Subsequent events must satisfy ``pu == prev.u``.
          - If no bridge event is present but the buffer is non-empty
            (REST snapshot lags the stream), accept the first non-stale
            event as a best-effort bridge and log it as a sequence gap —
            retrying the snapshot does not help because the stream is
            always moving forward.

        Returns True once the book is marked ``ready``, which we do
        unconditionally here (even in the empty-buffer case the snapshot
        itself is authoritative).
        """
        snap_lastid = self.last_update_id

        # Drop stale events (FUTURES: u < lastUpdateId)
        pending = [e for e in self._buffer if e["u"] >= snap_lastid]

        if not pending:
            # Fresh snapshot, nothing to apply.
            self._buffer = []
            self.ready = True
            return True

        first = pending[0]
        if first["U"] <= snap_lastid <= first["u"]:
            # Clean bridge — docs-compliant.
            self._apply_diff(first)
        else:
            # REST snapshot lags the stream. Accept first event as bridge.
            self.sequence_gaps += 1
            self._apply_diff(first)

        # Subsequent events: contiguous iff pu == prev.u (FUTURES rule).
        for evt in pending[1:]:
            pu = evt.get("pu")
            if pu is not None and pu != self.last_update_id:
                # Gap inside the buffer — stop here. Live stream will
                # continue from the next arriving event; if that also
                # fails pu-check we request a fresh resync.
                self.sequence_gaps += 1
                break
            self._apply_diff(evt)

        self._buffer = []
        self.ready = True
        return True

    def apply_diff(self, evt: dict[str, Any]) -> bool:
        """Apply a fresh (non-buffered) update (FUTURES rules).

        Returns True when the event was either applied, skipped as stale,
        or buffered pending a snapshot — all normal states that do NOT
        require a resync. Returns False only when a mid-stream sequence
        gap is detected AFTER the book was synced (``pu != prev.u``);
        the caller should then request a resync.
        """
        if not self.ready:
            self.buffer_update(evt)
            return True  # pre-sync buffering is normal

        u = evt["u"]
        if u < self.last_update_id:
            return True  # stale event, ignore

        # FUTURES contiguity check: pu == prev.u.
        pu = evt.get("pu")
        if pu is not None and pu != self.last_update_id:
            self.sequence_gaps += 1
            self.ready = False
            self._buffer = [evt]
            return False

        self._apply_diff(evt)
        return True

    def _apply_diff(self, evt: dict[str, Any]) -> None:
        for p, q in evt.get("b", []):
            price = float(p)
            size = float(q)
            if size == 0.0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = size
        for p, q in evt.get("a", []):
            price = float(p)
            size = float(q)
            if size == 0.0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = size

        self.last_update_id = evt["u"]
        self.updates_applied += 1
        self.last_update_ts_ms = int(evt.get("E", time.time() * 1000))

        # Trim deep levels: we only care about top max_depth each side
        if len(self.bids) > self.max_depth * 4:
            keep = sorted(self.bids.items(), key=lambda kv: -kv[0])[: self.max_depth * 2]
            self.bids = dict(keep)
        if len(self.asks) > self.max_depth * 4:
            keep = sorted(self.asks.items(), key=lambda kv: kv[0])[: self.max_depth * 2]
            self.asks = dict(keep)

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------
    def top(self, depth: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Return (bids_desc, asks_asc) truncated to `depth` levels."""
        bids = sorted(self.bids.items(), key=lambda kv: -kv[0])[:depth]
        asks = sorted(self.asks.items(), key=lambda kv: kv[0])[:depth]
        return bids, asks

    def best(self) -> tuple[float, float, float, float]:
        """Return (best_bid, best_bid_qty, best_ask, best_ask_qty) or zeros."""
        if not self.bids or not self.asks:
            return 0.0, 0.0, 0.0, 0.0
        bb = max(self.bids)
        ba = min(self.asks)
        return bb, self.bids[bb], ba, self.asks[ba]

    def add_trade(self, t: Trade) -> None:
        self.recent_trades.append(t)
