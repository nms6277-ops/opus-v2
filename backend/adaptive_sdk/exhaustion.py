"""Asymmetric exhaustion detector.

Two signals:

* ``SELL_EXHAUSTION``: aggressive-sell volume Z-score is very high but price
  has NOT made a fresh rolling low (flow burned out without pushing price).
* ``BUY_EXHAUSTION``: symmetric -- aggressive-buy Z-score high, no new high.

Design decisions (per spec sections A8, A9):

* **Flow Z-score is per-second, not per-tick.** Tick density varies by orders
  of magnitude across symbols and regimes; normalizing per wall-clock window
  (``flow_window_ms``, default 1000 ms) makes Z-scores comparable across
  BTCUSDT (thousands of trades/s) and thin L2 pairs (units of trades/s).
* **Local extrema use a time window (``price_extrema_window_ms``), not a
  tick count.** Same rationale.
* **EWMA / EWVar via West (1979)** -- single-pass, no allocation.
"""

from __future__ import annotations

import math
from collections import deque

from .types import SymbolConfig, TradeTick

_EPS = 1e-12
_VAR_EPS = 1e-24


class EWMAEWVar:
    """Exponentially-weighted mean and variance (West 1979).

    Update::

        delta   = x - mean
        mean   <- mean + lambda * delta
        var    <- (1 - lambda) * (var + lambda * delta^2)

    The first observation seeds ``mean = x, var = 0``; variance becomes
    meaningful only after multiple updates.
    """

    __slots__ = ("_lambda", "_mean", "_var", "_n")

    def __init__(self, lambda_: float) -> None:
        if not (0.0 < lambda_ < 1.0):
            raise ValueError("lambda must be in (0, 1)")
        self._lambda = lambda_
        self._mean = 0.0
        self._var = 0.0
        self._n = 0

    def update(self, x: float) -> None:
        if self._n == 0:
            self._mean = x
            self._var = 0.0
            self._n = 1
            return
        delta = x - self._mean
        self._mean = self._mean + self._lambda * delta
        self._var = (1.0 - self._lambda) * (self._var + self._lambda * delta * delta)
        self._n += 1

    def z_score(self, x: float) -> float:
        """Z-score of ``x`` against the current posterior (mean, var)."""
        if self._n < 2 or self._var <= _VAR_EPS:
            return 0.0
        return (x - self._mean) / math.sqrt(self._var)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def var(self) -> float:
        return self._var

    @property
    def count(self) -> int:
        return self._n


class RollingExtrema:
    """Time-windowed min/max with O(1) amortized queries.

    Uses the monotonic-deque trick (Knuth / sliding-window max): we maintain
    two deques of ``(ts, price)`` pairs, one non-decreasing (for min), one
    non-increasing (for max). Old entries are evicted from the front by
    timestamp; incoming entries pop from the back while violating monotonicity.
    """

    __slots__ = ("_window_s", "_min_dq", "_max_dq", "_count")

    def __init__(self, window_ms: int) -> None:
        if window_ms <= 0:
            raise ValueError("price_extrema_window_ms must be > 0")
        self._window_s = window_ms / 1000.0
        self._min_dq: deque[tuple[float, float]] = deque()
        self._max_dq: deque[tuple[float, float]] = deque()
        self._count = 0

    def update(self, ts: float, price: float) -> None:
        cutoff = ts - self._window_s
        # Evict expired entries from the front.
        while self._min_dq and self._min_dq[0][0] < cutoff:
            self._min_dq.popleft()
        while self._max_dq and self._max_dq[0][0] < cutoff:
            self._max_dq.popleft()
        # Maintain monotonicity from the back.
        while self._min_dq and self._min_dq[-1][1] >= price:
            self._min_dq.pop()
        self._min_dq.append((ts, price))
        while self._max_dq and self._max_dq[-1][1] <= price:
            self._max_dq.pop()
        self._max_dq.append((ts, price))
        self._count += 1

    @property
    def local_min(self) -> float:
        return self._min_dq[0][1] if self._min_dq else 0.0

    @property
    def local_max(self) -> float:
        return self._max_dq[0][1] if self._max_dq else 0.0

    @property
    def count(self) -> int:
        return self._count


class RollingRealizedVolatility:
    """Rolling realized volatility from high-frequency log returns."""

    __slots__ = ("_window_s", "_returns", "_prev_price", "_sum_sq")

    def __init__(self, window_ms: int) -> None:
        if window_ms <= 0:
            raise ValueError("realized_vol_window_ms must be > 0")
        self._window_s = window_ms / 1000.0
        self._returns: deque[tuple[float, float]] = deque()
        self._prev_price: float | None = None
        self._sum_sq = 0.0

    def update(self, ts: float, price: float) -> None:
        self._evict(ts)
        if price <= 0.0:
            return
        if self._prev_price is not None and self._prev_price > 0.0:
            ret = math.log(price / self._prev_price)
            ret_sq = ret * ret
            self._returns.append((ts, ret_sq))
            self._sum_sq += ret_sq
        self._prev_price = price
        self._evict(ts)

    def _evict(self, ts: float) -> None:
        cutoff = ts - self._window_s
        while self._returns and self._returns[0][0] < cutoff:
            _, ret_sq = self._returns.popleft()
            self._sum_sq -= ret_sq
        if self._sum_sq < 0.0 and self._sum_sq > -_EPS:
            self._sum_sq = 0.0

    @property
    def value(self) -> float:
        return math.sqrt(max(0.0, self._sum_sq))


class ExhaustionDetector:
    """Per-symbol exhaustion detector (EWMA flow Z-score + time-windowed extrema).

    Flow semantics: within each ``flow_window_ms`` wall-clock window we
    accumulate ``buy_vol`` and ``sell_vol`` (in aggressor-coin units). At
    window rollover, the accumulated totals are committed to
    ``EWMAEWVar`` and the accumulators reset. The instantaneous
    ``z_*_flow`` is computed against the current (possibly partial) window's
    running total -- i.e. a mid-window burst spikes Z before the window closes.
    """

    __slots__ = (
        "_cfg",
        "_extrema",
        "_realized_vol",
        "_flow_window_s",
        "_window_start",
        "_curr_buy",
        "_curr_sell",
        "_buy_stats",
        "_sell_stats",
        "_tick_count",
    )

    def __init__(self, config: SymbolConfig) -> None:
        self._cfg = config
        self._extrema = RollingExtrema(config.price_extrema_window_ms)
        self._realized_vol = RollingRealizedVolatility(config.realized_vol_window_ms)
        self._flow_window_s = config.flow_window_ms / 1000.0
        if self._flow_window_s <= 0.0:
            raise ValueError("flow_window_ms must be > 0")
        self._window_start: float | None = None
        self._curr_buy = 0.0
        self._curr_sell = 0.0
        self._buy_stats = EWMAEWVar(config.flow_lambda)
        self._sell_stats = EWMAEWVar(config.flow_lambda)
        self._tick_count = 0

    def on_trade(self, tick: TradeTick) -> None:
        """Ingest a trade: updates extrema, flow accumulators, and rolls windows."""
        self._tick_count += 1
        self._extrema.update(tick.timestamp, tick.price)
        self._realized_vol.update(tick.timestamp, tick.price)

        if self._window_start is None:
            self._window_start = tick.timestamp

        # Roll forward as many whole windows as have elapsed. On each rollover
        # commit accumulated volumes (could be zero for quiet seconds) and reset.
        while tick.timestamp - self._window_start >= self._flow_window_s:
            self._buy_stats.update(self._curr_buy)
            self._sell_stats.update(self._curr_sell)
            self._curr_buy = 0.0
            self._curr_sell = 0.0
            self._window_start += self._flow_window_s

        # Accumulate this trade into the current window.
        if tick.is_buyer_maker:
            # Seller is aggressor.
            self._curr_sell += tick.quantity
        else:
            self._curr_buy += tick.quantity

    @property
    def z_buy_flow(self) -> float:
        return self._buy_stats.z_score(self._curr_buy)

    @property
    def z_sell_flow(self) -> float:
        return self._sell_stats.z_score(self._curr_sell)

    @property
    def local_high(self) -> float:
        return self._extrema.local_max

    @property
    def local_low(self) -> float:
        return self._extrema.local_min

    @property
    def realized_vol(self) -> float:
        return self._realized_vol.value

    @property
    def tick_count(self) -> int:
        return self._tick_count
