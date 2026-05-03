"""True VPIN engine.

Implements the Easley-Lopez de Prado-O'Hara (2012) Volume-synchronized
Probability of INformed trading in "volume time":

    VPIN = mean( |V_buy - V_sell| ) / V_bucket        over the last N buckets

Two classification schemes:

* ``TRADE_SIGN`` -- uses the aggressor flag per trade. Canonical. Requires
  trade splitting across buckets when a trade overflows the current bucket
  (no volume may be discarded or "stretched").
* ``BVC`` (Bulk Volume Classification) -- each bucket is classified as a whole
  at close, by the standard normal CDF of its normalized intra-bucket
  log-return::

        V_buy  = V_total * Phi( (P_close - P_open) / sigma )
        V_sell = V_total - V_buy

  where ``sigma`` is the EWMA of *cross-bucket* log-returns
  ``r_i = ln(P_close_i / P_close_{i-1})`` per spec section A3.

In BVC, chunks of a split trade are UNSIGNED; they accumulate ``V_total`` and
update ``P_close``, and classification happens only at bucket close.

Two bucket-fill policies (per Panayides et al. on BVC bar-construction bias):

* ``EXACT_FILL``: split the overflowing trade -- default & canonical for
  TRADE_SIGN.
* ``MINIMUM_FILL``: close on first overflow without splitting -- default for
  BVC (chunk-level splitting under BVC adds boundary complexity without
  reducing the variance of Phi(.)).
"""

from __future__ import annotations

import math
from collections import deque

from .types import (
    BucketFillMode,
    BucketResult,
    BucketSizePolicy,
    ClassificationMode,
    SymbolConfig,
    TradeTick,
)

_EPS = 1e-12
_SQRT_2 = math.sqrt(2.0)


def _phi(x: float) -> float:
    """Standard normal CDF via ``math.erf`` (no scipy dependency).

    Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    """
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


class BVCSigmaTracker:
    """EWMA of squared *cross-bucket* log-returns.

    Update rule (on each bucket close, per spec A3):

        r_i       = ln(P_close_i / P_close_{i-1})
        ewma_r2_i = (1 - lambda) * ewma_r2_{i-1} + lambda * r_i^2
        sigma     = sqrt(ewma_r2 + eps)

    The estimator is centered (mean assumed ~0 at bucket scale); this matches
    the standard BVC practice.
    """

    __slots__ = ("_lambda", "_min_buckets", "_ewma_r2", "_prev_close", "_returns_count")

    def __init__(self, lambda_: float, min_buckets: int) -> None:
        if not (0.0 < lambda_ < 1.0):
            raise ValueError("bvc_lambda must be in (0, 1)")
        if min_buckets < 1:
            raise ValueError("min_buckets_for_bvc_sigma must be >= 1")
        self._lambda = lambda_
        self._min_buckets = min_buckets
        self._ewma_r2 = 0.0
        self._prev_close: float | None = None
        self._returns_count = 0  # number of r_i observations accumulated

    def observe_close(self, p_close: float) -> None:
        """Record a bucket close price.

        The first call merely stores the previous close; subsequent calls
        produce a log-return ``r_i`` that updates the EWMA.
        """
        if p_close <= 0.0:
            return  # guard against log(<=0); skip this close
        if self._prev_close is None:
            self._prev_close = p_close
            return
        r = math.log(p_close / self._prev_close)
        self._prev_close = p_close
        if self._returns_count == 0:
            # Seed with first squared return (avoids initial bias toward 0).
            self._ewma_r2 = r * r
        else:
            self._ewma_r2 = (1.0 - self._lambda) * self._ewma_r2 + self._lambda * r * r
        self._returns_count += 1

    @property
    def sigma(self) -> float:
        return math.sqrt(self._ewma_r2 + _EPS * _EPS)

    @property
    def ready(self) -> bool:
        return self._returns_count >= self._min_buckets


class TrueVPINEngine:
    """Feeds trades into volume buckets and exposes the rolling VPIN value."""

    __slots__ = (
        "_cfg",
        "_bucket_size",
        "_curr",
        "_closed",
        "_bvc_sigma",
        "_rolling_vol",
        "_last_recompute_ts",
        "_total_volume_observed",
    )

    def __init__(self, config: SymbolConfig) -> None:
        self._cfg = config
        self._bucket_size: float = max(config.min_bucket_size, config.fixed_bucket_size)
        self._curr: BucketResult | None = None
        self._closed: deque[BucketResult] = deque(maxlen=config.vpin_window)
        self._bvc_sigma = BVCSigmaTracker(config.bvc_lambda, config.min_buckets_for_bvc_sigma)
        # Rolling volume tracking (24h by default; configurable per symbol).
        self._rolling_vol: deque[tuple[float, float]] = deque()
        self._last_recompute_ts: float = 0.0
        self._total_volume_observed: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_trade(self, tick: TradeTick) -> None:
        """Ingest a trade; drives the bucket state machine."""
        if tick.quantity <= 0.0 or tick.price <= 0.0:
            return  # reject non-positive trades defensively

        self._maybe_update_bucket_size(tick.timestamp, tick.quantity)

        mode = self._cfg.classification_mode
        fill = self._cfg.bucket_fill_mode
        if mode is ClassificationMode.TRADE_SIGN:
            if fill is BucketFillMode.EXACT_FILL:
                self._trade_sign_exact_fill(tick)
            else:
                self._trade_sign_minimum_fill(tick)
        else:
            if fill is BucketFillMode.EXACT_FILL:
                self._bvc_exact_fill(tick)
            else:
                self._bvc_minimum_fill(tick)

    @property
    def vpin(self) -> float:
        """Rolling VPIN over the last ``vpin_window`` *classified* buckets.

        Returns ``0.0`` until the window has at least one classified bucket.
        """
        if not self._closed or self._bucket_size <= 0.0:
            return 0.0
        total_abs = 0.0
        for b in self._closed:
            total_abs += abs(b.v_buy - b.v_sell)
        return total_abs / (len(self._closed) * self._bucket_size)

    @property
    def buckets_filled(self) -> int:
        """Number of classified buckets currently in the VPIN window."""
        return len(self._closed)

    @property
    def bucket_size(self) -> float:
        return self._bucket_size

    @property
    def bvc_ready(self) -> bool:
        """True when BVC sigma has enough history (always True in TRADE_SIGN)."""
        if self._cfg.classification_mode is not ClassificationMode.BVC:
            return True
        return self._bvc_sigma.ready

    # ------------------------------------------------------------------
    # TRADE_SIGN mode
    # ------------------------------------------------------------------

    def _trade_sign_exact_fill(self, tick: TradeTick) -> None:
        """Classic VPIN behavior: split the trade across buckets by volume."""
        # Binance convention: is_buyer_maker=True  =>  seller is aggressor.
        #                     is_buyer_maker=False =>  buyer  is aggressor.
        buyer_aggressor = not tick.is_buyer_maker
        remaining = tick.quantity
        price = tick.price
        while remaining > 0.0:
            if self._curr is None:
                self._curr = BucketResult(
                    v_buy=0.0,
                    v_sell=0.0,
                    v_total=0.0,
                    p_open=price,
                    p_close=price,
                )
            capacity = self._bucket_size - self._curr.v_total
            if capacity <= _EPS:
                self._close_bucket_trade_sign()
                continue
            chunk = min(remaining, capacity)
            if buyer_aggressor:
                self._curr.v_buy += chunk
            else:
                self._curr.v_sell += chunk
            self._curr.v_total += chunk
            self._curr.p_close = price
            remaining -= chunk
            if self._curr.v_total >= self._bucket_size - _EPS:
                self._close_bucket_trade_sign()

    def _trade_sign_minimum_fill(self, tick: TradeTick) -> None:
        """Non-canonical variant: whole trade goes into one bucket, no split."""
        buyer_aggressor = not tick.is_buyer_maker
        if self._curr is None:
            self._curr = BucketResult(
                v_buy=0.0,
                v_sell=0.0,
                v_total=0.0,
                p_open=tick.price,
                p_close=tick.price,
            )
        if buyer_aggressor:
            self._curr.v_buy += tick.quantity
        else:
            self._curr.v_sell += tick.quantity
        self._curr.v_total += tick.quantity
        self._curr.p_close = tick.price
        if self._curr.v_total >= self._bucket_size:
            self._close_bucket_trade_sign()

    def _close_bucket_trade_sign(self) -> None:
        """Finalize a TRADE_SIGN bucket (signs already accumulated)."""
        if self._curr is None:
            return
        b = self._curr
        self._closed.append(b)
        # Still feed the BVC sigma tracker; it is cheap and allows the user to
        # switch modes at inspection time. This has no effect in TRADE_SIGN.
        self._bvc_sigma.observe_close(b.p_close)
        self._curr = None

    # ------------------------------------------------------------------
    # BVC mode
    # ------------------------------------------------------------------

    def _bvc_exact_fill(self, tick: TradeTick) -> None:
        """BVC with chunk splitting. Chunks are *unsigned*.

        Only ``V_total`` and ``P_close`` are updated per chunk; the buy/sell
        split is computed at close via Phi((P_close - P_open) / sigma).
        """
        remaining = tick.quantity
        price = tick.price
        while remaining > 0.0:
            if self._curr is None:
                self._curr = BucketResult(
                    v_buy=0.0,
                    v_sell=0.0,
                    v_total=0.0,
                    p_open=price,
                    p_close=price,
                )
            capacity = self._bucket_size - self._curr.v_total
            if capacity <= _EPS:
                self._close_bucket_bvc()
                continue
            chunk = min(remaining, capacity)
            self._curr.v_total += chunk
            self._curr.p_close = price
            remaining -= chunk
            if self._curr.v_total >= self._bucket_size - _EPS:
                self._close_bucket_bvc()

    def _bvc_minimum_fill(self, tick: TradeTick) -> None:
        """Default BVC: whole trade goes into bucket; close on first overflow."""
        if self._curr is None:
            self._curr = BucketResult(
                v_buy=0.0,
                v_sell=0.0,
                v_total=0.0,
                p_open=tick.price,
                p_close=tick.price,
            )
        self._curr.v_total += tick.quantity
        self._curr.p_close = tick.price
        if self._curr.v_total >= self._bucket_size:
            self._close_bucket_bvc()

    def _close_bucket_bvc(self) -> None:
        """Classify a BVC bucket at close.

        If BVC sigma is not yet warmed up (fewer than ``min_buckets_for_bvc_sigma``
        cross-bucket returns accumulated), the bucket is *not* appended to the
        VPIN window -- but its close price is still fed to the sigma tracker so
        that warmup can progress. This satisfies spec A3: "Before that, buckets
        accumulate but VPIN is not emitted (is_ready = False)."
        """
        if self._curr is None:
            return
        b = self._curr
        ready = self._bvc_sigma.ready
        sigma = self._bvc_sigma.sigma
        if ready and b.p_open > 0.0 and b.p_close > 0.0 and sigma > _EPS:
            # Intra-bucket log-return for the BVC classification argument.
            r = math.log(b.p_close / b.p_open)
            phi = _phi(r / sigma)
            b.v_buy = b.v_total * phi
            b.v_sell = b.v_total - b.v_buy
            self._closed.append(b)
        # Always update the sigma tracker with the bucket close (even during warmup).
        self._bvc_sigma.observe_close(b.p_close)
        self._curr = None

    # ------------------------------------------------------------------
    # Bucket sizing (FIXED / ROLLING_DAILY)
    # ------------------------------------------------------------------

    def _maybe_update_bucket_size(self, ts: float, qty: float) -> None:
        if self._cfg.bucket_policy is not BucketSizePolicy.ROLLING_DAILY:
            return
        # Record this trade's volume for the configured rolling window.
        self._rolling_vol.append((ts, qty))
        window_s = max(1.0, float(self._cfg.rolling_volume_window_s))
        cutoff = ts - window_s
        while self._rolling_vol and self._rolling_vol[0][0] < cutoff:
            self._rolling_vol.popleft()
        # Debounced recompute.
        if ts - self._last_recompute_ts < self._cfg.bucket_size_recompute_interval_s:
            return
        total_vol = 0.0
        for _, v in self._rolling_vol:
            total_vol += v
        target_buckets = (
            self._cfg.target_buckets_per_window
            if self._cfg.target_buckets_per_window is not None
            else self._cfg.target_buckets_per_day
        )
        if total_vol > 0.0 and target_buckets > 0:
            new_size = total_vol / float(target_buckets)
            self._bucket_size = max(self._cfg.min_bucket_size, new_size)
        self._last_recompute_ts = ts
