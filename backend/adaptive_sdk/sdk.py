"""AdaptiveAnalyticsSDK orchestrator.

Owns the per-symbol ``SymbolContext`` dictionary and wires the three
components (:class:`TrueVPINEngine`, :class:`ExhaustionDetector`,
:class:`ThompsonSamplingMAB`) into a single ingress API.

Signal lifecycle
----------------
1. ``on_trade(tick)`` updates VPIN and exhaustion metrics.
2. Once ``is_ready`` and not toxic, the MAB's ``select_arm`` picks a
   Z-threshold. Triggers are checked at that threshold.
3. If triggered, a :class:`Signal` is built with a fresh ``uuid4().hex`` and
   registered in the ``pending_signals`` dict.
4. ``report_outcome(symbol, outcome)`` resolves the pending entry, computes
   ``reward``, updates the arm, and marks the entry resolved (idempotent).

Toxicity gate
-------------
If ``vpin >= vpin_high`` no signal is emitted (market too informed to trade).

Confidence (per spec A11)::

    base       = sigmoid( (|z_relevant| - z_threshold_arm) / z_scale )
    toxicity   = clip( (vpin_high - vpin) / (vpin_high - vpin_mid), 0, 1 )
    obi_align  = clip( 0.5 + 0.5 * sign(direction) * obi, 0.5, 1.0 )
    confidence = base * toxicity * obi_align           in [0, 1]

``obi_align = 1.0`` when no book data has ever been observed for the symbol
(API does not degrade in trade-only setups).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from uuid import uuid4

from .exhaustion import ExhaustionDetector
from .mab import ThompsonSamplingMAB
from .types import (
    BookSnapshot,
    ClassificationMode,
    ExhaustionType,
    GlobalConfig,
    MetricsSnapshot,
    Outcome,
    PendingSignal,
    Signal,
    SymbolConfig,
    SymbolState,
    TradeTick,
)
from .vpin import TrueVPINEngine

_logger = logging.getLogger(__name__)

# Relative tolerance used when testing "price did NOT break the rolling
# low/high" during exhaustion. A trade at exactly the rolling extreme is
# considered to have broken it (we require strict progress to reject a setup).
_EXTREMA_GUARD = 1e-4


class SymbolContext:
    """Fully isolated per-symbol state (VPIN + exhaustion + bandit + OBI)."""

    __slots__ = (
        "symbol",
        "config",
        "vpin_engine",
        "exhaustion",
        "mab",
        "obi",
        "has_book_data",
        "zone",
        "tick_count",
    )

    def __init__(
        self,
        symbol: str,
        config: SymbolConfig,
        seed: int | None = None,
    ) -> None:
        self.symbol = symbol
        self.config = config
        self.vpin_engine = TrueVPINEngine(config)
        self.exhaustion = ExhaustionDetector(config)
        self.mab = ThompsonSamplingMAB(n_arms=3, seed=seed)
        self.obi: float = 0.0
        self.has_book_data: bool = False
        self.zone: tuple[float, float] | None = None
        self.tick_count: int = 0

    def is_ready(self) -> bool:
        """Warmup gate (per spec A13): all three conditions must hold."""
        if self.vpin_engine.buckets_filled < self.config.min_buckets_for_vpin:
            return False
        if self.tick_count < self.config.min_ticks_for_z:
            return False
        if self.config.classification_mode is ClassificationMode.BVC:
            if not self.vpin_engine.bvc_ready:
                return False
        return True


class AdaptiveAnalyticsSDK:
    """Public facade. All methods are synchronous and thread-unsafe by design.

    The SDK assumes a single-thread ingress (one event loop / one worker);
    consumers that need multi-threaded access should guard with their own lock.
    """

    __slots__ = (
        "_config",
        "_contexts",
        "_pending",
        "_pending_order",
        "_cleanup_tick_counter",
    )

    def __init__(self, config: GlobalConfig) -> None:
        self._config = config
        self._contexts: dict[str, SymbolContext] = {}
        self._pending: dict[str, PendingSignal] = {}
        # Insertion order for FIFO eviction when pending_signals_max is reached.
        self._pending_order: deque[str] = deque()
        self._cleanup_tick_counter: int = 0

    # ------------------------------------------------------------------
    # Symbol lifecycle
    # ------------------------------------------------------------------

    def register_symbol(
        self,
        symbol: str,
        custom_config: SymbolConfig | None = None,
    ) -> None:
        """Create an isolated context for ``symbol``. Idempotent."""
        if symbol in self._contexts:
            return
        cfg = custom_config if custom_config is not None else self._config.default_symbol_config
        self._contexts[symbol] = SymbolContext(symbol, cfg)

    # ------------------------------------------------------------------
    # Zone-of-interest filter
    # ------------------------------------------------------------------

    def set_zone_of_interest(
        self,
        symbol: str,
        upper_bound: float,
        lower_bound: float,
    ) -> None:
        """Only emit signals while ``lower_bound <= price <= upper_bound``."""
        ctx = self._get_ctx(symbol)
        if lower_bound > upper_bound:
            raise ValueError("lower_bound must be <= upper_bound")
        ctx.zone = (lower_bound, upper_bound)

    def clear_zone_of_interest(self, symbol: str) -> None:
        ctx = self._get_ctx(symbol)
        ctx.zone = None

    # ------------------------------------------------------------------
    # Ingress: trades and book snapshots
    # ------------------------------------------------------------------

    def on_trade(self, tick: TradeTick) -> Signal | None:
        ctx = self._get_ctx(tick.symbol)
        ctx.tick_count += 1
        ctx.vpin_engine.on_trade(tick)
        ctx.exhaustion.on_trade(tick)

        # Periodic lazy cleanup of expired pending signals.
        self._cleanup_tick_counter += 1
        if self._cleanup_tick_counter >= ctx.config.pending_cleanup_interval_ticks:
            self._cleanup_tick_counter = 0
            self._cleanup_pending(tick.timestamp, ctx.config.signal_ttl_seconds)

        if not ctx.is_ready():
            return None

        vpin = ctx.vpin_engine.vpin
        if vpin >= ctx.config.vpin_high:
            return None  # toxicity gate: market too informed

        if ctx.zone is not None:
            low, high = ctx.zone
            if not (low <= tick.price <= high):
                return None

        arm_id = ctx.mab.select_arm()
        z_threshold = ctx.config.z_thresholds[arm_id]

        z_buy = ctx.exhaustion.z_buy_flow
        z_sell = ctx.exhaustion.z_sell_flow
        local_high = ctx.exhaustion.local_high
        local_low = ctx.exhaustion.local_low
        realized_vol = ctx.exhaustion.realized_vol

        # "price did NOT break the rolling extreme" -- require strict margin.
        sell_trigger = (
            z_sell > z_threshold
            and local_low > 0.0
            and tick.price > local_low * (1.0 + _EXTREMA_GUARD)
            and _has_required_price_excursion(
                cfg=ctx.config,
                exhaustion_type=ExhaustionType.SELL_EXHAUSTION,
                price=tick.price,
                local_high=local_high,
                local_low=local_low,
                realized_vol=realized_vol,
            )
        )
        buy_trigger = (
            z_buy > z_threshold
            and local_high > 0.0
            and tick.price < local_high * (1.0 - _EXTREMA_GUARD)
            and _has_required_price_excursion(
                cfg=ctx.config,
                exhaustion_type=ExhaustionType.BUY_EXHAUSTION,
                price=tick.price,
                local_high=local_high,
                local_low=local_low,
                realized_vol=realized_vol,
            )
        )

        if sell_trigger and buy_trigger:
            if abs(z_sell) >= abs(z_buy):
                exhaustion_type: ExhaustionType | None = ExhaustionType.SELL_EXHAUSTION
                z_relevant = z_sell
            else:
                exhaustion_type = ExhaustionType.BUY_EXHAUSTION
                z_relevant = z_buy
        elif sell_trigger:
            exhaustion_type = ExhaustionType.SELL_EXHAUSTION
            z_relevant = z_sell
        elif buy_trigger:
            exhaustion_type = ExhaustionType.BUY_EXHAUSTION
            z_relevant = z_buy
        else:
            return None

        confidence = self._compute_confidence(
            z_relevant=z_relevant,
            z_threshold=z_threshold,
            cfg=ctx.config,
            vpin=vpin,
            obi=ctx.obi,
            has_book=ctx.has_book_data,
            exhaustion_type=exhaustion_type,
        )

        metrics = MetricsSnapshot(
            vpin=vpin,
            z_score_buy_flow=z_buy,
            z_score_sell_flow=z_sell,
            obi=ctx.obi if ctx.has_book_data else 0.0,
            bucket_size=ctx.vpin_engine.bucket_size,
            buckets_filled=ctx.vpin_engine.buckets_filled,
            realized_vol=realized_vol,
        )

        signal_id = uuid4().hex
        signal = Signal(
            signal_id=signal_id,
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            exhaustion_type=exhaustion_type,
            confidence=confidence,
            arm_id=arm_id,
            metrics=metrics,
        )

        self._register_pending(
            signal_id=signal_id,
            symbol=tick.symbol,
            arm_id=arm_id,
            ts=tick.timestamp,
            max_size=ctx.config.pending_signals_max,
        )
        return signal

    def on_book_update(self, snapshot: BookSnapshot) -> None:
        ctx = self._get_ctx(snapshot.symbol)
        denom = snapshot.bid_vol + snapshot.ask_vol
        if denom <= 0.0:
            # Degenerate snapshot (empty top-of-book); leave OBI untouched.
            return
        ctx.obi = (snapshot.bid_vol - snapshot.ask_vol) / denom
        ctx.has_book_data = True

    # ------------------------------------------------------------------
    # Polling & feedback
    # ------------------------------------------------------------------

    def get_state(self, symbol: str) -> SymbolState:
        ctx = self._get_ctx(symbol)
        pending_count = 0
        for p in self._pending.values():
            if p.symbol == symbol and not p.is_resolved:
                pending_count += 1
        return SymbolState(
            is_ready=ctx.is_ready(),
            vpin=ctx.vpin_engine.vpin,
            sell_exhaustion_z=ctx.exhaustion.z_sell_flow,
            buy_exhaustion_z=ctx.exhaustion.z_buy_flow,
            pending_signals_count=pending_count,
            buckets_filled=ctx.vpin_engine.buckets_filled,
            realized_vol=ctx.exhaustion.realized_vol,
        )

    def report_outcome(self, symbol: str, outcome: Outcome) -> bool:
        """Feed a post-trade outcome back into the MAB. Idempotent.

        Returns ``False`` (no update applied) if:

        * the signal id is unknown or has already been evicted / expired
          (lazy cleanup removes expired entries on the next ``on_trade``),
        * the entry has already been resolved (second call with same id),
        * the supplied ``symbol`` does not match the original signal.
        """
        pending = self._pending.get(outcome.signal_id)
        if pending is None or pending.is_resolved:
            return False
        if pending.symbol != symbol:
            return False
        ctx = self._contexts.get(symbol)
        if ctx is None:
            return False

        if ctx.config.reward_transform is not None:
            reward = float(ctx.config.reward_transform(outcome))
        else:
            reward = outcome.pnl - outcome.fees - outcome.slippage

        ctx.mab.update(pending.arm_id, reward)
        pending.is_resolved = True
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_ctx(self, symbol: str) -> SymbolContext:
        ctx = self._contexts.get(symbol)
        if ctx is None:
            raise ValueError(f"Symbol '{symbol}' not registered; call register_symbol() first.")
        return ctx

    def _register_pending(
        self,
        signal_id: str,
        symbol: str,
        arm_id: int,
        ts: float,
        max_size: int,
    ) -> None:
        # FIFO eviction of the oldest entries while at capacity.
        while len(self._pending) >= max_size:
            if not self._pending_order:
                break
            oldest = self._pending_order.popleft()
            self._pending.pop(oldest, None)
        self._pending[signal_id] = PendingSignal(symbol=symbol, arm_id=arm_id, timestamp=ts)
        self._pending_order.append(signal_id)

    def _cleanup_pending(self, now: float, ttl: float) -> None:
        cutoff = now - ttl
        while self._pending_order:
            oldest = self._pending_order[0]
            entry = self._pending.get(oldest)
            if entry is None:
                self._pending_order.popleft()
                continue
            if entry.timestamp < cutoff:
                self._pending.pop(oldest, None)
                self._pending_order.popleft()
            else:
                break

    # ------------------------------------------------------------------
    # Confidence formula (spec A11)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(
        z_relevant: float,
        z_threshold: float,
        cfg: SymbolConfig,
        vpin: float,
        obi: float,
        has_book: bool,
        exhaustion_type: ExhaustionType,
    ) -> float:
        # base: sigmoid of z-margin above the arm's threshold.
        arg = (abs(z_relevant) - z_threshold) / cfg.z_scale
        # Clamp to avoid math.exp overflow for extreme Z-scores.
        arg = max(-50.0, min(50.0, arg))
        base = 1.0 / (1.0 + math.exp(-arg))

        # toxicity: linear decay from 1 at vpin_mid to 0 at vpin_high.
        span = cfg.vpin_high - cfg.vpin_mid
        if span <= 0.0:
            toxicity = 1.0
        else:
            raw = (cfg.vpin_high - vpin) / span
            toxicity = max(0.0, min(1.0, raw))

        # OBI alignment: neutral (1.0) without book data.
        if has_book:
            if exhaustion_type is ExhaustionType.SELL_EXHAUSTION:
                # Sellers are exhausted; for a long bounce, prefer bid pressure.
                raw = 0.5 + 0.5 * obi
            else:
                # Buyers are exhausted; for a short fade, prefer ask pressure.
                raw = 0.5 - 0.5 * obi
            obi_align = max(0.5, min(1.0, raw))
        else:
            obi_align = 1.0

        return max(0.0, min(1.0, base * toxicity * obi_align))


def _has_required_price_excursion(
    *,
    cfg: SymbolConfig,
    exhaustion_type: ExhaustionType,
    price: float,
    local_high: float,
    local_low: float,
    realized_vol: float,
) -> bool:
    floor_distance = price * max(0.0, cfg.min_price_excursion_bps) / 10_000.0
    vol_distance = (
        price
        * max(0.0, realized_vol)
        * max(
            0.0,
            cfg.min_price_excursion_vol_multiplier,
        )
    )
    required = max(floor_distance, vol_distance)
    if required <= 0.0:
        return True

    if exhaustion_type is ExhaustionType.SELL_EXHAUSTION:
        excursion = max(0.0, local_high - price)
    else:
        excursion = max(0.0, price - local_low)
    return excursion >= required
