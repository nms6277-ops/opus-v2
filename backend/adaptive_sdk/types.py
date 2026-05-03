"""Public data contracts for AdaptiveAnalyticsSDK.

All value objects are immutable (``frozen=True``) to prevent accidental mutation
by downstream consumers. Configuration objects are mutable for ergonomic
construction via keyword arguments.

All public structures use ``@dataclass(slots=True)``; no untyped ``dict`` is
exposed across the API boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ClassificationMode(Enum):
    """How to assign aggressor sign to bucket volumes.

    - ``TRADE_SIGN``: uses each trade's aggressor flag (``is_buyer_maker``).
      This is the canonical VPIN formulation and should be preferred whenever
      a reliable aggressor flag is available (Binance, Bybit, OKX, ...).
    - ``BVC``: Bulk Volume Classification (Easley, Lopez de Prado, O'Hara 2012).
      Classifies the whole bucket at close via the normal CDF of the intra-
      bucket log-return normalized by a bucket-scale volatility. This is a
      *fallback* for feeds without an aggressor flag or for cross-venue
      normalization; Poppe-Moeller-Schiereck (2016) showed it underperforms
      tick-rule on aggressor-flag-rich exchanges.
    """

    TRADE_SIGN = 1
    BVC = 2


class BucketFillMode(Enum):
    """How a bucket handles a trade that overflows its remaining capacity.

    - ``EXACT_FILL``: the trade is split across buckets; the chunk that fits
      into the current bucket closes it, the remainder starts the next bucket.
      This is the canonical VPIN behavior and is mandatory for TRADE_SIGN.
    - ``MINIMUM_FILL``: the first trade that causes the bucket total to cross
      the threshold closes the bucket *without* splitting. The whole trade is
      attributed to the bucket. This follows Panayides et al. on bias
      reduction for BVC volume bars.
    """

    EXACT_FILL = 1
    MINIMUM_FILL = 2


class BucketSizePolicy(Enum):
    """How the bucket size is chosen."""

    FIXED = 1
    ROLLING_DAILY = 2


class BVCSigmaSource(Enum):
    """Source of the volatility used to normalize BVC log-returns.

    In v1 only ``BUCKET_RETURNS_EWMA`` is implemented: the EWMA of squared
    cross-bucket log-returns ``ln(P_close_i / P_close_{i-1})``.
    """

    BUCKET_RETURNS_EWMA = 1


class ExhaustionType(Enum):
    """Direction of the detected flow exhaustion."""

    SELL_EXHAUSTION = 1
    BUY_EXHAUSTION = 2


class ArmId(Enum):
    """Bandit arms differ by the Z-score threshold required to fire."""

    AGGRESSIVE = 0  # low threshold (more entries, noisier)
    NORMAL = 1
    CONSERVATIVE = 2  # high threshold (fewer but cleaner entries)


# ---------------------------------------------------------------------------
# Immutable value objects (public wire contracts)
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class TradeTick:
    """A single trade print.

    ``is_buyer_maker`` follows the Binance convention: ``True`` means the buyer
    was the maker, i.e. the *seller* initiated (was aggressor).
    """

    symbol: str
    price: float
    quantity: float
    is_buyer_maker: bool
    timestamp: float


@dataclass(slots=True, frozen=True)
class BookSnapshot:
    """Top-of-book snapshot used for Order Book Imbalance (OBI)."""

    symbol: str
    best_bid: float
    best_ask: float
    bid_vol: float
    ask_vol: float
    timestamp: float


@dataclass(slots=True, frozen=True)
class MetricsSnapshot:
    """Metrics attached to an emitted :class:`Signal` for diagnostics."""

    vpin: float
    z_score_buy_flow: float
    z_score_sell_flow: float
    obi: float
    bucket_size: float
    buckets_filled: int
    realized_vol: float = 0.0


@dataclass(slots=True, frozen=True)
class Signal:
    """An actionable exhaustion signal emitted by the SDK."""

    signal_id: str
    symbol: str
    timestamp: float
    exhaustion_type: ExhaustionType
    confidence: float
    arm_id: int
    metrics: MetricsSnapshot


@dataclass(slots=True, frozen=True)
class Outcome:
    """Post-trade report consumed by :meth:`AdaptiveAnalyticsSDK.report_outcome`.

    ``mfe``, ``mae`` and ``holding_time_ms`` are *not* in the default reward
    function (``pnl - fees - slippage``). They are preserved for analytics and
    for user-provided ``reward_transform`` hooks on :class:`SymbolConfig`.
    """

    signal_id: str
    pnl: float
    mfe: float
    mae: float
    fees: float
    slippage: float
    holding_time_ms: float


@dataclass(slots=True, frozen=True)
class SymbolState:
    """Polling view of a symbol's current analytics state."""

    is_ready: bool
    vpin: float
    sell_exhaustion_z: float
    buy_exhaustion_z: float
    pending_signals_count: int
    buckets_filled: int
    realized_vol: float = 0.0


# ---------------------------------------------------------------------------
# Mutable configuration objects
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SymbolConfig:
    """Per-symbol configuration.

    NOTE on ``vpin_mid`` / ``vpin_high``: these are *starting heuristics* for
    liquid spot crypto (BTC/ETH). Production deployments MUST calibrate them
    per ``(symbol, regime)`` from historical VPIN distributions (e.g. q60/q90
    quantiles). They are not universal constants.
    """

    classification_mode: ClassificationMode = ClassificationMode.TRADE_SIGN
    bucket_fill_mode: BucketFillMode = BucketFillMode.EXACT_FILL
    bucket_policy: BucketSizePolicy = BucketSizePolicy.FIXED

    # Bucket sizing
    fixed_bucket_size: float = 10.0
    target_buckets_per_day: int = 50
    target_buckets_per_window: int | None = None
    rolling_volume_window_s: float = 86_400.0
    min_bucket_size: float = 0.1
    bucket_size_recompute_interval_s: float = 60.0

    # BVC sigma
    bvc_sigma_source: BVCSigmaSource = BVCSigmaSource.BUCKET_RETURNS_EWMA
    bvc_lambda: float = 0.06
    min_buckets_for_bvc_sigma: int = 10

    # VPIN window & warmup
    vpin_window: int = 50
    min_buckets_for_vpin: int = 50
    min_ticks_for_z: int = 200

    # Exhaustion detector
    price_extrema_window_ms: int = 3000
    realized_vol_window_ms: int = 60_000
    min_price_excursion_bps: float = 0.0
    min_price_excursion_vol_multiplier: float = 0.0
    flow_window_ms: int = 1000
    flow_lambda: float = 0.06

    # Toxicity gate / confidence
    vpin_mid: float = 0.30
    vpin_high: float = 0.50
    z_thresholds: tuple[float, float, float] = (1.5, 2.0, 2.5)
    z_scale: float = 0.5

    # Pending signals registry
    signal_ttl_seconds: float = 3600.0
    pending_signals_max: int = 10_000
    pending_cleanup_interval_ticks: int = 1000

    # Optional reward hook (default = pnl - fees - slippage)
    reward_transform: Callable[[Outcome], float] | None = None


@dataclass(slots=True)
class GlobalConfig:
    """Top-level SDK configuration."""

    default_symbol_config: SymbolConfig = field(default_factory=SymbolConfig)


# ---------------------------------------------------------------------------
# Internal (package-private) value types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BucketResult:
    """A single volume bucket (mutable during fill, snapshotted on close)."""

    v_buy: float
    v_sell: float
    v_total: float
    p_open: float
    p_close: float


@dataclass(slots=True)
class PendingSignal:
    """Internal registry entry for tracking ``signal_id -> arm``."""

    symbol: str
    arm_id: int
    timestamp: float
    is_resolved: bool = False
