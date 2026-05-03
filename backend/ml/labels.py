"""Build 3-class direction labels for multiple horizons.

Two label modes are supported:

- ``cost_mode="mid"`` (legacy v1): labels look at the future mid price and
  pick a per-symbol FLAT band sized to the median absolute mid-to-mid
  return. **This ignores the bid-ask spread completely**: a "UP" label
  on a 4 bp predicted move is profitable on paper but bleeds money in
  live taker execution because crossing the spread costs an additional
  3-8 bp on micro-cap altcoins. We keep this mode for back-compat with
  models trained on v1 parquets, but it should NOT be used for any new
  training.

- ``cost_mode="taker"`` (v2 default): labels simulate the realistic
  taker round-trip directly. For a long entered at time ``t`` and held
  until ``t + horizon``::

      gross_long_bp = (bb_fut - ba_now) / mid_now * 10_000

  i.e. buy the offer now, sell the bid later. Symmetrically for shorts::

      gross_short_bp = (bb_now - ba_fut) / mid_now * 10_000

  The classifier pulls the trigger when EITHER side beats a per-symbol
  threshold ``= mean(spread_bp) + commission_bp + safety_margin_bp``::

      UP   if gross_long_bp  > threshold_bp  (profitable long round-trip)
      DOWN if gross_short_bp > threshold_bp  (profitable short round-trip)
      FLAT otherwise                         (spread + fees eat the move)

  Most rows that were UP/DOWN under ``cost_mode="mid"`` collapse to FLAT
  here — and that's the point. The model learns to fire only on
  predictively-profitable taker round-trips, not on ambient mid drift.

Both modes also produce a ``ret_{H}_bp`` column (mid-to-mid return) for
diagnostic use; the LightGBM target column is always ``y_{H}``.

Labels are valid only when a future snapshot was observed within
``horizon_ms + tolerance_ms``. Missing-label rows carry ``y_{H}_valid =
False`` and are filtered out at training time.

Class encoding for LightGBM: ``0=DOWN, 1=FLAT, 2=UP``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

log = logging.getLogger("backend.ml.labels")

LBL_DOWN = 0
LBL_FLAT = 1
LBL_UP = 2

# Default fee assumption: Binance USDT-M Futures taker, BNB-disabled, no VIP.
# Round-trip = 2 sides * 4.0 bp = 8.0 bp.
TAKER_FEE_PER_SIDE_BP = 4.0
TAKER_RT_FEE_BP = 2.0 * TAKER_FEE_PER_SIDE_BP

# Margin added to the threshold so we only label rows whose net edge
# exceeds the noise floor of the cost estimate. Empirically 3 bp is small
# enough to keep enough UP/DOWN samples on most alts and large enough that
# we don't mistake spread jitter for an edge.
DEFAULT_SAFETY_MARGIN_BP = 3.0


@dataclass(slots=True, frozen=True)
class HorizonSpec:
    """One labelling horizon."""

    name: str  # short tag, e.g. "1s"
    ms: int  # forward horizon in milliseconds
    tolerance_ms: int  # accept future snapshot if within ``ms + tolerance_ms``


DEFAULT_HORIZONS: tuple[HorizonSpec, ...] = (
    HorizonSpec("2s", 2_000, 500),
    HorizonSpec("5s", 5_000, 1_000),
    HorizonSpec("15s", 15_000, 2_000),
)


def parse_horizon_spec(spec: str) -> HorizonSpec:
    """Parse a horizon string like ``2s``, ``500ms``, ``15s``, ``1m``, ``5m``.

    Tolerance is auto-set to roughly ``min(50% of horizon, 5s)`` so very
    short horizons stay tight while longer ones tolerate small gaps in
    the snapshot stream. Used by training, backtest, and the paper trader
    so they all agree on what e.g. ``2s`` means.
    """
    s = spec.strip().lower()
    if s.endswith("ms"):
        ms = int(s[:-2])
    elif s.endswith("s"):
        ms = int(float(s[:-1]) * 1000)
    elif s.endswith("m"):
        ms = int(float(s[:-1]) * 60_000)
    else:
        raise ValueError(f"horizon '{spec}' must end in ms / s / m (e.g. 500ms, 2s, 1m)")
    if ms <= 0:
        raise ValueError(f"horizon '{spec}' must be positive")
    tol = max(min(ms // 2, 5_000), 250)
    return HorizonSpec(spec, ms, tol)


def parse_horizon_ms(spec: str) -> int:
    """Convenience wrapper for callers that only need milliseconds."""
    return parse_horizon_spec(spec).ms


def _classify_mid(returns_bp: np.ndarray, threshold_bp: float) -> np.ndarray:
    """v1: 3-class classification of mid-to-mid returns."""
    out = np.full(returns_bp.shape, LBL_FLAT, dtype=np.int8)
    out[returns_bp > threshold_bp] = LBL_UP
    out[returns_bp < -threshold_bp] = LBL_DOWN
    return out


def _classify_taker(
    gross_long_bp: np.ndarray,
    gross_short_bp: np.ndarray,
    threshold_bp: float,
) -> np.ndarray:
    """v2: label UP/DOWN only when the one-sided taker round-trip clears costs.

    A long round-trip can be profitable while a short round-trip on the
    same row also is (e.g. mid drifted up faster than the spread): in
    that case we pick whichever has the larger margin over threshold.
    Symmetrically for the rare opposite case.
    """
    out = np.full(gross_long_bp.shape, LBL_FLAT, dtype=np.int8)
    long_ok = gross_long_bp > threshold_bp
    short_ok = gross_short_bp > threshold_bp
    out[long_ok & ~short_ok] = LBL_UP
    out[short_ok & ~long_ok] = LBL_DOWN
    both = long_ok & short_ok
    if both.any():
        out[both] = np.where(gross_long_bp[both] >= gross_short_bp[both], LBL_UP, LBL_DOWN).astype(np.int8)
    return out


def _label_one_symbol(sym_df: pl.DataFrame, horizons: tuple[HorizonSpec, ...]) -> pl.DataFrame:
    """Compute per-row return diagnostics for a single sorted (by ts_ms) symbol frame.

    Always emits ``ret_{H}_bp`` (mid-to-mid, kept for compat) plus the v2
    spread-aware columns ``gross_long_{H}_bp`` / ``gross_short_{H}_bp``.
    The actual ``y_{H}`` integer label is filled in by :func:`add_labels`
    once the per-symbol threshold has been computed.

    Uses an asof-join via :func:`numpy.searchsorted`: for every row at
    ``ts_ms``, we find the first row with ``ts_ms >= t + h``. If that
    row's actual ts is within tolerance, we use its prices; otherwise
    the label is invalid.
    """
    if sym_df.height == 0:
        return sym_df

    ts = sym_df["ts_ms"].to_numpy()
    bb = sym_df["best_bid"].to_numpy().astype(np.float64)
    ba = sym_df["best_ask"].to_numpy().astype(np.float64)
    if "mid" in sym_df.columns:
        mid = sym_df["mid"].to_numpy().astype(np.float64)
    else:
        mid = (bb + ba) * 0.5

    inv_mid = np.where(mid > 0.0, 10_000.0 / np.maximum(mid, 1e-12), 0.0)

    new_cols: list[pl.Series] = []
    for h in horizons:
        target_ts = ts + h.ms
        idx = np.searchsorted(ts, target_ts, side="left")
        valid = idx < ts.size
        safe_idx = np.where(valid, idx, ts.size - 1)
        actual_ts = ts[safe_idx]
        within_tolerance = (actual_ts - target_ts) <= h.tolerance_ms
        valid &= within_tolerance & (mid > 0.0)

        bb_fut = bb[safe_idx]
        ba_fut = ba[safe_idx]
        future_mid = mid[safe_idx]

        ret_bp = np.where(valid, (future_mid - mid) * inv_mid, 0.0)
        gross_long_bp = np.where(valid, (bb_fut - ba) * inv_mid, 0.0)
        gross_short_bp = np.where(valid, (bb - ba_fut) * inv_mid, 0.0)

        new_cols.append(pl.Series(f"ret_{h.name}_bp", ret_bp, dtype=pl.Float32))
        new_cols.append(pl.Series(f"gross_long_{h.name}_bp", gross_long_bp, dtype=pl.Float32))
        new_cols.append(pl.Series(f"gross_short_{h.name}_bp", gross_short_bp, dtype=pl.Float32))
        new_cols.append(pl.Series(f"y_{h.name}_valid", valid, dtype=pl.Boolean))
        new_cols.append(pl.Series(f"y_{h.name}", np.zeros_like(ret_bp, dtype=np.int8), dtype=pl.Int8))

    return sym_df.with_columns(new_cols)


def _spread_bp_array(sym_df: pl.DataFrame) -> np.ndarray:
    """Return spread in bp; prefer the precomputed ``spread_bp`` column."""
    if "spread_bp" in sym_df.columns:
        return sym_df["spread_bp"].to_numpy().astype(np.float64)
    bb = sym_df["best_bid"].to_numpy().astype(np.float64)
    ba = sym_df["best_ask"].to_numpy().astype(np.float64)
    mid = (bb + ba) * 0.5
    return np.where(mid > 0.0, (ba - bb) / np.maximum(mid, 1e-12) * 10_000.0, 0.0)


def add_labels(
    df: pl.DataFrame,
    *,
    horizons: tuple[HorizonSpec, ...] = DEFAULT_HORIZONS,
    cost_mode: str = "taker",
    min_threshold_bp: float = 0.5,
    rt_fee_bp: float = TAKER_RT_FEE_BP,
    safety_margin_bp: float = DEFAULT_SAFETY_MARGIN_BP,
) -> tuple[pl.DataFrame, dict[tuple[str, str], float]]:
    """Add per-horizon labels to ``df``.

    Parameters
    ----------
    df:
        Snapshot frame with ``symbol``, ``ts_ms``, ``best_bid``, ``best_ask``,
        and (optionally) ``mid`` / ``spread_bp`` / ``part``.
    horizons:
        Tuple of :class:`HorizonSpec` to label.
    cost_mode:
        ``"taker"`` (v2 default) uses the spread-aware bid-ask round-trip
        and a threshold of ``mean(spread_bp) + rt_fee_bp + safety_margin_bp``.
        ``"mid"`` falls back to the v1 mid-to-mid logic with the
        median-absolute-return FLAT band.
    rt_fee_bp:
        Round-trip taker fee in bp; ``8.0`` corresponds to Binance USDT-M
        Futures default tier, no BNB discount.
    safety_margin_bp:
        Extra bp added on top of the spread+fee threshold so we only label
        rows that decisively beat the cost noise floor.

    Returns
    -------
    df:
        Original frame plus ``ret_{H}_bp``, ``gross_long_{H}_bp``,
        ``gross_short_{H}_bp``, ``y_{H}_valid``, ``y_{H}`` per horizon.
    thresholds:
        ``(symbol, horizon_name) -> threshold_bp`` dict; saved alongside
        the trained model so live inference uses the same band.
    """
    if "symbol" not in df.columns:
        raise ValueError("df must have a 'symbol' column")
    if cost_mode not in ("taker", "mid"):
        raise ValueError(f"cost_mode must be 'taker' or 'mid', got {cost_mode!r}")

    out_frames: list[pl.DataFrame] = []
    thresholds: dict[tuple[str, str], float] = {}

    has_part = "part" in df.columns

    for sym, sym_df in df.partition_by("symbol", as_dict=True).items():
        sym_name = sym[0] if isinstance(sym, tuple) else sym
        sym_df = sym_df.sort("ts_ms")
        sym_df = _label_one_symbol(sym_df, horizons)

        if has_part:
            train_mask = sym_df["part"] == "train"
        else:
            train_mask = pl.repeat(True, n=sym_df.height, eager=True)

        if cost_mode == "taker":
            train_spread = _spread_bp_array(sym_df.filter(train_mask))
            if train_spread.size > 0:
                mean_spread = float(np.mean(train_spread[train_spread >= 0.0]))
            else:
                mean_spread = 0.0
            base_threshold = max(
                min_threshold_bp,
                mean_spread + rt_fee_bp + safety_margin_bp,
            )
        else:
            base_threshold = None  # filled per-horizon below in mid mode

        for h in horizons:
            valid_col = f"y_{h.name}_valid"
            ret_col = f"ret_{h.name}_bp"
            long_col = f"gross_long_{h.name}_bp"
            short_col = f"gross_short_{h.name}_bp"
            y_col = f"y_{h.name}"
            valid_arr = sym_df[valid_col].to_numpy()

            if cost_mode == "taker":
                threshold = float(base_threshold)
                gross_long = sym_df[long_col].to_numpy().astype(np.float64)
                gross_short = sym_df[short_col].to_numpy().astype(np.float64)
                y = _classify_taker(gross_long, gross_short, threshold)
            else:
                train_returns = sym_df.filter(train_mask & sym_df[valid_col])[ret_col]
                if train_returns.is_empty():
                    threshold = float(min_threshold_bp)
                else:
                    threshold = float(max(min_threshold_bp, train_returns.abs().median()))
                ret_arr = sym_df[ret_col].to_numpy()
                y = _classify_mid(ret_arr, threshold)

            thresholds[(sym_name, h.name)] = threshold
            y[~valid_arr] = LBL_FLAT
            sym_df = sym_df.with_columns(pl.Series(y_col, y, dtype=pl.Int8))
            log.info(
                "labels[%s]: %s/%s threshold=%.3fbp; class counts (train, valid only) = %s",
                cost_mode,
                sym_name,
                h.name,
                threshold,
                _class_counts(sym_df.filter(train_mask & sym_df[valid_col])[y_col]),
            )
        out_frames.append(sym_df)

    df_out = pl.concat(out_frames, how="diagonal_relaxed").sort(["symbol", "ts_ms"])
    return df_out, thresholds


def _class_counts(s: pl.Series) -> dict[str, int]:
    n = s.len()
    if n == 0:
        return {"DOWN": 0, "FLAT": 0, "UP": 0, "total": 0}
    counts = s.value_counts()
    by = {int(v): int(c) for v, c in zip(counts[s.name], counts["count"], strict=True)}
    return {
        "DOWN": by.get(LBL_DOWN, 0),
        "FLAT": by.get(LBL_FLAT, 0),
        "UP": by.get(LBL_UP, 0),
        "total": n,
    }


__all__ = [
    "DEFAULT_HORIZONS",
    "DEFAULT_SAFETY_MARGIN_BP",
    "HorizonSpec",
    "LBL_DOWN",
    "LBL_FLAT",
    "LBL_UP",
    "TAKER_FEE_PER_SIDE_BP",
    "TAKER_RT_FEE_BP",
    "add_labels",
    "parse_horizon_spec",
    "parse_horizon_ms",
]
