"""Build 3-class direction labels for multiple horizons.

For each row at time ``t`` we look forward by ``horizon_ms`` and find the
nearest snapshot at time ``>= t + horizon_ms``. The label is::

    return_bp = (future_mid - mid) / mid * 10_000
    UP   if return_bp >  threshold_bp
    DOWN if return_bp < -threshold_bp
    FLAT otherwise

The ``threshold_bp`` is **per-symbol** and adaptive: we set it to the
median absolute return at that horizon on the **train slice** (so train
labels are roughly balanced 33/33/33). Pinning it to a fixed bp value
biases the loss heavily toward FLAT for stable assets like BTC and
toward UP/DOWN for noisy microcaps.

The label is only valid if a future snapshot was actually observed
within ``horizon_ms + tolerance_ms``. Missing-label rows are dropped at
training time (we add a ``y_{H}_valid`` boolean column).

We map labels to integers ``0=DOWN, 1=FLAT, 2=UP`` for LightGBM.
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


@dataclass(slots=True, frozen=True)
class HorizonSpec:
    """One labelling horizon."""

    name: str  # short tag, e.g. "1s"
    ms: int  # forward horizon in milliseconds
    tolerance_ms: int  # accept future snapshot if within ``ms + tolerance_ms``


DEFAULT_HORIZONS: tuple[HorizonSpec, ...] = (
    HorizonSpec("1s", 1_000, 500),
    HorizonSpec("5s", 5_000, 1_000),
    HorizonSpec("30s", 30_000, 2_000),
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
        raise ValueError(
            f"horizon '{spec}' must end in ms / s / m (e.g. 500ms, 2s, 1m)"
        )
    if ms <= 0:
        raise ValueError(f"horizon '{spec}' must be positive")
    tol = max(min(ms // 2, 5_000), 250)
    return HorizonSpec(spec, ms, tol)


def parse_horizon_ms(spec: str) -> int:
    """Convenience wrapper for callers that only need milliseconds."""
    return parse_horizon_spec(spec).ms


def _classify(returns_bp: np.ndarray, threshold_bp: float) -> np.ndarray:
    """Vectorized 3-class classification of bp returns."""
    out = np.full(returns_bp.shape, LBL_FLAT, dtype=np.int8)
    out[returns_bp > threshold_bp] = LBL_UP
    out[returns_bp < -threshold_bp] = LBL_DOWN
    return out


def _label_one_symbol(sym_df: pl.DataFrame, horizons: tuple[HorizonSpec, ...]) -> pl.DataFrame:
    """Compute labels for a single sorted (by ts_ms) symbol frame.

    Uses an asof-join via numpy searchsorted. We find, for every row at
    ``ts_ms``, the index of the first row with ``ts_ms >= t + h``. If
    that row's actual ts is within the tolerance, we use its mid as the
    future price; otherwise the label is invalid.
    """
    if sym_df.height == 0:
        return sym_df

    ts = sym_df["ts_ms"].to_numpy()
    if "mid" in sym_df.columns:
        mid = sym_df["mid"].to_numpy().astype(np.float64)
    else:
        # Fallback for old snapshots without an explicit mid column.
        bb = sym_df["best_bid"].to_numpy().astype(np.float64)
        ba = sym_df["best_ask"].to_numpy().astype(np.float64)
        mid = (bb + ba) * 0.5

    new_cols: list[pl.Series] = []
    for h in horizons:
        target_ts = ts + h.ms
        # Index of the first row with ts_ms >= target_ts (or n if none).
        idx = np.searchsorted(ts, target_ts, side="left")
        valid = idx < ts.size
        # Clamp idx so we can index safely; we'll mask invalids later.
        safe_idx = np.where(valid, idx, ts.size - 1)
        actual_ts = ts[safe_idx]
        within_tolerance = (actual_ts - target_ts) <= h.tolerance_ms
        valid &= within_tolerance & (mid > 0.0)

        future_mid = mid[safe_idx]
        ret_bp = np.where(
            (mid > 0.0) & valid,
            (future_mid - mid) / np.maximum(mid, 1e-12) * 10_000.0,
            0.0,
        )

        new_cols.append(pl.Series(f"ret_{h.name}_bp", ret_bp, dtype=pl.Float32))
        new_cols.append(pl.Series(f"y_{h.name}_valid", valid, dtype=pl.Boolean))
        # Placeholder for y; threshold is applied after we know train slice.
        new_cols.append(pl.Series(f"y_{h.name}", np.zeros_like(ret_bp, dtype=np.int8), dtype=pl.Int8))

    return sym_df.with_columns(new_cols)


def add_labels(
    df: pl.DataFrame,
    *,
    horizons: tuple[HorizonSpec, ...] = DEFAULT_HORIZONS,
    min_threshold_bp: float = 0.5,
) -> tuple[pl.DataFrame, dict[tuple[str, str], float]]:
    """Add per-horizon labels to ``df``.

    Threshold is computed per (symbol, horizon) on the **train slice**
    only — never look at val/test when sizing the FLAT band. Returns
    the augmented frame **and** the dict of thresholds (in bp), so they
    can be saved alongside the trained model and re-applied at
    inference time on live snapshots.

    ``min_threshold_bp`` is a floor to avoid degenerate FLAT bands on
    extremely stable symbols.
    """
    if "symbol" not in df.columns:
        raise ValueError("df must have a 'symbol' column")

    out_frames: list[pl.DataFrame] = []
    thresholds: dict[tuple[str, str], float] = {}

    for sym, sym_df in df.partition_by("symbol", as_dict=True).items():
        sym_name = sym[0] if isinstance(sym, tuple) else sym
        sym_df = sym_df.sort("ts_ms")
        sym_df = _label_one_symbol(sym_df, horizons)

        train_mask = sym_df["part"] == "train"
        for h in horizons:
            valid_col = f"y_{h.name}_valid"
            ret_col = f"ret_{h.name}_bp"
            y_col = f"y_{h.name}"
            train_returns = sym_df.filter(train_mask & sym_df[valid_col])[ret_col]
            if train_returns.is_empty():
                threshold = float(min_threshold_bp)
            else:
                threshold = float(max(min_threshold_bp, train_returns.abs().median()))
            thresholds[(sym_name, h.name)] = threshold
            ret_arr = sym_df[ret_col].to_numpy()
            valid_arr = sym_df[valid_col].to_numpy()
            y = _classify(ret_arr, threshold)
            # Mark invalid rows as FLAT (will be filtered by valid mask anyway).
            y[~valid_arr] = LBL_FLAT
            sym_df = sym_df.with_columns(pl.Series(y_col, y, dtype=pl.Int8))
            log.info(
                "labels: %s/%s threshold=%.3fbp; class counts (train, valid only) = %s",
                sym_name, h.name, threshold,
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
    "HorizonSpec",
    "LBL_DOWN",
    "LBL_FLAT",
    "LBL_UP",
    "add_labels",
]
