"""Derive predictive features from raw snapshot columns.

The collector already writes a rich set of "raw" features per snapshot
(see :mod:`backend.features.snapshot`):

- ``mid``, ``microprice``, ``spread_bp``
- ``imbalance_top1`` / ``imbalance_top5`` / ``imbalance_top20``
- ``bid_vol_top5`` / ``bid_vol_top20`` (and ask)
- ``bid_weighted_depth_20`` / ``ask_weighted_depth_20`` (slope proxy)
- ``buy_volume_win`` / ``sell_volume_win`` / ``trade_count_win``
- ``bid_bkt_qty_05bp`` / ``..._50bp`` (depth in price buckets)
- top-40 bid/ask price+qty levels

These are statics. The model needs **dynamics**: short-horizon
realised return, rolling volatility, OFI (changes in best level qty
between snapshots), trade-flow imbalance, bucket-volume z-scores, ...

This module computes those features from a sorted (per-symbol)
DataFrame and returns the augmented frame plus the list of feature
column names to feed into LightGBM.

Implementation note: instead of relying on Polars' ``.over("symbol")``
window expressions everywhere (which mix poorly with ``shift`` /
``rolling_*`` chained inside arithmetic, see polars
``"window expression not allowed in aggregation"`` errors), we
partition the frame by symbol and apply the windowed expressions
per partition. With only a handful of symbols the loop overhead is
negligible compared to the heavy lifting Polars does inside each
partition.
"""

from __future__ import annotations

import logging

import polars as pl

log = logging.getLogger("backend.ml.features")

# Lag windows in number of snapshots. At 250 ms sampling these are
# 0.25 s, 1 s, 5 s, 30 s, 2 min.
LAG_STEPS = (1, 4, 20, 120, 480)

# Rolling-statistic windows (also in snapshots).
ROLL_WINDOWS = (4, 20, 120)

# Public list of column names this module appends. Returned alongside
# the augmented DataFrame for downstream consumers (training pipeline).
DERIVED_COLUMNS: tuple[str, ...] = (
    *(f"mid_ret_lag{lag}_bp" for lag in LAG_STEPS),
    *(f"vol_w{w}_bp" for w in ROLL_WINDOWS),
    "microprice_dev_bp",
    "ofi_top1",
    *(f"ofi_top1_sum_w{w}" for w in ROLL_WINDOWS),
    *(f"spread_norm_w{w}" for w in ROLL_WINDOWS),
    "bkt_concentration_5_50",
    "bkt_concentration_10_50",
    *(f"bkt_imb_{bp:02d}bp" for bp in (5, 10, 25, 50)),
    "trade_imb_win",
    *(f"trade_imb_w{w}" for w in ROLL_WINDOWS),
    "log_book_vol_top20",
    "vwap_dev_bp",
)


def _derived_for_one_symbol(sym_df: pl.DataFrame) -> pl.DataFrame:
    """Compute every derived column for a single symbol's sorted frame.

    Only ``shift`` / ``rolling_*`` operations live here, so we never
    need ``.over("symbol")`` and we never hit the "window expression
    not allowed in aggregation" trap.
    """
    eps = 1e-9

    # --- column-name shortcuts -------------------------------------------------
    mid_safe = pl.col("mid").cast(pl.Float64).clip(lower_bound=1e-12)

    new_cols: list[pl.Expr] = []

    # --- mid returns (in bp) at multiple lags ---------------------------------
    for lag in LAG_STEPS:
        prev_mid = mid_safe.shift(lag)
        new_cols.append(
            ((mid_safe - prev_mid) / prev_mid * 10_000.0).alias(f"mid_ret_lag{lag}_bp").cast(pl.Float32)
        )

    # --- rolling realised volatility of 1-step returns (bp) -------------------
    ret_1 = (mid_safe - mid_safe.shift(1)) / mid_safe.shift(1) * 10_000.0
    for w in ROLL_WINDOWS:
        new_cols.append(ret_1.rolling_std(window_size=w).alias(f"vol_w{w}_bp").cast(pl.Float32))

    # --- microprice deviation from mid (in bp) --------------------------------
    new_cols.append(
        ((pl.col("microprice") - pl.col("mid")) / mid_safe * 10_000.0)
        .alias("microprice_dev_bp")
        .cast(pl.Float32)
    )

    # --- order-flow imbalance proxy at top-of-book ----------------------------
    # Standard OFI: increase in bid-side qty when bid price didn't drop, etc.
    # We approximate with the change in best-bid-qty minus best-ask-qty between
    # consecutive snapshots, conditional on the price level being unchanged.
    # When the best level moves we reset (set OFI to 0) — in expectation this
    # is fine for a 1-step proxy at 250 ms sampling.
    bb = pl.col("best_bid").cast(pl.Float64)
    ba = pl.col("best_ask").cast(pl.Float64)
    bb_q = pl.col("best_bid_qty").cast(pl.Float64)
    ba_q = pl.col("best_ask_qty").cast(pl.Float64)
    bb_prev = bb.shift(1)
    ba_prev = ba.shift(1)
    bb_q_prev = bb_q.shift(1)
    ba_q_prev = ba_q.shift(1)

    bid_ofi = (
        pl.when(bb == bb_prev).then(bb_q - bb_q_prev).when(bb > bb_prev).then(bb_q).otherwise(-bb_q_prev)
    )
    ask_ofi = (
        pl.when(ba == ba_prev).then(-(ba_q - ba_q_prev)).when(ba < ba_prev).then(-ba_q).otherwise(ba_q_prev)
    )
    ofi = (bid_ofi + ask_ofi).fill_null(0.0)
    new_cols.append(ofi.alias("ofi_top1").cast(pl.Float32))

    # --- rolling OFI (more stable signal) -------------------------------------
    for w in ROLL_WINDOWS:
        new_cols.append(ofi.rolling_sum(window_size=w).alias(f"ofi_top1_sum_w{w}").cast(pl.Float32))

    # --- spread normalised by rolling median spread ---------------------------
    spr = pl.col("spread_bp").cast(pl.Float64)
    for w in ROLL_WINDOWS:
        med = spr.rolling_median(window_size=w).clip(lower_bound=1e-6)
        new_cols.append((spr / med).alias(f"spread_norm_w{w}").cast(pl.Float32))

    # --- bucket-volume slope: ratio of inner-bucket to outer-bucket size ------
    # If the book is concentrated near mid, inner/outer >> 1 (firm support).
    # If thin near mid (price likely to whip), inner/outer << 1.
    new_cols.append(
        (
            (pl.col("bid_bkt_qty_05bp") + pl.col("ask_bkt_qty_05bp"))
            / (pl.col("bid_bkt_qty_50bp") + pl.col("ask_bkt_qty_50bp") + eps)
        )
        .alias("bkt_concentration_5_50")
        .cast(pl.Float32)
    )
    new_cols.append(
        (
            (pl.col("bid_bkt_qty_10bp") + pl.col("ask_bkt_qty_10bp"))
            / (pl.col("bid_bkt_qty_50bp") + pl.col("ask_bkt_qty_50bp") + eps)
        )
        .alias("bkt_concentration_10_50")
        .cast(pl.Float32)
    )

    # --- bucket imbalance at multiple bp distances ----------------------------
    for bp_val in (5, 10, 25, 50):
        b = pl.col(f"bid_bkt_qty_{bp_val:02d}bp").cast(pl.Float64)
        a = pl.col(f"ask_bkt_qty_{bp_val:02d}bp").cast(pl.Float64)
        new_cols.append(((b - a) / (b + a + eps)).alias(f"bkt_imb_{bp_val:02d}bp").cast(pl.Float32))

    # --- trade-flow imbalance over the trade window ---------------------------
    bv = pl.col("buy_volume_win").cast(pl.Float64)
    sv = pl.col("sell_volume_win").cast(pl.Float64)
    new_cols.append(((bv - sv) / (bv + sv + eps)).alias("trade_imb_win").cast(pl.Float32))

    # --- rolling trade imbalance ----------------------------------------------
    for w in ROLL_WINDOWS:
        bv_w = bv.rolling_sum(window_size=w)
        sv_w = sv.rolling_sum(window_size=w)
        new_cols.append(((bv_w - sv_w) / (bv_w + sv_w + eps)).alias(f"trade_imb_w{w}").cast(pl.Float32))

    # --- log book volume (compressing the heavy tail) --------------------------
    new_cols.append(
        (pl.col("bid_vol_top20") + pl.col("ask_vol_top20"))
        .clip(lower_bound=eps)
        .log()
        .alias("log_book_vol_top20")
        .cast(pl.Float32)
    )

    # --- VWAP deviation from mid (bp) -----------------------------------------
    vwap = pl.col("vwap_win").cast(pl.Float64)
    new_cols.append(
        pl.when(vwap > 0)
        .then(((vwap - pl.col("mid")) / mid_safe) * 10_000.0)
        .otherwise(0.0)
        .alias("vwap_dev_bp")
        .cast(pl.Float32)
    )

    return sym_df.with_columns(new_cols)


def add_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Add derived features to ``df`` (must have ``symbol`` + ``ts_ms``).

    We partition the frame by symbol, apply per-partition feature
    expressions, then re-concat. With ~10 symbols the partition
    overhead is negligible — and avoiding ``.over("symbol")`` lets us
    use the natural ``shift``/``rolling_*`` API without hitting the
    polars "window expression not allowed in aggregation" rule.

    Returns the augmented frame and the list of derived column names.
    Together with the static raw-snapshot columns these form the full
    feature set passed to LightGBM (see :func:`feature_columns`).
    """
    if "symbol" not in df.columns or "ts_ms" not in df.columns:
        raise ValueError("df must have 'symbol' and 'ts_ms' columns")

    df = df.sort(["symbol", "ts_ms"])

    parts: list[pl.DataFrame] = []
    for key, sym_df in df.partition_by("symbol", as_dict=True).items():
        sym = key[0] if isinstance(key, tuple) else key
        sym_df = sym_df.sort("ts_ms")
        parts.append(_derived_for_one_symbol(sym_df))
        log.debug("features: %s done (%d rows)", sym, sym_df.height)

    df_out = pl.concat(parts, how="diagonal_relaxed").sort(["symbol", "ts_ms"])

    derived = list(DERIVED_COLUMNS)
    log.info("features: added %d derived columns across %d symbols", len(derived), len(parts))
    return df_out, derived


def feature_columns(df: pl.DataFrame, derived: list[str]) -> list[str]:
    """Return the full ordered list of feature column names for LightGBM.

    Includes the static raw-snapshot columns (best_bid_qty, imbalances,
    bucket qtys, ...) plus the ``derived`` ones added by
    :func:`add_features`.

    Excluded:
    - identifiers / timestamps (``ts_ms``, ``symbol``, ``part``)
    - raw prices (``best_bid``, ``best_ask``, ``mid``, ``microprice``,
      ``vwap_win``, top-N bid_p_NN / ask_p_NN) — using the absolute
      price as a feature would overfit to specific BTC/ETH price ranges
    - any label / target column (``y_*``, ``ret_*``)

    Top-N depth **quantities** are kept (they're price-invariant). Top-N
    **prices** are dropped — but their information is captured via the
    ``bkt_*`` and slope columns.
    """
    drop = {"ts_ms", "symbol", "part"}
    drop.update(c for c in df.columns if c.startswith("y_") or c.startswith("ret_"))
    drop.update({"best_bid", "best_ask", "mid", "microprice", "vwap_win", "spread"})
    drop.update(c for c in df.columns if c.startswith("bid_p_") or c.startswith("ask_p_"))

    static = [c for c in df.columns if c not in drop and c not in derived]
    # Filter to numeric columns only (LightGBM doesn't accept str/bool here).
    numeric_static: list[str] = []
    for c in static:
        if df[c].dtype.is_numeric():
            numeric_static.append(c)
    feats = numeric_static + list(derived)
    log.info("features: %d total (static=%d, derived=%d)", len(feats), len(numeric_static), len(derived))
    return feats


__all__ = ["DERIVED_COLUMNS", "LAG_STEPS", "ROLL_WINDOWS", "add_features", "feature_columns"]
