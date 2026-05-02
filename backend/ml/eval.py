"""Evaluate trained models on the held-out test slice.

Metrics we care about for a market-taking strategy:

- ``AUC (UP vs DOWN)`` — pure ranking ability, ignores FLAT class.
- ``hit_rate@thr`` — fraction of confidently-predicted directions that
  realised in the right direction (excluding FLAT-realised cases).
- ``avg_net_bp@thr`` — average realised return per "trade" net of
  taker fees + half-spread. Negative means the model loses money.
- ``sim_sharpe_per_day`` — naive Sharpe assuming i.i.d. trades and
  86_400 / horizon trades per day.

These are coarse metrics. They don't model order book impact or
queue position; that comes in MVP-2. For MVP-1 they're enough to tell
us whether there's *any* edge in the features.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import polars as pl

from backend.ml.labels import LBL_DOWN, LBL_UP, HorizonSpec

log = logging.getLogger("backend.ml.eval")

# Binance USDT-M Futures taker fee (BNB-disabled, no VIP). One-way.
TAKER_FEE_BP = 4.0


def _confidence(p: np.ndarray) -> np.ndarray:
    """``conf = P(UP) - P(DOWN)`` in [-1, 1]. Sign is the predicted direction."""
    return p[:, LBL_UP] - p[:, LBL_DOWN]


def _binary_auc(y_true_up: np.ndarray, score: np.ndarray) -> float:
    """AUC of ``score`` as a UP probability against binary labels.

    Standard rank-based estimator; no sklearn dependency.
    """
    pos = score[y_true_up == 1]
    neg = score[y_true_up == 0]
    n_pos = pos.size
    n_neg = neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Mann-Whitney U.
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, all_scores.size + 1)
    # Average ranks for ties.
    sorted_scores = all_scores[order]
    i = 0
    while i < sorted_scores.size:
        j = i + 1
        while j < sorted_scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = (ranks[order[i]] + ranks[order[j - 1]]) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg
        i = j
    rank_sum_pos = ranks[:n_pos].sum()  # (the first n_pos in concat are pos)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _pick_threshold(p_test: np.ndarray, target_trade_frac: float = 0.05) -> float:
    """Choose a confidence threshold that retains ``target_trade_frac`` of rows.

    We pick the threshold so the top ``target_trade_frac`` of |confidence|
    rows pass. 5% default = ~17_280 trades/day on a 250 ms stream — already
    aggressive, but reasonable for evaluation.
    """
    conf = np.abs(_confidence(p_test))
    if conf.size == 0:
        return 1.0
    q = 1.0 - target_trade_frac
    return float(np.quantile(conf, q))


def evaluate_holdout(
    booster,
    df_lab: pl.DataFrame,
    feats: list[str],
    horizon: HorizonSpec,
    symbols: list[str],
    use_symbol_feature: bool,
) -> dict:
    """Run prediction on test slice and produce a metrics dict.

    Returns a JSON-serialisable dict with the headline metrics + a few
    diagnostics.
    """
    test = df_lab.filter((pl.col("part") == "test") & pl.col(f"y_{horizon.name}_valid"))
    if test.height == 0:
        log.warning("eval: no valid test rows for horizon %s", horizon.name)
        return {"horizon": horizon.name, "n_test": 0}

    X_cols = (["_symbol_id"] if use_symbol_feature else []) + feats
    X = test.select(X_cols).to_numpy().astype(np.float32, copy=False)
    y = test[f"y_{horizon.name}"].to_numpy()
    ret_bp = test[f"ret_{horizon.name}_bp"].to_numpy().astype(np.float64)

    p = booster.predict(X)  # shape (n, 3)

    conf = _confidence(p)

    # AUC(UP vs DOWN): use only rows realised UP or DOWN.
    mask_dir = (y == LBL_UP) | (y == LBL_DOWN)
    auc_up_dn = float("nan")
    if mask_dir.sum() > 0:
        y_bin = (y[mask_dir] == LBL_UP).astype(np.int8)
        auc_up_dn = _binary_auc(y_bin, conf[mask_dir])

    thr = _pick_threshold(p, target_trade_frac=0.05)

    # "Trades" = rows where |conf| > thr, side = sign(conf).
    trade_mask = np.abs(conf) > thr
    n_trades = int(trade_mask.sum())
    side = np.sign(conf)
    realized_bp = ret_bp * side  # signed realised return for our hypothetical trade
    cost_bp = 2.0 * TAKER_FEE_BP  # entry + exit
    net_bp = realized_bp - cost_bp

    if n_trades > 0:
        avg_net = float(net_bp[trade_mask].mean())
        std_net = float(net_bp[trade_mask].std())
        # Hit rate: realised in same direction as our trade, excluding FLAT-realisations.
        realized_dir = np.sign(ret_bp[trade_mask])
        traded_side = side[trade_mask]
        hits = int(((realized_dir != 0) & (realized_dir == traded_side)).sum())
        hit_rate = hits / max(int((realized_dir != 0).sum()), 1)
    else:
        avg_net = float("nan")
        std_net = float("nan")
        hit_rate = float("nan")

    # Naive Sharpe per day. We're doing 'one trade per snapshot' max; trades/day
    # at 250 ms snapshot rate and 5% trade frac = 0.05 * 86400 / 0.25 = 17_280.
    trades_per_day = 0.05 * 86_400.0 / 0.25
    sim_sharpe = float("nan")
    if std_net and not math.isnan(std_net) and std_net > 0:
        sim_sharpe = (avg_net / std_net) * math.sqrt(trades_per_day)

    # Per-symbol breakdown.
    by_symbol: dict[str, dict] = {}
    if "_symbol_id" in test.columns:
        sym_ids = test["_symbol_id"].to_numpy()
        for sid in np.unique(sym_ids):
            sym_name = symbols[int(sid)] if int(sid) < len(symbols) else f"id{int(sid)}"
            m = sym_ids == sid
            if m.sum() == 0:
                continue
            md = m & trade_mask
            if md.sum() == 0:
                by_symbol[sym_name] = {"n_test": int(m.sum()), "n_trades": 0}
                continue
            by_symbol[sym_name] = {
                "n_test": int(m.sum()),
                "n_trades": int(md.sum()),
                "avg_net_bp": float(net_bp[md].mean()),
                "hit_rate": float(
                    ((np.sign(ret_bp[md]) != 0) & (np.sign(ret_bp[md]) == side[md])).sum()
                    / max(int((np.sign(ret_bp[md]) != 0).sum()), 1)
                ),
            }

    return {
        "horizon": horizon.name,
        "n_test": int(test.height),
        "auc_up_vs_down": auc_up_dn,
        "threshold_confidence": thr,
        "n_trades_at_threshold": n_trades,
        "hit_rate_at_threshold": hit_rate,
        "avg_net_bp_at_threshold": avg_net,
        "std_net_bp_at_threshold": std_net,
        "sim_sharpe_per_day": sim_sharpe,
        "taker_fee_bp": TAKER_FEE_BP,
        "by_symbol": by_symbol,
    }


__all__ = ["evaluate_holdout", "TAKER_FEE_BP"]
