"""Vectorised backtest: maker vs taker scenarios on the held-out test slice.

The training pipeline emits a labelled DataFrame with one row per snapshot
and a ``ret_{H}_bp`` column for each trained horizon. This module loads
the trained LightGBM bundle, replays it on the ``part == "test"`` slice,
turns the predicted ``confidence = P(UP) − P(DOWN)`` into one of three
trade decisions per row (long, short, no-trade) using a top-N
threshold, and computes realised PnL under several fee assumptions:

- **Taker only**: pay ``2 × taker_fee_bp`` round-trip. This is what the
  paper trader currently does in :mod:`backend.traders.paper`.
- **Maker only (always filled)**: pay ``2 × maker_fee_bp`` (could be
  negative for VIPs / rebates) and skip the spread. Best-case ceiling.
- **Maker only (crossing-only fill)**: only count rows where the next
  snapshot crossed our limit — i.e. the bid moved down into our long
  limit, or the ask moved up into our short limit. More realistic
  approximation than always-filled because it discards rows where we
  would have queued forever.

The result is a JSON report saved to
``models_dir/backtest/h{horizon}.json`` and an aggregated summary
``models_dir/backtest/summary.json``. The summary is also printed in a
human-readable table at the end of ``main``.

Run as::

    python -m backend.ml.backtest --data-dir D:/opus-data --models-dir models/global \
        --horizons 1s 5s 30s --target-trade-frac 0.05

This module is **read-only on disk**: it never mutates parquets, models,
or the live runtime.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl

from backend.ml.dataset import load_dataset
from backend.ml.features import add_features, feature_columns
from backend.ml.labels import DEFAULT_HORIZONS, HorizonSpec, add_labels
from backend.ml.train import _add_symbol_id

log = logging.getLogger("backend.ml.backtest")


SCENARIOS = (
    # (label, fee per side bp, label_for_fill_model)
    ("taker", 4.0, "always"),       # market-take both ways
    ("maker_best", 2.0, "always"),   # default Binance maker, always filled
    ("maker_zero", 0.0, "always"),   # VIP3+/rebated maker, always filled
    ("maker_cross", 2.0, "cross"),   # default maker, only crossing fills
)


def _pick_threshold(conf: np.ndarray, target_frac: float) -> float:
    if conf.size == 0:
        return 1.0
    target_frac = min(max(target_frac, 0.001), 0.5)
    n_keep = max(1, int(conf.size * target_frac))
    sorted_abs = np.sort(np.abs(conf))[::-1]
    return float(sorted_abs[n_keep - 1])


def _maker_cross_fill_mask(side: np.ndarray, bb: np.ndarray, ba: np.ndarray) -> np.ndarray:
    """Approximate maker fill: a long limit at bid_t fills only if bid_{t+1} <= bid_t.

    Equivalently, a short limit at ask_t fills only if ask_{t+1} >= ask_t.
    This is a coarse proxy for "next-tick aggression came to our side".
    """
    if side.size < 2:
        return np.zeros_like(side, dtype=bool)
    next_bb = np.concatenate([bb[1:], bb[-1:]])
    next_ba = np.concatenate([ba[1:], ba[-1:]])
    long_fill = (side == 1) & (next_bb <= bb)
    short_fill = (side == -1) & (next_ba >= ba)
    return long_fill | short_fill


def _agg_metrics(
    net_bp: np.ndarray,
    ret_bp: np.ndarray,
    fill: np.ndarray,
    side: np.ndarray,
) -> dict:
    """Per-bucket metrics.

    ``hit_rate`` compares the realised price direction (``sign(ret_bp)``) to
    the trade ``side`` (``+1`` long, ``-1`` short, ``0`` no-trade). Computing
    it from the *net* PnL sign would be wrong on shorts, where a profitable
    trade has ``ret_bp < 0`` but ``net_bp > 0`` — see :mod:`backend.ml.eval`
    for the canonical reference implementation.
    """
    n_total = int(net_bp.size)
    n_filled = int(fill.sum())
    if n_filled == 0:
        return {
            "n_total": n_total,
            "n_filled": 0,
            "fill_rate": 0.0,
            "avg_net_bp": 0.0,
            "median_net_bp": 0.0,
            "hit_rate": 0.0,
            "sum_pnl_bp": 0.0,
            "std_net_bp": 0.0,
            "sharpe": 0.0,
        }
    nb = net_bp[fill]
    rb = ret_bp[fill]
    sd = side[fill]
    realised_dir = np.sign(rb)
    moved = realised_dir != 0
    n_moved = int(moved.sum())
    hits = int(((realised_dir == sd) & moved).sum())
    hit_rate = hits / n_moved if n_moved > 0 else 0.0
    return {
        "n_total": n_total,
        "n_filled": n_filled,
        "fill_rate": n_filled / max(n_total, 1),
        "avg_net_bp": float(nb.mean()),
        "median_net_bp": float(np.median(nb)),
        "hit_rate": float(hit_rate),
        "sum_pnl_bp": float(nb.sum()),
        "std_net_bp": float(nb.std(ddof=1)) if nb.size > 1 else 0.0,
        "sharpe": float(nb.mean() / (nb.std(ddof=1) + 1e-9)) if nb.size > 1 else 0.0,
    }


def _backtest_horizon(
    df_test: pl.DataFrame,
    horizon: HorizonSpec,
    feat_names: list[str],
    booster,
    target_trade_frac: float,
) -> dict:
    valid_col = f"y_{horizon.name}_valid"
    ret_col = f"ret_{horizon.name}_bp"
    test = df_test.filter(pl.col(valid_col))
    if test.height == 0:
        return {"horizon": horizon.name, "error": "no valid test rows"}

    X = test.select(feat_names).to_numpy()
    proba = booster.predict(X, num_iteration=booster.best_iteration or None)
    proba = np.asarray(proba, dtype=np.float64)
    conf = proba[:, 2] - proba[:, 0]
    ret_bp = test[ret_col].to_numpy().astype(np.float64)

    bb = test["best_bid"].to_numpy().astype(np.float64)
    ba = test["best_ask"].to_numpy().astype(np.float64)

    thr = _pick_threshold(conf, target_trade_frac)
    side = np.where(conf >= thr, 1, np.where(conf <= -thr, -1, 0)).astype(np.int8)
    is_trade = side != 0

    # gross PnL per row (before fees), in basis points. side * realised return.
    gross = side.astype(np.float64) * ret_bp

    # Cross-fill mask depends on whether the next tick crossed our limit price.
    # The dataset is sorted (symbol, ts_ms), so the row immediately after the
    # last row of symbol A belongs to a different symbol B with a totally
    # different price scale. Comparing those two would produce spurious fills
    # at every symbol boundary — instead, compute the mask per symbol so the
    # last row of each symbol can never report a "fill" against the next.
    syms_arr = test["symbol"].to_numpy()
    cross_mask_full = np.zeros_like(is_trade, dtype=bool)
    for s in np.unique(syms_arr):
        m = syms_arr == s
        cross_mask_full[m] = _maker_cross_fill_mask(side[m], bb[m], ba[m])

    by_scenario: dict[str, dict] = {}
    for label, fee_per_side_bp, fill_model in SCENARIOS:
        rt_fee_bp = 2.0 * fee_per_side_bp
        net = gross - rt_fee_bp
        if fill_model == "always":
            fill = is_trade
        elif fill_model == "cross":
            fill = is_trade & cross_mask_full
        else:
            fill = is_trade
        by_scenario[label] = {
            "fee_per_side_bp": fee_per_side_bp,
            "fill_model": fill_model,
            **_agg_metrics(net, ret_bp, fill, side),
        }

    # Per-symbol breakdown for the taker scenario (most operationally relevant).
    per_symbol: dict[str, dict] = {}
    for s in np.unique(syms_arr):
        m = syms_arr == s
        per_symbol[str(s)] = {
            label: _agg_metrics(
                gross[m] - 2.0 * data["fee_per_side_bp"],
                ret_bp[m],
                (is_trade & (cross_mask_full if data["fill_model"] == "cross" else np.ones_like(is_trade)))[m],
                side[m],
            )
            for label, data in (
                ("taker", by_scenario["taker"]),
                ("maker_best", by_scenario["maker_best"]),
                ("maker_zero", by_scenario["maker_zero"]),
                ("maker_cross", by_scenario["maker_cross"]),
            )
        }

    return {
        "horizon": horizon.name,
        "n_test_rows": int(test.height),
        "threshold_confidence": thr,
        "target_trade_frac": target_trade_frac,
        "n_trades": int(is_trade.sum()),
        "scenarios": by_scenario,
        "per_symbol": per_symbol,
    }


def run_backtest(
    *,
    data_dir: Path,
    models_dir: Path,
    horizons: Iterable[str] | None = None,
    target_trade_frac: float = 0.05,
    out_dir: Path | None = None,
    symbols: list[str] | None = None,
) -> dict:
    """End-to-end backtest. Returns the summary dict (also written to disk)."""
    try:
        import lightgbm as lgb  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError("lightgbm not installed; pip install lightgbm") from e

    out_dir = Path(out_dir) if out_dir is not None else Path(models_dir) / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("backtest: load %s", data_dir)
    df = load_dataset(data_dir, symbols=symbols)

    # Decide which horizons to backtest. If the user didn't specify any,
    # discover them by listing trained-model folders ``h{name}`` so custom
    # horizons (e.g. ``h2s``, ``h15s``) trained via ``train --horizons`` are
    # picked up automatically.
    #
    # train.py auto-appends ``global/`` to its --models-dir, so a typical
    # tree is::
    #     models/
    #     └── agnostic/
    #         └── global/
    #             ├── h1s/{model.lgb,meta.json,eval.json}
    #             ├── h5s/...
    #             └── h30s/...
    # The user is likely to type ``opus backtest --models-dir models\agnostic``
    # rather than ``models\agnostic\global``, so we look in both locations
    # transparently. The "global subfolder" wins if both contain models.
    from backend.ml.labels import parse_horizon_spec
    defaults_by_name = {h.name.lower(): h for h in DEFAULT_HORIZONS}

    def _resolve(name: str) -> HorizonSpec:
        key = name.strip().lower()
        return defaults_by_name.get(key) or parse_horizon_spec(name)

    models_root = Path(models_dir)
    global_root = models_root / "global"
    # Pick the directory that actually contains h*-folders with a model.
    if any((global_root / d).is_dir() and (global_root / d / "model.lgb").exists()
           for d in (p.name for p in global_root.glob("h*") if p.is_dir())):
        models_root = global_root
        log.info("backtest: resolved --models-dir to %s (auto-detected "
                 "global/ subfolder created by train)", models_root)

    if horizons:
        use_horizons = [_resolve(h) for h in horizons]
    else:
        discovered = [
            p.name[1:] for p in sorted(models_root.glob("h*"))
            if p.is_dir() and (p / "model.lgb").exists()
        ]
        use_horizons = [_resolve(n) for n in discovered] if discovered else list(DEFAULT_HORIZONS)
    log.info("backtest: horizons = %s", [h.name for h in use_horizons])

    df, derived = add_features(df)
    feats = feature_columns(df, derived)
    df, _thresholds = add_labels(df, horizons=tuple(use_horizons))
    df, _syms = _add_symbol_id(df)
    df_test = df.filter(pl.col("part") == "test")
    log.info("backtest: test slice rows=%d", df_test.height)

    summary: dict = {"horizons": {}}
    for h in use_horizons:
        model_path = models_root / f"h{h.name}" / "model.lgb"
        meta_path = models_root / f"h{h.name}" / "meta.json"
        if not model_path.exists() or not meta_path.exists():
            log.warning(
                "backtest: missing model for h%s at %s, skipping",
                h.name, model_path,
            )
            continue
        booster = lgb.Booster(model_file=str(model_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feat_names = list(meta.get("feature_names") or meta.get("features") or feats)

        log.info("backtest: horizon=%s features=%d", h.name, len(feat_names))
        result = _backtest_horizon(df_test, h, feat_names, booster, target_trade_frac)
        (out_dir / f"h{h.name}.json").write_text(json.dumps(result, indent=2))
        summary["horizons"][h.name] = {
            "n_trades": result.get("n_trades", 0),
            "scenarios": {
                label: {k: v for k, v in s.items() if k in ("avg_net_bp", "hit_rate", "sharpe", "fill_rate", "n_filled")}
                for label, s in result.get("scenarios", {}).items()
            },
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _print_summary(summary: dict) -> None:
    print()
    print("=========== BACKTEST SUMMARY ===========")
    for hname, h in summary["horizons"].items():
        print(f"  horizon={hname:>3}  n_trades={h.get('n_trades', 0):>7}")
        for label, s in h["scenarios"].items():
            avg = s.get("avg_net_bp", 0.0)
            hit = s.get("hit_rate", 0.0)
            shp = s.get("sharpe", 0.0)
            fr = s.get("fill_rate", 0.0)
            nf = s.get("n_filled", 0)
            print(
                f"    {label:<12} avg_net={avg:+7.2f}bp  hit={hit:.3f}  sharpe={shp:+6.2f}"
                f"  fill_rate={fr:.3f}  n_filled={nf}"
            )
    print("========================================")
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=Path("./models/global"))
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--horizons", nargs="*", default=None)
    parser.add_argument("--target-trade-frac", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
    )

    if args.data_dir is None:
        from backend.config import settings
        args.data_dir = settings.data_dir

    summary = run_backtest(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        horizons=args.horizons,
        target_trade_frac=args.target_trade_frac,
        out_dir=args.out_dir,
        symbols=args.symbols,
    )
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
