"""Train LightGBM models, one per horizon.

The script is invoked as ``python -m backend.ml.train`` (see also
``scripts/train.ps1`` for a Windows-friendly wrapper). It does, in
order:

1. Discover and load all parquet snapshots from ``OPUS_DATA_DIR``.
2. Add derived features and 3-class direction labels at
   1s / 5s / 30s horizons.
3. Train one LightGBM multi-class classifier per horizon. By default
   ``symbol`` is added as a categorical feature so a single global
   model handles all symbols (the alternative — per-symbol models —
   is supported via ``--per-symbol``).
4. Save model + feature list + label thresholds to
   ``{models_dir}/h{horizon}/`` along with a JSON metadata blob.

Hyperparameter search via Optuna is optional (``--optuna-trials N``).
With N=0 (default) we use a sensible fixed config; the MVP-1 goal is
proving the pipeline works, not squeezing the last 0.5% AUC.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

from backend.ml.dataset import load_dataset
from backend.ml.eval import evaluate_holdout
from backend.ml.features import add_features, feature_columns
from backend.ml.labels import (
    DEFAULT_HORIZONS,
    HorizonSpec,
    add_labels,
    parse_horizon_spec,
)


def _parse_horizon(spec: str) -> HorizonSpec:
    """CLI-friendly wrapper around :func:`parse_horizon_spec`.

    Converts ``ValueError`` to ``SystemExit`` so argparse-style errors
    surface nicely from the CLI without a traceback.
    """
    try:
        return parse_horizon_spec(spec)
    except ValueError as e:
        raise SystemExit(str(e)) from e


log = logging.getLogger("backend.ml.train")


DEFAULT_LGB_PARAMS = dict(
    objective="multiclass",
    num_class=3,
    metric="multi_logloss",
    learning_rate=0.05,
    num_leaves=63,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    min_data_in_leaf=200,
    lambda_l2=0.1,
    verbose=-1,
    num_threads=0,  # use all available
)


def _import_lgb():
    """Defer the import so ``backend.ml`` doesn't pull lightgbm at module load.

    LightGBM is an optional dep installed via ``pip install -e .[ml]``.
    """
    try:
        import lightgbm as lgb  # noqa: WPS433
    except ImportError as e:
        raise RuntimeError("LightGBM is not installed. Run: pip install -e .[ml]") from e
    return lgb


def _xy(
    df: pl.DataFrame,
    feats: list[str],
    horizon_name: str,
    *,
    use_symbol_feature: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (X, y, weights) numpy arrays for a slice of the frame."""
    valid_col = f"y_{horizon_name}_valid"
    y_col = f"y_{horizon_name}"
    df_valid = df.filter(pl.col(valid_col))
    if df_valid.height == 0:
        raise RuntimeError(f"no valid rows for horizon {horizon_name}")

    X_cols: list[str] = list(feats)
    if use_symbol_feature:
        X_cols = ["_symbol_id"] + X_cols
    X = df_valid.select(X_cols).to_numpy().astype(np.float32, copy=False)
    # ``LGBM_DatasetCreateFromMat`` reads the matrix as row-major. polars'
    # ``.to_numpy()`` for multi-column frames sometimes returns Fortran-
    # ordered (column-major) on Windows, which causes the C side to walk
    # past the allocated buffer and crash with a NULL-pointer access
    # violation. Force C-contiguous before passing to LightGBM. ``np.
    # ascontiguousarray`` is a no-op when the array is already C-order.
    X = np.ascontiguousarray(X)
    # NaN / Inf in features cause access-violation crashes in some
    # LightGBM Windows builds (instead of being silently bin-encoded as
    # missing). Replace with finite sentinels: NaN -> 0.0 (LightGBM
    # treats it as "missing" later via use_missing=True anyway), +/-Inf
    # capped at the float32 representable range. Hot path is unchanged
    # (no pandas, no per-row Python).
    if not np.all(np.isfinite(X)):
        np.nan_to_num(X, copy=False, nan=0.0, posinf=3.4e38, neginf=-3.4e38)
    y = df_valid[y_col].to_numpy().astype(np.int8, copy=False)

    # Class-balanced weights (helps when FLAT dominates after threshold).
    w = np.ones_like(y, dtype=np.float32)
    counts = np.bincount(y, minlength=3).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = counts.sum() / (counts * 3.0)
    for cls in (0, 1, 2):
        w[y == cls] = inv[cls]
    return X, y, w


def _add_symbol_id(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Add ``_symbol_id`` int column and return (df, ordered_symbols)."""
    symbols = sorted(df["symbol"].unique().to_list())
    sym_to_id = {s: i for i, s in enumerate(symbols)}
    df = df.with_columns(
        pl.col("symbol").replace_strict(sym_to_id, return_dtype=pl.Int32).alias("_symbol_id")
    )
    return df, symbols


def _compact_training_frame(
    df: pl.DataFrame,
    feats: list[str],
    horizons: tuple[HorizonSpec, ...],
) -> pl.DataFrame:
    """Drop columns no longer needed after feature/label construction.

    The raw snapshot frame contains top-level prices and intermediate columns
    that are useful for deriving features and labels, but not for LightGBM
    training or holdout evaluation. Removing them before converting slices to
    numpy keeps peak RAM materially lower on small Windows boxes.
    """
    # ``best_bid``/``best_ask``/``spread_bp`` are required by the v2 honest
    # backtest (taker bid-ask round-trip fills); ``gross_long_*``/``gross_short_*``
    # let backtest reuse the exact same gross-PnL expression as labels did.
    keep: list[str] = ["symbol", "part", "best_bid", "best_ask", "spread_bp"]
    keep.extend(feats)
    for h in horizons:
        keep.extend(
            (
                f"ret_{h.name}_bp",
                f"gross_long_{h.name}_bp",
                f"gross_short_{h.name}_bp",
                f"y_{h.name}_valid",
                f"y_{h.name}",
            )
        )

    seen: set[str] = set()
    existing = []
    for col in keep:
        if col in df.columns and col not in seen:
            existing.append(col)
            seen.add(col)

    cast_exprs = [
        pl.col(col).cast(pl.Float32) for col in feats if col in df.columns and df[col].dtype.is_numeric()
    ]
    if cast_exprs:
        df = df.with_columns(cast_exprs)
    return df.select(existing)


def _train_one_horizon(
    df_lab: pl.DataFrame,
    horizon: HorizonSpec,
    feats: list[str],
    *,
    out_dir: Path,
    thresholds: dict[tuple[str, str], float],
    symbols: list[str],
    optuna_trials: int = 0,
    early_stopping_rounds: int = 50,
    num_boost_round: int = 800,
    use_symbol_feature: bool | None = None,
) -> dict:
    """Train one model for one horizon. Returns a metrics dict.

    ``use_symbol_feature``:
        ``None`` (default) — auto: enable iff training on >1 symbol.
        ``True``           — force enable. Predictions will be skewed for
                             symbols not seen during training.
        ``False``          — force disable. Model becomes purely
                             microstructure-based and **transfers** to
                             unseen symbols. AUC may drop ~1-3 points
                             compared to the symbol-aware variant.
    """
    lgb = _import_lgb()

    if use_symbol_feature is None:
        use_symbol_feature = len(symbols) > 1
    X_train, y_train, w_train = _xy(
        df_lab.filter(pl.col("part") == "train"),
        feats,
        horizon.name,
        use_symbol_feature=use_symbol_feature,
    )
    X_val, y_val, w_val = _xy(
        df_lab.filter(pl.col("part") == "val"),
        feats,
        horizon.name,
        use_symbol_feature=use_symbol_feature,
    )

    lgb_feats = (["_symbol_id"] if use_symbol_feature else []) + feats
    cat_features = ["_symbol_id"] if use_symbol_feature else "auto"
    train_set = lgb.Dataset(
        X_train,
        y_train,
        weight=w_train,
        feature_name=lgb_feats,
        categorical_feature=cat_features,
    )
    val_set = lgb.Dataset(
        X_val,
        y_val,
        weight=w_val,
        feature_name=lgb_feats,
        categorical_feature=cat_features,
        reference=train_set,
    )

    params = dict(DEFAULT_LGB_PARAMS)
    if optuna_trials > 0:
        params = _optuna_search(
            lgb,
            X_train,
            y_train,
            w_train,
            X_val,
            y_val,
            w_val,
            lgb_feats,
            cat_features,
            optuna_trials,
        )
        log.info("train: optuna best params %s", params)

    log.info(
        "train: horizon=%s  X_train=%s  X_val=%s  feats=%d  use_sym=%s",
        horizon.name,
        X_train.shape,
        X_val.shape,
        len(lgb_feats),
        use_symbol_feature,
    )
    t0 = time.perf_counter()
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    dt = time.perf_counter() - t0
    log.info("train: horizon=%s trained %d trees in %.1fs", horizon.name, booster.current_iteration(), dt)

    # Save artefacts.
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.lgb"
    booster.save_model(str(model_path))

    meta = {
        "horizon_name": horizon.name,
        "horizon_ms": horizon.ms,
        "tolerance_ms": horizon.tolerance_ms,
        "feature_names": lgb_feats,
        "use_symbol_feature": use_symbol_feature,
        "symbols": symbols,
        "thresholds_bp": {f"{s}|{h}": v for (s, h), v in thresholds.items() if h == horizon.name},
        "lgb_params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in params.items()},
        # Iteration with the best validation score (early-stopping winner). Always
        # <= booster.current_iteration(); the latter counts the extra rounds run
        # past the best before stopping. Inference loads model.lgb which embeds
        # the same best_iteration, so this is purely metadata for humans.
        "best_iteration": booster.best_iteration,
        "trained_at": time.time(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Holdout evaluation on the test slice — written next to the model.
    eval_report = evaluate_holdout(
        booster,
        df_lab,
        feats,
        horizon,
        symbols,
        use_symbol_feature,
    )
    (out_dir / "eval.json").write_text(json.dumps(eval_report, indent=2))

    fi = booster.feature_importance(importance_type="gain")
    fi_pairs = sorted(zip(lgb_feats, fi.tolist(), strict=True), key=lambda x: -x[1])[:30]
    (out_dir / "feature_importance.json").write_text(json.dumps(fi_pairs, indent=2))
    log.info(
        "train: horizon=%s top-5 features by gain: %s",
        horizon.name,
        [(n, round(g, 1)) for n, g in fi_pairs[:5]],
    )

    return {
        "horizon": horizon.name,
        "best_iteration": booster.best_iteration,
        "eval": eval_report,
    }


def _optuna_search(
    lgb,
    X_train,
    y_train,
    w_train,
    X_val,
    y_val,
    w_val,
    lgb_feats,
    cat_features,
    trials: int,
) -> dict:
    """Tiny Optuna budget for MVP. Tunes a handful of hyperparams."""
    try:
        import optuna  # noqa: WPS433
    except ImportError as e:
        raise RuntimeError("Optuna requested but not installed (pip install -e .[ml]).") from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_set = lgb.Dataset(
        X_train, y_train, weight=w_train, feature_name=lgb_feats, categorical_feature=cat_features
    )
    val_set = lgb.Dataset(
        X_val,
        y_val,
        weight=w_val,
        feature_name=lgb_feats,
        categorical_feature=cat_features,
        reference=train_set,
    )

    def objective(trial) -> float:
        params = dict(DEFAULT_LGB_PARAMS)
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.02, 0.10, log=True)
        params["num_leaves"] = trial.suggest_int("num_leaves", 31, 127)
        params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.6, 1.0)
        params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.6, 1.0)
        params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 50, 1000, log=True)
        params["lambda_l2"] = trial.suggest_float("lambda_l2", 0.0, 1.0)
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=300,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        return float(booster.best_score["val"]["multi_logloss"])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = dict(DEFAULT_LGB_PARAMS)
    best.update(study.best_params)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Train LightGBM models for opus MVP-1")
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="OPUS_DATA_DIR (defaults to env / settings)"
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("./models"), help="output dir for trained artefacts"
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help=(
            "restrict to these symbols (default: all found, minus the "
            "settings.train_symbol_blocklist - typically BTC/ETH whose "
            "sub-bp spreads make spread-aware edge undetectable)"
        ),
    )
    parser.add_argument(
        "--include-blocklisted",
        action="store_true",
        help=(
            "override settings.train_symbol_blocklist; only meaningful when "
            "--symbols is omitted, otherwise the explicit list always wins"
        ),
    )
    parser.add_argument(
        "--horizons",
        nargs="*",
        default=None,
        help=(
            "custom horizons to train. Accepts any duration "
            "ending in ms/s/m, e.g. '500ms 2s 5s 15s 1m'. "
            "If omitted, trains the defaults: 1s 5s 30s."
        ),
    )
    parser.add_argument(
        "--from-date", type=str, default=None, help="only use parquet files dated on/after this YYYY-MM-DD"
    )
    parser.add_argument(
        "--to-date", type=str, default=None, help="only use parquet files dated on/before this YYYY-MM-DD"
    )
    parser.add_argument(
        "--max-rows-per-symbol",
        type=int,
        default=None,
        help=(
            "cap loaded rows per symbol by taking an even time-ordered "
            "sample; useful for low-RAM training machines"
        ),
    )
    parser.add_argument(
        "--window",
        type=str,
        default=None,
        help=(
            "in-play short-cycle filter: only train on the last N of data, "
            "e.g. --window 4h. Cutoff uses the parquet's most recent ts_ms "
            "per symbol, so the call is deterministic across re-runs."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("per-symbol", "global"),
        default="per-symbol",
        help=(
            "per-symbol (default) trains one model per coin under "
            "models/per_symbol/{SYM}/h{H}; global pools every symbol into "
            "one model under models/global/h{H} (use for ablation only)"
        ),
    )
    parser.add_argument(
        "--per-symbol",
        action="store_true",
        help="alias for --mode=per-symbol (kept for back-compat)",
    )
    parser.add_argument(
        "--global",
        dest="global_mode",
        action="store_true",
        help="alias for --mode=global",
    )
    parser.add_argument(
        "--no-symbol-feature",
        action="store_true",
        help=(
            "drop the categorical _symbol_id feature from "
            "training so the model uses only microstructure "
            "signals. Resulting model TRANSFERS to symbols "
            "it has never seen during training (e.g. coins "
            "newly listed after data collection). AUC may "
            "drop 1-3 points vs the symbol-aware variant."
        ),
    )
    parser.add_argument(
        "--optuna-trials", type=int, default=0, help="number of Optuna trials per horizon (0 = skip search)"
    )
    parser.add_argument("--rounds", type=int, default=800)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
    )

    # Resolve mode: --global / --per-symbol > --mode (default per-symbol).
    if args.global_mode and args.per_symbol:
        parser.error("--global and --per-symbol are mutually exclusive")
    if args.global_mode:
        args.mode = "global"
    elif args.per_symbol:
        args.mode = "per-symbol"
    log.info("train: mode=%s", args.mode)

    # Apply default blocklist when the user didn't pass --symbols. We never
    # silently drop something the user explicitly asked for.
    from backend.config import settings as _settings

    if args.symbols is None and not args.include_blocklisted:
        blocked = tuple(s.upper() for s in _settings.train_symbol_blocklist)
        if blocked:
            log.info(
                "train: applying default blocklist %s (override with "
                "--include-blocklisted or pass --symbols explicitly)",
                blocked,
            )
            args._blocked = blocked
        else:
            args._blocked = ()
    else:
        args._blocked = ()

    if args.data_dir is None:
        from backend.config import settings

        args.data_dir = settings.data_dir
    log.info("train: data_dir=%s models_dir=%s", args.data_dir, args.models_dir)

    # 1. Load + split.
    df = load_dataset(
        args.data_dir,
        symbols=args.symbols,
        from_date=args.from_date,
        to_date=args.to_date,
        max_rows_per_symbol=args.max_rows_per_symbol,
        window=args.window,
    )
    if args._blocked:
        before = df.height
        df = df.filter(~pl.col("symbol").is_in(list(args._blocked)))
        log.info(
            "train: blocklist removed %d rows (kept %d); symbols left: %s",
            before - df.height,
            df.height,
            sorted(df["symbol"].unique().to_list()),
        )

    # 2. Decide horizons to train.
    if args.horizons:
        # Parse arbitrary user-provided horizons (e.g. "2s 5s 15s 1m"). If a
        # name happens to match a default, use the default's tolerance; for
        # genuinely new horizons, derive a reasonable tolerance automatically.
        defaults_by_name = {h.name.lower(): h for h in DEFAULT_HORIZONS}
        horizons_list = []
        for spec in args.horizons:
            key = spec.strip().lower()
            if key in defaults_by_name:
                horizons_list.append(defaults_by_name[key])
            else:
                horizons_list.append(_parse_horizon(spec))
        horizons = tuple(horizons_list)
        log.info("train: custom horizons = %s", [(h.name, h.ms) for h in horizons])
    else:
        horizons = DEFAULT_HORIZONS

    # 3. Features + labels (labels respect the chosen horizons).
    df, derived = add_features(df)
    feats = feature_columns(df, derived)
    df, thresholds = add_labels(df, horizons=horizons)
    df = _compact_training_frame(df, feats, horizons)
    gc.collect()

    # 3. Train per horizon (and optionally per symbol).
    # When --no-symbol-feature is set, force `use_symbol_feature=False` so
    # the resulting model transfers to symbols outside the training set.
    use_sym = False if args.no_symbol_feature else None
    if args.no_symbol_feature:
        log.info(
            "train: --no-symbol-feature set; _symbol_id will be DROPPED from "
            "training features so model transfers to unseen symbols."
        )

    results: list[dict] = []
    if args.mode == "per-symbol":
        for sym in sorted(df["symbol"].unique().to_list()):
            sym_df = df.filter(pl.col("symbol") == sym)
            sym_df, ordered_syms = _add_symbol_id(sym_df)
            for h in horizons:
                out_dir = args.models_dir / f"per_symbol/{sym}/h{h.name}"
                results.append(
                    _train_one_horizon(
                        sym_df,
                        h,
                        feats,
                        out_dir=out_dir,
                        thresholds=thresholds,
                        symbols=ordered_syms,
                        optuna_trials=args.optuna_trials,
                        early_stopping_rounds=args.early_stopping,
                        num_boost_round=args.rounds,
                        use_symbol_feature=use_sym,
                    )
                )
    else:
        df, ordered_syms = _add_symbol_id(df)
        for h in horizons:
            out_dir = args.models_dir / f"global/h{h.name}"
            results.append(
                _train_one_horizon(
                    df,
                    h,
                    feats,
                    out_dir=out_dir,
                    thresholds=thresholds,
                    symbols=ordered_syms,
                    optuna_trials=args.optuna_trials,
                    early_stopping_rounds=args.early_stopping,
                    num_boost_round=args.rounds,
                    use_symbol_feature=use_sym,
                )
            )

    # 4. Summary.
    summary_path = args.models_dir / "summary.json"
    args.models_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2))
    log.info("train: wrote summary -> %s", summary_path)

    print("\n=========== TRAIN SUMMARY ===========")
    for r in results:
        ev = r["eval"]
        print(
            f"  horizon={r['horizon']:>3}  "
            f"trees={r['best_iteration']:>4}  "
            f"AUC(UP-DOWN)={ev.get('auc_up_vs_down', float('nan')):.4f}  "
            f"hit@thr={ev.get('hit_rate_at_threshold', float('nan')):.3f}  "
            f"trades={ev.get('n_trades_at_threshold', 0):>6d}  "
            f"net_bp={ev.get('avg_net_bp_at_threshold', float('nan')):+.2f}  "
            f"sim_sharpe={ev.get('sim_sharpe_per_day', float('nan')):+.2f}"
        )
    print("=====================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
