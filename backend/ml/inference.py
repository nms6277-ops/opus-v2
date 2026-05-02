"""Online inference: load a trained LightGBM bundle and predict on live snapshots.

Training (:mod:`backend.ml.train`) ingests a Polars DataFrame where each row
is a snapshot and every "derived" feature (rolling vol, OFI, lag returns,
bucket imbalances, ...) is pre-computed via :mod:`backend.ml.features`.
At inference time we receive exactly **one** snapshot dict at a time from
the snapshot loop — so we need to maintain a short per-symbol history of
past snapshots, recompute the derived features, and feed the feature
vector to LightGBM.

We do this with a numpy-backed ring buffer per symbol. Only the columns
needed to derive features are stored; the static columns come straight
from the current snapshot row. The buffer is deliberately small (a few
hundred rows × a dozen columns) so the full derivation runs in a few
hundred microseconds, well under the 250 ms snapshot cadence.

The predictor is a read-only consumer of the collector. It is safe to
construct it even when no model is configured — every ``predict`` call
then simply returns ``None``.

API surface is small and explicit:

- ``Predictor(...)``: constructed once at runtime startup.
- ``.horizons``: list of horizon names (e.g. ``["1s", "5s", "30s"]``).
- ``.predict(symbol, snap_row)``: returns a :class:`Prediction` (or None
  if the symbol is cold, the model failed to load, or the feature vector
  contained non-finite values).
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("backend.ml.inference")

# Number of past snapshots we must remember to compute every derived feature.
# Must exceed ``max(LAG_STEPS)`` from :mod:`backend.ml.features` (480). We add
# margin so ``rolling_std`` windows anchored at the latest row are fully
# populated.
HISTORY_CAPACITY = 600

# Columns from the snapshot row that feed into derived-feature computations.
# Must stay in sync with :func:`backend.ml.features._derived_for_one_symbol`.
_HISTORY_COLS: tuple[str, ...] = (
    "mid",
    "microprice",
    "best_bid",
    "best_ask",
    "best_bid_qty",
    "best_ask_qty",
    "spread_bp",
    "buy_volume_win",
    "sell_volume_win",
)

LAG_STEPS = (1, 4, 20, 120, 480)
ROLL_WINDOWS = (4, 20, 120)


@dataclass
class Prediction:
    """Single-horizon prediction for one snapshot."""

    horizon: str
    p_down: float
    p_flat: float
    p_up: float
    confidence: float  # P(UP) - P(DOWN), in [-1, 1]


@dataclass
class SymbolHistory:
    """Rolling per-symbol history of recent snapshot columns."""

    capacity: int
    # Each deque stores float scalars; we keep the latest ``capacity`` values.
    cols: dict[str, deque[float]] = field(default_factory=dict)
    last_ts_ms: int = 0

    def __post_init__(self) -> None:
        if not self.cols:
            self.cols = {c: deque(maxlen=self.capacity) for c in _HISTORY_COLS}

    def push(self, snap: dict[str, Any]) -> None:
        for c in _HISTORY_COLS:
            self.cols[c].append(float(snap.get(c, 0.0)))
        self.last_ts_ms = int(snap.get("ts_ms", 0))

    @property
    def size(self) -> int:
        return len(self.cols["mid"])

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {c: np.asarray(q, dtype=np.float64) for c, q in self.cols.items()}


def _rolling_std(arr: np.ndarray, window: int) -> float:
    """Standard deviation of the last ``window`` values (ddof=1)."""
    if arr.size < window:
        return float("nan")
    sub = arr[-window:]
    m = sub.mean()
    var = ((sub - m) ** 2).sum() / (window - 1)
    return float(np.sqrt(var)) if var >= 0 else float("nan")


def _rolling_sum(arr: np.ndarray, window: int) -> float:
    if arr.size < window:
        return float("nan")
    return float(arr[-window:].sum())


def _rolling_median(arr: np.ndarray, window: int) -> float:
    if arr.size < window:
        return float("nan")
    return float(np.median(arr[-window:]))


def _derive_row(hist: SymbolHistory, snap: dict[str, Any]) -> dict[str, float]:
    """Compute the latest row's derived features using past-history + current snap.

    Mirrors :func:`backend.ml.features._derived_for_one_symbol` one row at a time.
    Returns only the derived columns; static columns are read straight from
    ``snap`` by :meth:`Predictor._build_feature_vector`.
    """
    eps = 1e-9

    arrs = hist.as_arrays()
    mid = arrs["mid"]
    microprice = arrs["microprice"]
    bb = arrs["best_bid"]
    ba = arrs["best_ask"]
    bb_q = arrs["best_bid_qty"]
    ba_q = arrs["best_ask_qty"]
    spread = arrs["spread_bp"]
    buy_v = arrs["buy_volume_win"]
    sell_v = arrs["sell_volume_win"]

    # Guard against log/divide-by-zero: clip mid to a tiny positive floor.
    mid_safe = np.where(mid > 1e-12, mid, 1e-12)

    out: dict[str, float] = {}

    # Lag returns (bp) at each LAG_STEPS[k]. Polars shift(k) at row i reads
    # mid[i - k]; numpy equivalent is mid[-1-k] relative to the end.
    cur_mid = float(mid_safe[-1])
    for lag in LAG_STEPS:
        if mid.size > lag:
            prev_mid = float(mid_safe[-1 - lag])
            out[f"mid_ret_lag{lag}_bp"] = (cur_mid - prev_mid) / prev_mid * 10_000.0
        else:
            out[f"mid_ret_lag{lag}_bp"] = float("nan")

    # 1-step returns for rolling vol (we need at least 2 points for a single return).
    if mid.size >= 2:
        prev_all = np.concatenate(([mid_safe[0]], mid_safe[:-1]))
        ret_1 = (mid_safe - prev_all) / prev_all * 10_000.0
    else:
        ret_1 = np.zeros(mid.size)
    for w in ROLL_WINDOWS:
        out[f"vol_w{w}_bp"] = _rolling_std(ret_1, w)

    # microprice deviation from mid
    if cur_mid > 0:
        out["microprice_dev_bp"] = (float(microprice[-1]) - float(mid[-1])) / cur_mid * 10_000.0
    else:
        out["microprice_dev_bp"] = 0.0

    # OFI proxy at top-of-book
    if bb.size >= 2:
        bb_prev = float(bb[-2])
        ba_prev = float(ba[-2])
        bb_q_prev = float(bb_q[-2])
        ba_q_prev = float(ba_q[-2])
        bb_cur = float(bb[-1])
        ba_cur = float(ba[-1])
        bb_q_cur = float(bb_q[-1])
        ba_q_cur = float(ba_q[-1])

        if bb_cur == bb_prev:
            bid_ofi = bb_q_cur - bb_q_prev
        elif bb_cur > bb_prev:
            bid_ofi = bb_q_cur
        else:
            bid_ofi = -bb_q_prev

        if ba_cur == ba_prev:
            ask_ofi = -(ba_q_cur - ba_q_prev)
        elif ba_cur < ba_prev:
            ask_ofi = -ba_q_cur
        else:
            ask_ofi = ba_q_prev

        ofi_cur = bid_ofi + ask_ofi
    else:
        ofi_cur = 0.0
    out["ofi_top1"] = float(ofi_cur)

    # Rolling OFI requires the full OFI series; we recompute it fast in numpy.
    if bb.size >= 2:
        bb_prev_arr = np.concatenate(([bb[0]], bb[:-1]))
        ba_prev_arr = np.concatenate(([ba[0]], ba[:-1]))
        bb_q_prev_arr = np.concatenate(([bb_q[0]], bb_q[:-1]))
        ba_q_prev_arr = np.concatenate(([ba_q[0]], ba_q[:-1]))

        bid_ofi_arr = np.where(
            bb == bb_prev_arr,
            bb_q - bb_q_prev_arr,
            np.where(bb > bb_prev_arr, bb_q, -bb_q_prev_arr),
        )
        ask_ofi_arr = np.where(
            ba == ba_prev_arr,
            -(ba_q - ba_q_prev_arr),
            np.where(ba < ba_prev_arr, -ba_q, ba_q_prev_arr),
        )
        ofi_arr = bid_ofi_arr + ask_ofi_arr
    else:
        ofi_arr = np.zeros(bb.size)
    for w in ROLL_WINDOWS:
        out[f"ofi_top1_sum_w{w}"] = _rolling_sum(ofi_arr, w)

    # Spread normalisation
    cur_spread = float(spread[-1])
    for w in ROLL_WINDOWS:
        med = _rolling_median(spread, w)
        if med is not None and med == med and med > 1e-6:
            out[f"spread_norm_w{w}"] = cur_spread / med
        else:
            out[f"spread_norm_w{w}"] = float("nan")

    # Bucket concentrations & imbalances — these rely only on the current
    # snapshot, not on history. Caller passes them via ``snap``.
    b5 = float(snap.get("bid_bkt_qty_05bp", 0.0))
    a5 = float(snap.get("ask_bkt_qty_05bp", 0.0))
    b10 = float(snap.get("bid_bkt_qty_10bp", 0.0))
    a10 = float(snap.get("ask_bkt_qty_10bp", 0.0))
    b25 = float(snap.get("bid_bkt_qty_25bp", 0.0))
    a25 = float(snap.get("ask_bkt_qty_25bp", 0.0))
    b50 = float(snap.get("bid_bkt_qty_50bp", 0.0))
    a50 = float(snap.get("ask_bkt_qty_50bp", 0.0))

    denom_50 = (b50 + a50) + eps
    out["bkt_concentration_5_50"] = (b5 + a5) / denom_50
    out["bkt_concentration_10_50"] = (b10 + a10) / denom_50

    for bp_val, b, a in ((5, b5, a5), (10, b10, a10), (25, b25, a25), (50, b50, a50)):
        out[f"bkt_imb_{bp_val:02d}bp"] = (b - a) / (b + a + eps)

    # Trade-flow imbalance (instantaneous + rolling)
    cur_buy = float(buy_v[-1])
    cur_sell = float(sell_v[-1])
    out["trade_imb_win"] = (cur_buy - cur_sell) / (cur_buy + cur_sell + eps)
    for w in ROLL_WINDOWS:
        bvw = _rolling_sum(buy_v, w)
        svw = _rolling_sum(sell_v, w)
        if bvw != bvw or svw != svw:  # NaN
            out[f"trade_imb_w{w}"] = float("nan")
        else:
            out[f"trade_imb_w{w}"] = (bvw - svw) / (bvw + svw + eps)

    # Log book volume
    bv20 = float(snap.get("bid_vol_top20", 0.0))
    av20 = float(snap.get("ask_vol_top20", 0.0))
    total_book = bv20 + av20
    out["log_book_vol_top20"] = float(np.log(total_book)) if total_book > eps else float("nan")

    # VWAP deviation from mid
    vwap = float(snap.get("vwap_win", 0.0))
    if vwap > 0 and cur_mid > 0:
        out["vwap_dev_bp"] = (vwap - float(snap.get("mid", 0.0))) / cur_mid * 10_000.0
    else:
        out["vwap_dev_bp"] = 0.0

    return out


class Predictor:
    """Loads one LightGBM model per horizon and runs predictions on live snapshots."""

    def __init__(self, model_dir: Path, horizons: tuple[str, ...] | None = None):
        self.model_dir = Path(model_dir)
        self._histories: dict[str, SymbolHistory] = {}
        self._boosters: dict[str, Any] = {}
        self._features: dict[str, list[str]] = {}
        self._symbols: dict[str, list[str]] = {}
        self._use_sym: dict[str, bool] = {}
        self._latest_derived: dict[str, dict[str, float]] = {}
        self._cold_warned: set[str] = set()

        if not self.model_dir.exists():
            log.warning("inference: model_dir %s does not exist; predictor disabled", self.model_dir)
            self._available_horizons: list[str] = []
            return

        # Auto-resolve ``models/something`` -> ``models/something/global``
        # if the latter is where train.py actually wrote the artefacts.
        # This lets ``OPUS_MODEL_DIR=./models/agnostic`` Just Work without
        # the user having to remember to append ``/global``.
        global_root = self.model_dir / "global"
        if global_root.is_dir() and any(
            (global_root / p.name).is_dir() and (global_root / p.name / "model.lgb").exists()
            for p in global_root.iterdir() if p.name.startswith("h")
        ):
            log.info(
                "inference: resolved model_dir %s -> %s (auto-detected global/ subfolder)",
                self.model_dir, global_root,
            )
            self.model_dir = global_root

        # Auto-discover horizons from sub-dirs if none specified.
        if horizons is None:
            horizons = tuple(
                p.name[1:] for p in sorted(self.model_dir.iterdir())
                if p.is_dir() and p.name.startswith("h")
            )
        self._available_horizons = []
        for h in horizons:
            if self._load_horizon(h):
                self._available_horizons.append(h)

        if not self._available_horizons:
            log.warning("inference: no horizons loaded from %s; predictor disabled", self.model_dir)
        else:
            log.info("inference: loaded horizons=%s from %s", self._available_horizons, self.model_dir)

    @property
    def horizons(self) -> list[str]:
        return list(self._available_horizons)

    @property
    def enabled(self) -> bool:
        return bool(self._available_horizons)

    def latest_derived(self, symbol: str) -> dict[str, float]:
        return dict(self._latest_derived.get(symbol, {}))

    def _load_horizon(self, horizon: str) -> bool:
        model_path = self.model_dir / f"h{horizon}" / "model.lgb"
        meta_path = self.model_dir / f"h{horizon}" / "meta.json"
        if not model_path.exists() or not meta_path.exists():
            log.warning("inference: h%s missing model.lgb or meta.json at %s", horizon, model_path.parent)
            return False
        try:
            import lightgbm as lgb  # noqa: PLC0415  (deferred import)
        except ImportError as e:
            log.warning("inference: lightgbm not installed (%s); cannot load models", e)
            return False

        try:
            booster = lgb.Booster(model_file=str(model_path))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.error("inference: failed to load h%s: %s", horizon, e)
            return False

        # ``feature_names`` is the canonical key written by :mod:`backend.ml.train`;
        # ``features`` is accepted as a fallback for older meta dumps.
        feat_names = meta.get("feature_names") or meta.get("features") or []
        self._boosters[horizon] = booster
        self._features[horizon] = list(feat_names)
        self._symbols[horizon] = list(meta.get("symbols", []))
        self._use_sym[horizon] = bool(meta.get("use_symbol_feature", True))
        return True

    def _symbol_id(self, horizon: str, symbol: str) -> int:
        syms = self._symbols.get(horizon, [])
        try:
            return syms.index(symbol)
        except ValueError:
            return -1  # unknown — LightGBM tolerates missing categorical codes

    def _build_feature_vector(
        self, horizon: str, symbol: str, snap: dict[str, Any], derived: dict[str, float]
    ) -> np.ndarray | None:
        feats = self._features.get(horizon)
        if not feats:
            return None
        row = np.empty(len(feats), dtype=np.float64)
        for i, name in enumerate(feats):
            if name == "_symbol_id":
                row[i] = float(self._symbol_id(horizon, symbol))
            elif name in derived:
                row[i] = derived[name]
            else:
                row[i] = float(snap.get(name, 0.0))
        if not np.isfinite(row).all():
            return None
        return row.reshape(1, -1)

    def update(self, symbol: str, snap: dict[str, Any]) -> None:
        """Push a fresh snapshot into the history buffer without predicting.

        Useful for priming the buffer during the first few seconds after
        startup, before enough history exists to generate features.
        """
        hist = self._histories.get(symbol)
        if hist is None:
            hist = SymbolHistory(capacity=HISTORY_CAPACITY)
            self._histories[symbol] = hist
        hist.push(snap)

    def predict(self, symbol: str, snap: dict[str, Any]) -> dict[str, Prediction] | None:
        """Update per-symbol history and return predictions for every horizon.

        Returns ``None`` when:
        - the predictor is disabled (no models loaded)
        - history is too short to compute the longest lag feature
        - the resulting feature vector has any non-finite value
        """
        self.update(symbol, snap)
        if not self._available_horizons:
            return None

        hist = self._histories[symbol]
        if hist.size < max(LAG_STEPS) + 2:
            if symbol not in self._cold_warned:
                log.info("inference: %s warming up (%d / %d rows)",
                         symbol, hist.size, max(LAG_STEPS) + 2)
                self._cold_warned.add(symbol)
            return None

        derived = _derive_row(hist, snap)
        self._latest_derived[symbol] = derived
        preds: dict[str, Prediction] = {}
        for horizon in self._available_horizons:
            X = self._build_feature_vector(horizon, symbol, snap, derived)
            if X is None:
                continue
            booster = self._boosters[horizon]
            try:
                proba = booster.predict(X, num_iteration=booster.best_iteration or None)
            except Exception as e:
                log.error("inference: predict h%s %s failed: %s", horizon, symbol, e)
                continue
            # booster outputs shape (1, 3): [p_down, p_flat, p_up]
            arr = np.asarray(proba).reshape(-1)
            if arr.size != 3:
                continue
            p_down, p_flat, p_up = float(arr[0]), float(arr[1]), float(arr[2])
            preds[horizon] = Prediction(
                horizon=horizon,
                p_down=p_down,
                p_flat=p_flat,
                p_up=p_up,
                confidence=p_up - p_down,
            )
        return preds or None


__all__ = ["HISTORY_CAPACITY", "Prediction", "Predictor"]
