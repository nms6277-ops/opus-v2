import numpy as np
import polars as pl

from backend.ml.dataset import load_dataset
from backend.ml.labels import HorizonSpec
from backend.ml.train import _compact_training_frame, _xy


def test_xy_uses_float32_feature_matrix():
    df = pl.DataFrame(
        {
            "part": ["train", "train"],
            "y_1s_valid": [True, True],
            "y_1s": [0, 2],
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
        }
    )

    X, y, w = _xy(df, ["feature_a", "feature_b"], "1s", use_symbol_feature=False)

    assert X.dtype == np.float32
    assert y.dtype == np.int8
    assert w.dtype == np.float32


def test_compact_training_frame_keeps_only_training_and_eval_columns():
    df = pl.DataFrame(
        {
            "symbol": ["AAAUSDT"],
            "part": ["train"],
            "best_bid": [1.0],
            "best_ask": [1.1],
            "spread_bp": [50.0],
            "bid_p_00": [1.0],
            "ask_p_00": [1.1],
            "bid_q_00": [10.0],
            "ask_q_00": [11.0],
            "derived_feature": [0.5],
            "ret_1s_bp": [2.0],
            "gross_long_1s_bp": [-3.0],
            "gross_short_1s_bp": [-3.0],
            "y_1s_valid": [True],
            "y_1s": [2],
        }
    )

    out = _compact_training_frame(
        df,
        feats=["bid_q_00", "ask_q_00", "derived_feature"],
        horizons=(HorizonSpec("1s", 1000, 500),),
    )

    # Schema (in order): identifiers, taker-fill price columns, features,
    # then per-horizon return/gross/y columns. Top-N raw price columns
    # (``bid_p_00`` / ``ask_p_00``) are intentionally dropped.
    assert out.columns == [
        "symbol",
        "part",
        "best_bid",
        "best_ask",
        "spread_bp",
        "bid_q_00",
        "ask_q_00",
        "derived_feature",
        "ret_1s_bp",
        "gross_long_1s_bp",
        "gross_short_1s_bp",
        "y_1s_valid",
        "y_1s",
    ]


def test_load_dataset_can_cap_rows_per_symbol(tmp_path):
    snap_dir = tmp_path / "snapshots" / "AAAUSDT" / "2026-05-01"
    snap_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "ts_ms": list(range(10)),
            "symbol": ["AAAUSDT"] * 10,
            "mid": [1.0] * 10,
        }
    ).write_parquet(snap_dir / "00.parquet")

    df = load_dataset(tmp_path, max_rows_per_symbol=4)

    assert df.height == 4
    assert df["ts_ms"].to_list() == sorted(df["ts_ms"].to_list())
    assert set(df["part"].to_list()) <= {"train", "val", "test"}
