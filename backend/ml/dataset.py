"""Snapshot parquet discovery + loading + time-aware split.

The collector writes one parquet per hour per symbol at
``{data_dir}/snapshots/{SYMBOL}/{YYYY-MM-DD}/{HH}.parquet``.

This module exposes two main entrypoints:

- :func:`discover` walks ``data_dir`` and returns a list of
  :class:`SymbolFiles` describing every (symbol, file) pair found,
  with the file's row count and timestamp range. Cheap — uses parquet
  metadata only, no full read.

- :func:`load_dataset` loads one or more symbols into a single
  ``polars.DataFrame``, sorted by ``(symbol, ts_ms)``, with a ``part``
  column added (``"train" | "val" | "test"``) according to time-based
  quantile splits.

We never shuffle. Time-based splits are mandatory in microstructure ML
to avoid look-ahead bias.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

log = logging.getLogger("backend.ml.dataset")


@dataclass(slots=True, frozen=True)
class SymbolFile:
    """One parquet file produced by the collector."""

    symbol: str
    path: Path
    rows: int
    ts_ms_min: int
    ts_ms_max: int


@dataclass(slots=True, frozen=True)
class SymbolFiles:
    """All parquet files we found for a single symbol."""

    symbol: str
    files: list[SymbolFile]

    @property
    def total_rows(self) -> int:
        return sum(f.rows for f in self.files)

    @property
    def ts_range(self) -> tuple[int, int]:
        if not self.files:
            return (0, 0)
        return (
            min(f.ts_ms_min for f in self.files),
            max(f.ts_ms_max for f in self.files),
        )


def _parquet_metadata(path: Path) -> tuple[int, int, int]:
    """Return ``(rows, ts_ms_min, ts_ms_max)`` for a single parquet file.

    Reads only the ``ts_ms`` column to compute the range — much cheaper
    than a full load on a multi-hundred-MB file.
    """
    try:
        meta = pq.read_metadata(path)
        rows = meta.num_rows
    except Exception as e:
        log.warning("dataset: failed to read metadata for %s: %s", path, e)
        return (0, 0, 0)
    if rows == 0:
        return (0, 0, 0)
    try:
        df = pl.read_parquet(path, columns=["ts_ms"], memory_map=False)
        if df.height == 0:
            return (rows, 0, 0)
        return (rows, int(df["ts_ms"].min()), int(df["ts_ms"].max()))
    except Exception as e:
        log.warning("dataset: failed to read ts_ms range for %s: %s", path, e)
        return (rows, 0, 0)


def discover(
    data_dir: Path,
    *,
    symbols: Iterable[str] | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> list[SymbolFiles]:
    """Walk ``data_dir/snapshots`` and list parquet files per symbol.

    Args:
        data_dir: ``OPUS_DATA_DIR`` (the dir containing ``snapshots/``).
        symbols: Optional whitelist; if None, all symbols found are used.
        from_date: Inclusive lower bound on the per-day folder name
            (``YYYY-MM-DD``). Folders with names lexicographically less
            than this string are skipped.
        to_date: Inclusive upper bound, same format.

    Returns:
        One :class:`SymbolFiles` per symbol, sorted alphabetically.
    """
    snapshots_root = Path(data_dir) / "snapshots"
    if not snapshots_root.is_dir():
        log.error("dataset: %s does not exist", snapshots_root)
        return []

    wanted = {s.upper() for s in symbols} if symbols else None
    out: dict[str, list[SymbolFile]] = {}
    for sym_dir in sorted(snapshots_root.iterdir()):
        if not sym_dir.is_dir():
            continue
        sym = sym_dir.name.upper()
        if wanted is not None and sym not in wanted:
            continue
        files: list[SymbolFile] = []
        for date_dir in sorted(sym_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            d = date_dir.name
            if from_date is not None and d < from_date:
                continue
            if to_date is not None and d > to_date:
                continue
            for parquet in sorted(date_dir.glob("*.parquet")):
                rows, lo, hi = _parquet_metadata(parquet)
                if rows == 0:
                    continue
                files.append(SymbolFile(sym, parquet, rows, lo, hi))
        if files:
            out[sym] = files

    return [SymbolFiles(sym, fs) for sym, fs in sorted(out.items())]


def _parse_window_ms(spec: str | None) -> int | None:
    """Parse a duration string like ``"4h"`` / ``"30m"`` / ``"15s"`` to ms.

    Returns ``None`` for ``None``/empty input. Used by the in-play ``--window``
    CLI flag so the caller can say "only the last 4 hours of data" without
    having to hand-compute timestamps.
    """
    if spec is None:
        return None
    s = spec.strip().lower()
    if not s:
        return None
    if s.endswith("ms"):
        return int(s[:-2])
    if s.endswith("h"):
        return int(float(s[:-1]) * 3600_000)
    if s.endswith("m"):
        return int(float(s[:-1]) * 60_000)
    if s.endswith("s"):
        return int(float(s[:-1]) * 1_000)
    raise ValueError(f"window '{spec}' must end in ms / s / m / h (e.g. 500ms, 30s, 1m, 4h)")


def load_dataset(
    data_dir: Path,
    *,
    symbols: Iterable[str] | None = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    from_date: str | None = None,
    to_date: str | None = None,
    max_rows_per_symbol: int | None = None,
    window: str | None = None,
    train_val_test_pct: tuple[float, float, float] | None = None,
) -> pl.DataFrame:
    """Load all snapshots for ``symbols`` and add a ``part`` column.

    The split is **per-symbol** and **time-based**: we sort by ``ts_ms``
    and assign the first ``train_frac`` fraction to ``"train"``, the
    next ``val_frac`` to ``"val"``, and the rest (``1 - train_frac -
    val_frac``) to ``"test"``. This guarantees we never train on data
    that comes after the validation/test windows.
    """
    if train_val_test_pct is not None:
        train_frac, val_frac, _test_frac = train_val_test_pct
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1.0:
        raise ValueError(
            f"invalid splits: train_frac={train_frac}, val_frac={val_frac}; must be positive and sum to <1"
        )

    window_ms = _parse_window_ms(window)
    inventory = discover(data_dir, symbols=symbols, from_date=from_date, to_date=to_date)
    if not inventory:
        raise RuntimeError(f"no parquet files found under {data_dir}/snapshots")

    frames: list[pl.DataFrame] = []
    for sf in inventory:
        log.info(
            "dataset: %s — %d files, %d rows, ts span=%s..%s",
            sf.symbol,
            len(sf.files),
            sf.total_rows,
            *sf.ts_range,
        )
        sym_df = pl.concat(
            [pl.read_parquet(f.path, memory_map=False) for f in sf.files],
            how="diagonal_relaxed",
        ).sort("ts_ms")

        if window_ms is not None and sym_df.height > 0:
            # In-play short-cycle workflow: only keep the last ``window``
            # of data, where the cutoff is the most recent ts_ms in this
            # symbol's parquet (NOT wall-clock now, so the call is
            # deterministic across reruns of the same data).
            latest_ms = int(sym_df["ts_ms"].max())
            cutoff_ms = latest_ms - window_ms
            before = sym_df.height
            sym_df = sym_df.filter(pl.col("ts_ms") >= cutoff_ms)
            log.info(
                "dataset: %s windowed last %s -> %d rows (was %d, cutoff_ms=%d)",
                sf.symbol,
                window,
                sym_df.height,
                before,
                cutoff_ms,
            )

        if (
            max_rows_per_symbol is not None
            and max_rows_per_symbol > 0
            and sym_df.height > max_rows_per_symbol
        ):
            step = max(sym_df.height // max_rows_per_symbol, 1)
            sym_df = (
                sym_df.with_row_index("__opus_row_nr")
                .filter((pl.col("__opus_row_nr") % step) == 0)
                .head(max_rows_per_symbol)
                .drop("__opus_row_nr")
            )
            log.info(
                "dataset: %s capped to %d rows (step=%d)",
                sf.symbol,
                sym_df.height,
                step,
            )

        # Time-based split using row indices on the sorted frame.
        n = sym_df.height
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        part = pl.Series(
            "part",
            (["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)),
            dtype=pl.Utf8,
        )
        frames.append(sym_df.with_columns(part))

    df = pl.concat(frames, how="diagonal_relaxed").sort(["symbol", "ts_ms"])
    log.info(
        "dataset: loaded %d rows across %d symbols (train=%d, val=%d, test=%d)",
        df.height,
        len(inventory),
        df.filter(pl.col("part") == "train").height,
        df.filter(pl.col("part") == "val").height,
        df.filter(pl.col("part") == "test").height,
    )
    return df


__all__ = ["SymbolFile", "SymbolFiles", "discover", "load_dataset"]
