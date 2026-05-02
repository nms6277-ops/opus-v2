"""Compact parquet writer for feature snapshots.

Rotates files every ``parquet_rotation_min`` minutes per symbol:

    {data_dir}/snapshots/{symbol}/{YYYY-MM-DD}/{HH}-{epoch_ms}-{seq}.parquet

Appends are batched in memory. Files are flushed on the rotation boundary
or when the batch reaches ``_BATCH_ROWS`` rows. zstd compression keeps
size low (150-300 MB / day across 10 symbols).

The writer is async-friendly: all disk IO happens in a background task.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from backend.log import get_logger

log = get_logger(__name__)

_BATCH_ROWS = 240  # ~60s of 250ms snapshots


class SnapshotWriter:
    """Append-only parquet writer, hourly rotation.

    Also used (with different ``subdir``) for raw depth/trade event logs.
    """

    def __init__(
        self,
        data_dir: Path,
        symbol: str,
        rotation_min: int = 60,
        subdir: str = "snapshots",
        batch_rows: int = _BATCH_ROWS,
    ) -> None:
        self.symbol = symbol
        self.rotation_min = rotation_min
        self.subdir = subdir
        self.base = data_dir / subdir / symbol
        self.base.mkdir(parents=True, exist_ok=True)

        self._batch_rows = batch_rows
        self._buffer: list[dict[str, Any]] = []
        self._current_path: Path | None = None
        self._flush_seq = 0
        self._lock = asyncio.Lock()
        self._stopped = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    async def append(self, row: dict[str, Any]) -> None:
        """Append a single snapshot row. Non-blocking except on flush."""
        async with self._lock:
            self._buffer.append(row)
            if len(self._buffer) >= self._batch_rows:
                await self._flush_locked()

    async def periodic_flush(self, interval_s: float = 10.0) -> None:
        """Background task: flush any buffered rows every `interval_s` seconds."""
        while not self._stopped:
            try:
                await asyncio.sleep(interval_s)
                async with self._lock:
                    if self._buffer:
                        await self._flush_locked()
            except asyncio.CancelledError:  # pragma: no cover
                break
            except Exception as e:
                log.error("writer %s flush error: %s", self.symbol, e)

    async def close(self) -> None:
        self._stopped = True
        async with self._lock:
            if self._buffer:
                await self._flush_locked()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _path_prefix_for_now(self) -> Path:
        now = datetime.now(tz=UTC)
        if self.rotation_min == 60:
            d = now.strftime("%Y-%m-%d")
            h = now.strftime("%H")
            dir_ = self.base / d
            dir_.mkdir(parents=True, exist_ok=True)
            return dir_ / h
        # Sub-hour rotation
        bucket = (now.minute // self.rotation_min) * self.rotation_min
        d = now.strftime("%Y-%m-%d")
        h = now.strftime("%H")
        dir_ = self.base / d
        dir_.mkdir(parents=True, exist_ok=True)
        return dir_ / f"{h}-{bucket:02d}"

    async def _flush_locked(self) -> None:
        if not self._buffer:
            return
        rows = self._buffer
        self._buffer = []
        prefix = self._path_prefix_for_now()
        self._flush_seq += 1
        path = prefix.with_name(f"{prefix.name}-{int(time.time() * 1000)}-{self._flush_seq:06d}.parquet")
        t0 = time.perf_counter()
        try:
            await asyncio.to_thread(self._write_rows, path, rows)
        except Exception as e:
            log.error("writer %s failed to flush: %s", self.symbol, e)
            # Put rows back for a retry on next flush
            self._buffer = rows + self._buffer
            return
        dt = (time.perf_counter() - t0) * 1000.0
        log.debug("writer %s wrote %d rows to %s in %.1fms", self.symbol, len(rows), path, dt)

    @staticmethod
    def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
        """Write one immutable parquet chunk atomically."""
        df = pl.DataFrame(rows)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            df.write_parquet(tmp, compression="zstd", compression_level=3)
            tmp.replace(path)
        finally:
            # Defensive cleanup if replace failed midway.
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
