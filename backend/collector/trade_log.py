"""Paper-trade log: one parquet file per UTC day, streaming append.

Every closed paper trade is recorded so we can analyse hit-rate, net
PnL, and drawdown after the fact — and, more importantly, use the
recorded (features → our decision → realised outcome) tuples as extra
training data for future model iterations.

Schema is intentionally flat and self-describing. All price columns are
in the quoted-currency of the contract. All PnL columns are signed.

Location:  ``{data_dir}/trades/{YYYY-MM-DD}.parquet``
Format:    parquet + zstd level 4 (same as snapshot writer)
Rotation:  per UTC day
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

log = logging.getLogger("backend.collector.trade_log")


@dataclass
class TradeRecord:
    """One closed paper trade. Written verbatim to the daily parquet file."""

    ts_open_ms: int
    ts_close_ms: int
    symbol: str
    side: str  # "long" or "short"
    qty: float
    notional_usd: float
    entry_price: float
    exit_price: float
    pnl_bp: float  # gross (before fees), signed
    fee_bp: float  # round-trip fee (positive number)
    net_bp: float  # pnl_bp - fee_bp
    pnl_usd: float  # net PnL in USD
    horizon: str
    predicted_confidence: float
    p_up: float
    p_flat: float
    p_down: float
    exit_reason: str  # "timeout" | "opposing_signal" | "stop_loss" | "manual"
    is_maker_entry: bool
    is_maker_exit: bool


class TradeLogWriter:
    """Async, daily-rotated parquet writer for paper-trade records."""

    def __init__(self, data_dir: Path, batch_rows: int = 32) -> None:
        self.data_dir = Path(data_dir) / "trades"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.batch_rows = batch_rows
        self._buffer: list[dict] = []
        self._day_bucket = 0
        self._lock = asyncio.Lock()

    @staticmethod
    def _today_path(root: Path) -> Path:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        return root / f"{day}.parquet"

    async def append(self, rec: TradeRecord) -> None:
        """Append a trade record. Flushes when ``batch_rows`` is reached."""
        d = asdict(rec)
        async with self._lock:
            self._buffer.append(d)
            if len(self._buffer) >= self.batch_rows:
                await self._flush_locked()

    async def flush(self) -> None:
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        if not self._buffer:
            return
        rows = self._buffer
        self._buffer = []
        target = self._today_path(self.data_dir)
        try:
            await asyncio.to_thread(self._write_rows_sync, target, rows)
            log.info("trade_log: flushed %d rows -> %s", len(rows), target.name)
        except Exception as e:
            # Put rows back so a future flush can retry
            self._buffer = rows + self._buffer
            log.error("trade_log: write failed: %s", e)

    @staticmethod
    def _write_rows_sync(target: Path, rows: list[dict]) -> None:
        """Append-then-rewrite: read existing file, concat, rewrite atomically."""
        new_df = pl.DataFrame(rows)
        if target.exists():
            try:
                existing = pl.read_parquet(target)
                out = pl.concat([existing, new_df], how="diagonal_relaxed")
            except Exception:
                # corrupt/partial file — start fresh with the batch
                out = new_df
        else:
            out = new_df
        tmp = target.with_suffix(target.suffix + ".tmp")
        out.write_parquet(tmp, compression="zstd", compression_level=4)
        tmp.replace(target)

    async def periodic_flush(self, interval_s: float = 30.0) -> None:
        """Background task: call periodically to persist buffered records."""
        while True:
            try:
                await asyncio.sleep(interval_s)
                await self.flush()
            except asyncio.CancelledError:
                await self.flush()
                return
            except Exception as e:
                log.error("trade_log: periodic flush error: %s", e)

    async def close(self) -> None:
        await self.flush()


__all__ = ["TradeLogWriter", "TradeRecord"]
