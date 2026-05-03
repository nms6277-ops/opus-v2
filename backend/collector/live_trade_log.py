"""Append-only live Binance trade-event log.

Writes Binance USER-DATA stream events (`ORDER_TRADE_UPDATE`) to daily
JSONL files so live fills survive process crashes without expensive
parquet read-concat-rewrite cycles. Strictly side-channel: never on the
order-submission hot path.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import orjson


class LiveTradeLogWriter:
    """Write Binance private-WS trade events to daily JSONL files."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir) / "live_trades"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _today_path(root: Path) -> Path:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        return root / f"{day}.jsonl"

    def append_order_update(self, data: dict[str, Any]) -> None:
        order = data.get("o") or {}
        row = {
            "ts_ms": int(data.get("E") or order.get("T") or time.time() * 1000),
            "event_type": data.get("e", "ORDER_TRADE_UPDATE"),
            "symbol": str(order.get("s", "")).upper(),
            "side": str(order.get("S", "")).upper(),
            "order_type": str(order.get("o", "")).upper(),
            "execution_type": str(order.get("x", "")).upper(),
            "order_status": str(order.get("X", "")).upper(),
            "client_order_id": str(order.get("c", "")),
            "order_id": str(order.get("i", "")),
            "trade_id": str(order.get("t", "")),
            "reduce_only": bool(order.get("R")),
            "last_qty": _float(order.get("l")),
            "cum_qty": _float(order.get("z")),
            "last_price": _float(order.get("L")),
            "avg_price": _float(order.get("ap")),
            "stop_price": _float(order.get("sp")),
            "commission": _float(order.get("n")),
            "commission_asset": str(order.get("N", "")),
            "realized_pnl": _float(order.get("rp")),
            "raw": data,
        }
        path = self._today_path(self.data_dir)
        line = orjson.dumps(row).decode("utf-8")
        with self._lock:
            with path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line)
                f.write("\n")

    def close(self) -> None:
        return


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["LiveTradeLogWriter"]
