"""Bybit v5 linear public WebSocket client (READ-ONLY, analysis).

We subscribe to ``orderbook.50.<symbol>`` and ``publicTrade.<symbol>`` for
the same symbols as Binance and stash the most recent best bid/ask +
trade volumes in a per-symbol dict for cross-exchange features.

No order submission from Bybit — trading is Binance-only.

Docs: https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import orjson
import websockets
from websockets.exceptions import ConnectionClosed

from backend.config import settings
from backend.log import get_logger

log = get_logger(__name__)


class BybitWS:
    def __init__(
        self,
        on_book: Callable[[str, dict], None],
        on_trade: Callable[[str, dict], None],
        on_state: Callable[[bool, float], None] | None = None,
    ) -> None:
        self._on_book = on_book
        self._on_trade = on_trade
        self._on_state = on_state or (lambda connected, ts: None)

        self._symbols: set[str] = set()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._stopped = False
        self._ws: websockets.WebSocketClientProtocol | None = None

    async def start(self, symbols: list[str]) -> None:
        self._symbols = {s.upper() for s in symbols}
        self._stopped = False
        self._task = asyncio.create_task(self._run(), name="bybit-ws")

    async def stop(self) -> None:
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def add_symbols(self, symbols: list[str]) -> None:
        async with self._lock:
            new = {s.upper() for s in symbols} - self._symbols
            if not new:
                return
            self._symbols |= new
        if self._ws is not None:
            await self._subscribe(self._ws, list(new))

    async def remove_symbols(self, symbols: list[str]) -> None:
        async with self._lock:
            gone = {s.upper() for s in symbols} & self._symbols
            if not gone:
                return
            self._symbols -= gone
        if self._ws is not None:
            await self._unsubscribe(self._ws, list(gone))

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stopped:
            try:
                log.info("bybit-ws connecting")
                async with websockets.connect(
                    settings.bybit_ws,
                    ping_interval=20,
                    ping_timeout=15,
                    max_queue=4096,
                    close_timeout=1,
                ) as ws:
                    self._ws = ws
                    self._on_state(True, time.time())
                    backoff = 1.0
                    # Subscribe all current symbols
                    async with self._lock:
                        syms = list(self._symbols)
                    if syms:
                        await self._subscribe(ws, syms)
                    await self._read_loop(ws)
            except asyncio.CancelledError:
                break
            except ConnectionClosed as e:
                log.warning("bybit-ws closed: %s", e)
            except Exception as e:  # noqa: BLE001
                log.error("bybit-ws error: %r", e)
            finally:
                self._ws = None
                self._on_state(False, time.time())
            if self._stopped:
                break
            await asyncio.sleep(min(backoff, 30.0))
            backoff = min(backoff * 2.0, 30.0)

    async def _subscribe(self, ws, symbols: list[str]) -> None:
        args = []
        for s in symbols:
            args.append(f"orderbook.50.{s.upper()}")
            args.append(f"publicTrade.{s.upper()}")
        if not args:
            return
        msg = {"op": "subscribe", "args": args}
        await ws.send(orjson.dumps(msg).decode())

    async def _unsubscribe(self, ws, symbols: list[str]) -> None:
        args = []
        for s in symbols:
            args.append(f"orderbook.50.{s.upper()}")
            args.append(f"publicTrade.{s.upper()}")
        if not args:
            return
        msg = {"op": "unsubscribe", "args": args}
        await ws.send(orjson.dumps(msg).decode())

    async def _read_loop(self, ws) -> None:
        while not self._stopped:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
            except TimeoutError:
                log.warning("bybit-ws: no msg for 30s, reconnecting")
                return
            self._on_state(True, time.time())
            try:
                data = orjson.loads(msg)
            except Exception:
                continue
            topic = data.get("topic", "")
            if topic.startswith("orderbook."):
                sym = topic.split(".", 2)[-1]
                self._on_book(sym, data)
            elif topic.startswith("publicTrade."):
                sym = topic.split(".", 1)[-1]
                self._on_trade(sym, data)
