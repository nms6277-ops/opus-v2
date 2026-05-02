"""Binance USDT-M Futures private user-data WebSocket client."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from typing import Any

import orjson
import websockets
from websockets.exceptions import ConnectionClosed

from backend.config import settings
from backend.exchanges.binance_rest import BinanceRest, get_client
from backend.log import get_logger

log = get_logger(__name__)

OnPrivateUpdate = Callable[[dict[str, Any]], None]
OnPrivateState = Callable[[bool, float], None]


class BinancePrivateWS:
    """Manage listenKey lifecycle and private account/order stream."""

    def __init__(
        self,
        *,
        rest_client: BinanceRest | None = None,
        ws_connect: Callable[..., Any] | None = None,
        on_order_update: OnPrivateUpdate | None = None,
        on_account_update: OnPrivateUpdate | None = None,
        on_state: OnPrivateState | None = None,
        keepalive_interval_s: float = 30 * 60.0,
        base_url: str | None = None,
    ) -> None:
        self.rest = rest_client or get_client()
        self._ws_connect = ws_connect or websockets.connect
        self._on_order_update = on_order_update or (lambda data: None)
        self._on_account_update = on_account_update or (lambda data: None)
        self._on_state = on_state or (lambda connected, ts: None)
        self.keepalive_interval_s = keepalive_interval_s
        self.base_url = (base_url or settings.binance_ws_private).rstrip("/")

        self.listen_key: str = ""
        self.last_msg_ts: float = 0.0
        self.connected: bool = False
        self._stopped = False
        self._run_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._run_task is not None and not self._run_task.done():
            return
        self._stopped = False
        self.listen_key = await self.rest.start_user_data_stream()
        self._keepalive_task = asyncio.create_task(self._keepalive_loop(), name="binance-private-keepalive")
        self._run_task = asyncio.create_task(self._run_loop(), name="binance-private-ws")

    async def stop(self) -> None:
        self._stopped = True
        for task in (self._run_task, self._keepalive_task):
            if task is not None:
                task.cancel()
        for task in (self._run_task, self._keepalive_task):
            if task is not None:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._set_state(False)
        if self.listen_key:
            try:
                await self.rest.close_user_data_stream(self.listen_key)
            except Exception as e:
                log.debug("binance-private: listenKey close failed: %s", e)

    def _set_state(self, connected: bool) -> None:
        self.connected = connected
        ts = time.time()
        if connected:
            self.last_msg_ts = ts
        self._on_state(connected, ts)

    def _url(self) -> str:
        return f"{self.base_url}/{self.listen_key}"

    async def _connect(self):
        conn = self._ws_connect(
            self._url(),
            ping_interval=20,
            ping_timeout=15,
            max_queue=1024,
            close_timeout=1,
        )
        if inspect.isawaitable(conn):
            conn = await conn
        return conn

    async def _run_loop(self) -> None:
        backoff = 1.0
        while not self._stopped:
            try:
                async with await self._connect() as ws:
                    log.info("binance-private: connected")
                    self._set_state(True)
                    backoff = 1.0
                    await self._read_loop(ws)
            except asyncio.CancelledError:
                break
            except ConnectionClosed as e:
                log.warning("binance-private: closed: %s", e)
            except Exception as e:
                log.error("binance-private: error: %r", e)
            finally:
                self._set_state(False)
            if self._stopped:
                break
            await asyncio.sleep(min(backoff, 30.0))
            backoff = min(backoff * 2.0, 30.0)

    async def _read_loop(self, ws) -> None:
        while not self._stopped:
            msg = await asyncio.wait_for(ws.recv(), timeout=60.0)
            self.last_msg_ts = time.time()
            self._on_state(True, self.last_msg_ts)
            self._handle_message(msg)

    def _handle_message(self, msg: str) -> None:
        try:
            data = orjson.loads(msg)
        except Exception:
            log.debug("binance-private: bad json")
            return
        event_type = data.get("e", "")
        if event_type == "ORDER_TRADE_UPDATE":
            self._on_order_update(data)
        elif event_type == "ACCOUNT_UPDATE":
            self._on_account_update(data)

    async def _keepalive_loop(self) -> None:
        while not self._stopped:
            try:
                await asyncio.sleep(self.keepalive_interval_s)
                if self.listen_key:
                    await self.rest.keepalive_user_data_stream(self.listen_key)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("binance-private: listenKey keepalive failed: %s", e)
