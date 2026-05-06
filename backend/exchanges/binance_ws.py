"""Binance USDT-M Futures WebSocket client.

Subscribes to a dynamic set of symbols for:
  - ``<symbol>@depth@100ms`` — book diff updates
  - ``<symbol>@aggTrade``    — aggressor trades
  - ``<symbol>@bookTicker``  — best bid/ask (cheap heartbeat, optional)

Reconnects with exponential backoff. On reconnect and on sequence gaps,
triggers a REST depth snapshot re-sync per affected symbol.

The client is *symbol-aware*: pass a list of symbols at start, and call
``add_symbols`` / ``remove_symbols`` to mutate the subscription at runtime.
Binance combined streams are resubscribed with new streams lists; we do
this by reconnecting — cheap and keeps the code simple.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import httpx
import orjson
import websockets
from websockets.exceptions import ConnectionClosed

from backend.config import settings
from backend.exchanges.binance_rest import get_client
from backend.log import get_logger

log = get_logger(__name__)


OnBookUpdate = Callable[[str, dict], None]
OnTrade = Callable[[str, dict], None]
OnResync = Callable[[str, dict], None]  # called with the REST snapshot


# Global REST rate-limit semaphore and ban window.
# Binance futures depth weight is 20/symbol for limit=1000. /fapi raw weight
# is 2400/min, so theoretically we could do 120 snapshots/min. In practice,
# with 5 symbols, we want << 10 snapshots/min steady state.
_REST_LOCK = asyncio.Lock()
_REST_MIN_INTERVAL_S = 0.5  # at most ~120/min globally
_LAST_REST_TS = 0.0
_REST_BACKOFF_UNTIL = 0.0  # epoch seconds; 0 = no backoff


async def _global_rest_gate() -> bool:
    """Wait for global REST slot. Returns False if we are currently banned."""
    global _LAST_REST_TS
    if time.time() < _REST_BACKOFF_UNTIL:
        return False
    async with _REST_LOCK:
        delta = time.time() - _LAST_REST_TS
        if delta < _REST_MIN_INTERVAL_S:
            await asyncio.sleep(_REST_MIN_INTERVAL_S - delta)
        _LAST_REST_TS = time.time()
    return True


def _engage_rest_backoff(seconds: float) -> None:
    global _REST_BACKOFF_UNTIL
    _REST_BACKOFF_UNTIL = max(_REST_BACKOFF_UNTIL, time.time() + seconds)


_PUBLIC = "public"  # high-frequency depth streams
_MARKET = "market"  # regular market streams (aggTrade, etc.)


class BinanceWS:
    """Manages two Binance Futures WS connections per the 2025 split:

    - ``/public`` for ``@depth@100ms`` (order book diffs)
    - ``/market`` for ``@aggTrade``   (aggressor trades)

    Each connection runs an independent reconnect/backoff loop and carries
    the same watchlist of symbols. Symbol add/remove triggers a debounced
    resubscribe on BOTH channels.
    """

    def __init__(
        self,
        on_book: OnBookUpdate,
        on_trade: OnTrade,
        on_resync: OnResync,
        on_state: Callable[[bool, float], None] | None = None,
    ) -> None:
        self._on_book = on_book
        self._on_trade = on_trade
        self._on_resync = on_resync
        self._on_state = on_state or (lambda connected, ts: None)

        self._symbols: set[str] = set()
        self._symbols_lock = asyncio.Lock()
        self._stopped = False

        self._tasks: dict[str, asyncio.Task] = {}
        self._resubscribe_events: dict[str, asyncio.Event] = {
            _PUBLIC: asyncio.Event(),
            _MARKET: asyncio.Event(),
        }
        self._connected: dict[str, bool] = {_PUBLIC: False, _MARKET: False}

        # Debounce rapid add/remove calls: coalesce multiple UI adds
        # into one WS reconnect instead of reconnecting per-symbol.
        self._resubscribe_debounce_s = 0.75
        self._resubscribe_last_change_ts = 0.0
        # Per-symbol in-flight guard: don't start a second REST resync
        # for a symbol while the first is still outstanding. Combined
        # with the global REST gate (500ms), this is enough to prevent
        # spam without blocking legitimate retries.
        self._resync_in_flight: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self, symbols: list[str]) -> None:
        self._symbols = {s.upper() for s in symbols}
        self._stopped = False
        self._tasks[_PUBLIC] = asyncio.create_task(self._run_channel(_PUBLIC), name="binance-ws-public")
        self._tasks[_MARKET] = asyncio.create_task(self._run_channel(_MARKET), name="binance-ws-market")

    async def stop(self) -> None:
        self._stopped = True
        for ev in self._resubscribe_events.values():
            ev.set()
        for t in list(self._tasks.values()):
            t.cancel()
        for t in list(self._tasks.values()):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()

    async def add_symbols(self, symbols: list[str]) -> None:
        async with self._symbols_lock:
            new = {s.upper() for s in symbols} - self._symbols
            if not new:
                return
            self._symbols |= new
            self._resubscribe_last_change_ts = time.time()
        log.info("binance-ws: +symbols %s (total %d)", new, len(self._symbols))
        asyncio.create_task(self._schedule_resubscribe())

    async def remove_symbols(self, symbols: list[str]) -> None:
        async with self._symbols_lock:
            gone = {s.upper() for s in symbols} & self._symbols
            if not gone:
                return
            self._symbols -= gone
            self._resubscribe_last_change_ts = time.time()
        log.info("binance-ws: -symbols %s (total %d)", gone, len(self._symbols))
        asyncio.create_task(self._schedule_resubscribe())

    async def _schedule_resubscribe(self) -> None:
        """Debounce rapid add/remove calls — wait for the stream of
        changes to settle before actually triggering a WS reconnect."""
        await asyncio.sleep(self._resubscribe_debounce_s)
        if time.time() - self._resubscribe_last_change_ts >= self._resubscribe_debounce_s * 0.9:
            for ev in self._resubscribe_events.values():
                ev.set()

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------
    def _base_url(self, channel: str) -> str:
        if channel == _PUBLIC:
            return settings.binance_ws_public
        if channel == _MARKET:
            return settings.binance_ws_market
        raise ValueError(f"unknown channel {channel}")

    def _streams_url(self, channel: str) -> str:
        parts: list[str] = []
        if channel == _PUBLIC:
            for s in sorted(self._symbols):
                parts.append(f"{s.lower()}@depth@100ms")
            if not parts:
                parts.append("btcusdt@bookTicker")  # harmless keepalive
        elif channel == _MARKET:
            for s in sorted(self._symbols):
                parts.append(f"{s.lower()}@aggTrade")
            if not parts:
                parts.append("btcusdt@aggTrade")  # harmless keepalive
        return f"{self._base_url(channel)}?streams={'/'.join(parts)}"

    def _connect_kwargs(self) -> dict:
        return {
            "ping_interval": 20,
            "ping_timeout": 15,
            "open_timeout": settings.binance_ws_open_timeout_s,
            "max_queue": 4096,
            "max_size": 2**20,
            "close_timeout": 1,
        }

    def _report_state(self) -> None:
        overall = self._connected[_PUBLIC] and self._connected[_MARKET]
        self._on_state(overall, time.time())

    async def _run_channel(self, channel: str) -> None:
        backoff = 1.0
        while not self._stopped:
            url = self._streams_url(channel)
            try:
                log.info(
                    "binance-ws[%s] connecting (%d symbols)",
                    channel,
                    len(self._symbols),
                )
                async with websockets.connect(url, **self._connect_kwargs()) as ws:
                    self._connected[channel] = True
                    self._report_state()
                    backoff = 1.0
                    # On (re)connect of the public (depth) channel, resync
                    # the order book via REST for every current symbol.
                    # The market (trade) channel needs no REST bootstrap.
                    if channel == _PUBLIC:
                        asyncio.create_task(self._resync_all())
                    await self._read_loop(ws, channel)
            except asyncio.CancelledError:
                break
            except ConnectionClosed as e:
                log.warning("binance-ws[%s] closed: %s", channel, e)
            except Exception as e:  # noqa: BLE001
                log.error("binance-ws[%s] error: %r", channel, e)
            finally:
                self._connected[channel] = False
                self._report_state()
            if self._stopped:
                break
            sleep = min(backoff, 30.0)
            log.info("binance-ws[%s] reconnect in %.1fs", channel, sleep)
            await asyncio.sleep(sleep)
            backoff = min(backoff * 2.0, 30.0)

    async def _read_loop(self, ws, channel: str) -> None:
        ev = self._resubscribe_events[channel]
        while not self._stopped:
            if ev.is_set():
                ev.clear()
                log.info("binance-ws[%s]: resubscribing", channel)
                return

            recv_task = asyncio.create_task(ws.recv())
            event_task = asyncio.create_task(ev.wait())
            done, pending = await asyncio.wait(
                {recv_task, event_task},
                timeout=30.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()

            if not done:
                log.warning("binance-ws[%s]: no msg for 30s, reconnecting", channel)
                return

            if recv_task in done:
                try:
                    msg = recv_task.result()
                except Exception as e:
                    log.warning("binance-ws[%s] recv failed: %s", channel, e)
                    return
                self._handle_message(msg, channel)

            if event_task in done:
                continue

    def _handle_message(self, msg: str, channel: str) -> None:
        self._connected[channel] = True
        self._report_state()
        try:
            envelope = orjson.loads(msg)
        except Exception as e:
            log.warning("binance-ws[%s] bad json: %s", channel, e)
            return
        data = envelope.get("data") or envelope
        stream = envelope.get("stream", "")
        event_type = data.get("e", "")
        if event_type == "depthUpdate":
            sym = data.get("s", "").upper()
            if sym:
                self._on_book(sym, data)
        elif event_type == "aggTrade":
            sym = data.get("s", "").upper()
            if sym:
                self._on_trade(sym, data)
        elif "bookTicker" in stream:
            pass

    async def _resync_all(self) -> None:
        """Fetch REST depth snapshots for every currently-subscribed symbol.

        Runs sequentially through the global gate so we do not spike REST
        beyond ``_REST_MIN_INTERVAL_S`` per request.
        """
        async with self._symbols_lock:
            syms = list(self._symbols)
        for sym in syms:
            await self._resync_one(sym)

    async def _resync_one(self, symbol: str) -> None:
        # In-flight guard: if a resync for this symbol is already running
        # (or scheduled on the REST gate), skip. No per-symbol cooldown
        # here — that was the v0.7 bug where the first (blocked) retry
        # consumed the only chance we had to resync and the book
        # stayed unready forever. The global REST gate (500ms) is
        # sufficient to prevent spam across symbols, and the in-flight
        # guard prevents self-concurrency per-symbol.
        if symbol in self._resync_in_flight:
            return
        self._resync_in_flight.add(symbol)
        try:
            if not await _global_rest_gate():
                # Currently in 418/429 backoff — skip silently
                return
            try:
                client = get_client()
                snap = await client.depth_snapshot(symbol, limit=1000)
                self._on_resync(symbol, snap)
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                if code in (418, 429):
                    # 418 = IP banned; 429 = too many requests.
                    # Binance may include Retry-After header; fall back to 120s.
                    retry_after = 120.0
                    try:
                        ra = e.response.headers.get("Retry-After")
                        if ra:
                            retry_after = max(retry_after, float(ra))
                    except Exception:
                        pass
                    _engage_rest_backoff(retry_after)
                    log.error(
                        "binance-ws rate-limited (HTTP %d) on %s — backing off %.0fs",
                        code,
                        symbol,
                        retry_after,
                    )
                else:
                    log.error("binance-ws resync %s HTTP %d: %s", symbol, code, e)
            except Exception as e:  # noqa: BLE001
                log.error("binance-ws resync %s failed: %r", symbol, e)
        finally:
            self._resync_in_flight.discard(symbol)

    async def request_resync(self, symbol: str) -> None:
        asyncio.create_task(self._resync_one(symbol))
