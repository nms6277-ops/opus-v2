"""Minimal REST client for Binance USDT-M Futures.

Used for:
  - Fetching the order book snapshot on (re)sync.
  - Placing / cancelling orders in LIVE mode (stub here for now).
  - Reading exchange info (tick size, step size) on symbol start.

Only public endpoints are implemented now; signed endpoints are wired up
later when we activate LIVE mode.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from decimal import Decimal
from typing import Any
from urllib.parse import urlencode

import httpx

from backend.config import settings
from backend.exchanges.binance_filters import SymbolFilters, parse_symbol_filters
from backend.log import get_logger


def _fmt_decimal(value: float, precision: int = 8) -> str:
    """Format a float for Binance order fields WITHOUT scientific notation.

    Python's default ``f"{x}"`` falls back to scientific notation for small
    values (e.g. ``f"{9.95e-06}"`` → ``"9.95e-06"``). Binance Futures rejects
    scientific notation on ``quantity`` / ``price`` / ``stopPrice``. For
    micro-cap tokens (SHIB ~$0.00001, PEPE ~$0.000008, FLOKI ~$0.00001)
    every protective STOP_MARKET / TAKE_PROFIT_MARKET would silently fail
    without this, leaving the live position with no exchange-side safety net.

    8 decimal places is the safe upper bound for Binance Futures; trailing
    zeros are stripped so ``1.2`` stays ``"1.2"`` and ``0.00001194`` becomes
    ``"0.00001194"`` instead of ``"1.194e-05"``.
    """
    d = Decimal(str(value))
    formatted = format(d, f".{precision}f")
    # Strip trailing zeros and a dangling decimal point, but keep at least
    # the integer portion so "0" stays "0" (not "").
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted or "0"


log = get_logger(__name__)


class BinanceRest:
    def __init__(self, api_key: str = "", api_secret: str = "", base: str | None = None) -> None:
        self._api_key = api_key
        self._api_secret = api_secret.encode() if api_secret else b""
        self._base = (base or settings.binance_rest).rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=settings.binance_connect_timeout_s,
                read=15.0,
                write=15.0,
                pool=10.0,
            ),
            headers={"User-Agent": "opus/0.1", **({"X-MBX-APIKEY": api_key} if api_key else {})},
        )
        self._filters_cache: dict[str, SymbolFilters] = {}

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    async def depth_snapshot(self, symbol: str, limit: int = 1000) -> dict[str, Any]:
        """GET /fapi/v1/depth — returns full snapshot with lastUpdateId."""
        url = f"{self._base}/fapi/v1/depth"
        r = await self._client.get(url, params={"symbol": symbol.upper(), "limit": limit})
        r.raise_for_status()
        return r.json()

    async def exchange_info(self) -> dict[str, Any]:
        """GET /fapi/v1/exchangeInfo — symbols, filters, tick sizes."""
        url = f"{self._base}/fapi/v1/exchangeInfo"
        r = await self._client.get(url)
        r.raise_for_status()
        return r.json()

    async def symbol_filters(self, symbol: str) -> dict[str, Any] | None:
        """Return the filters dict for one symbol, or None if unknown."""
        info = await self.exchange_info()
        target = symbol.upper()
        for s in info.get("symbols", []):
            if s.get("symbol") == target:
                return s
        return None

    async def ping(self) -> bool:
        try:
            r = await self._client.get(f"{self._base}/fapi/v1/ping")
            return r.status_code == 200
        except Exception:
            return False

    async def server_time(self) -> int:
        r = await self._client.get(f"{self._base}/fapi/v1/time")
        r.raise_for_status()
        return int(r.json()["serverTime"])

    async def cached_symbol_filters(self, symbol: str) -> SymbolFilters:
        target = symbol.upper()
        if target in self._filters_cache:
            return self._filters_cache[target]
        info = await self.symbol_filters(target)
        if info is None:
            raise ValueError(f"unknown Binance Futures symbol {target}")
        filters = parse_symbol_filters(info)
        self._filters_cache[target] = filters
        return filters

    # ------------------------------------------------------------------
    # Signed (LIVE only)
    # ------------------------------------------------------------------
    def _sign(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._api_secret:
            raise RuntimeError("API secret not configured — cannot sign request")
        params = {**params, "timestamp": int(time.time() * 1000), "recvWindow": 5000}
        qs = urlencode(params, doseq=True)
        sig = hmac.new(self._api_secret, qs.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    async def _signed_post(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base}{path}"
        signed = self._sign(params)
        r = await self._client.post(url, params=signed)
        self._raise_for_status(r, "POST", path)
        return r.json()

    async def _signed_put(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base}{path}"
        signed = self._sign(params)
        r = await self._client.put(url, params=signed)
        self._raise_for_status(r, "PUT", path)
        return r.json()

    async def _signed_get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | list[Any]:
        url = f"{self._base}{path}"
        signed = self._sign(params or {})
        r = await self._client.get(url, params=signed)
        self._raise_for_status(r, "GET", path)
        return r.json()

    async def _signed_delete(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base}{path}"
        signed = self._sign(params)
        r = await self._client.delete(url, params=signed)
        self._raise_for_status(r, "DELETE", path)
        return r.json()

    @staticmethod
    def _raise_for_status(response: httpx.Response, method: str, path: str) -> None:
        """Surface the Binance error body in the exception message.

        ``httpx.Response.raise_for_status`` only includes the status code,
        which is useless for diagnosing rejected orders (the actionable
        part is in the JSON body, e.g. ``-2021 Order would immediately
        trigger.``). We re-raise with the body inlined so live-trader
        rejections show up readable in logs.
        """

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = response.text[:1000]
            raise RuntimeError(f"Binance {method} {path} failed {response.status_code}: {body}") from e

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        working_type: str | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """POST /fapi/v1/order. Raises if not configured."""
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": _fmt_decimal(quantity),
        }
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("LIMIT order requires price")
            params["price"] = _fmt_decimal(price)
            params["timeInForce"] = time_in_force
        if order_type.upper() in {
            "STOP",
            "STOP_MARKET",
            "TAKE_PROFIT",
            "TAKE_PROFIT_MARKET",
        }:
            if stop_price is None:
                raise ValueError(f"{order_type.upper()} order requires stop_price")
            params["stopPrice"] = _fmt_decimal(stop_price)
        if working_type:
            params["workingType"] = working_type
        if reduce_only:
            params["reduceOnly"] = "true"
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        return await self._signed_post("/fapi/v1/order", params)

    async def cancel_all(self, symbol: str) -> dict[str, Any]:
        return await self._signed_delete("/fapi/v1/allOpenOrders", {"symbol": symbol.upper()})

    async def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        return await self._signed_post(
            "/fapi/v1/leverage",
            {"symbol": symbol.upper(), "leverage": int(leverage)},
        )

    async def account(self) -> dict[str, Any]:
        data = await self._signed_get("/fapi/v2/account")
        if not isinstance(data, dict):
            raise RuntimeError("unexpected Binance account response")
        return data

    async def position_risk(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._signed_get("/fapi/v2/positionRisk", params)
        if isinstance(data, dict):
            return [data]
        return list(data)

    async def get_open_position_amt(self, symbol: str) -> float:
        target = symbol.upper()
        for pos in await self.position_risk(target):
            if str(pos.get("symbol", "")).upper() == target:
                return float(pos.get("positionAmt", 0.0))
        return 0.0

    async def market_close_position(self, symbol: str, position_amt: float) -> dict[str, Any] | None:
        if position_amt == 0:
            return None
        side = "SELL" if position_amt > 0 else "BUY"
        return await self.place_order(
            symbol=symbol,
            side=side,
            order_type="MARKET",
            quantity=abs(position_amt),
            reduce_only=True,
            client_order_id=f"OPUS_FLATTEN_{symbol.upper()}_{int(time.time() * 1000)}",
        )

    async def start_user_data_stream(self) -> str:
        r = await self._client.post(f"{self._base}/fapi/v1/listenKey")
        r.raise_for_status()
        return str(r.json()["listenKey"])

    async def keepalive_user_data_stream(self, listen_key: str) -> None:
        r = await self._client.put(f"{self._base}/fapi/v1/listenKey", params={"listenKey": listen_key})
        r.raise_for_status()

    async def close_user_data_stream(self, listen_key: str) -> None:
        r = await self._client.delete(f"{self._base}/fapi/v1/listenKey", params={"listenKey": listen_key})
        r.raise_for_status()


# One shared client per process
_client: BinanceRest | None = None


def get_client() -> BinanceRest:
    global _client
    if _client is None:
        _client = BinanceRest(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
        )
    return _client


async def shutdown() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


async def _selftest() -> None:  # pragma: no cover
    c = get_client()
    ok = await c.ping()
    log.info("binance ping: %s", ok)
    snap = await c.depth_snapshot("BTCUSDT", 5)
    log.info("depth top: %s / %s", snap["bids"][0], snap["asks"][0])


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_selftest())
