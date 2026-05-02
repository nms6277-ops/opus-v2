"""Small Telegram notification helper for VPS operation."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

import httpx

from backend.config import settings
from backend.log import get_logger
from backend.state import AppState

log = get_logger(__name__)


class TelegramNotifier:
    """Hourly status push plus `/status` command polling."""

    def __init__(self, state: AppState) -> None:
        self.state = state
        self.token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id
        self._client: httpx.AsyncClient | None = None
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()
        self._offset = 0

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    async def start(self) -> None:
        if not self.enabled:
            return
        self._client = httpx.AsyncClient(timeout=10.0)
        self._tasks = [
            asyncio.create_task(self._hourly_loop(), name="telegram-hourly"),
            asyncio.create_task(self._poll_loop(), name="telegram-poll"),
        ]
        await self.send_status("opus started")

    async def stop(self) -> None:
        self._stopped.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._client is not None:
            await self._client.aclose()

    async def send_status(self, prefix: str = "status") -> None:
        await self._send_text(self._status_text(prefix))

    async def send_risk_event(self, event) -> None:
        if not self.enabled or self._client is None:
            return
        symbol = f" {event.symbol}" if getattr(event, "symbol", "") else ""
        text = (
            f"OPUS risk guard{symbol}: {event.reason}\n"
            f"scope={event.scope} pnl=${event.pnl_usd:.4f} peak=${event.peak_usd:.4f} "
            f"drawdown={event.drawdown_pct * 100:.1f}%\n"
            f"rolling_wr={event.rolling_win_rate * 100:.1f}% "
            f"sum_net_bp={event.rolling_sum_net_bp:.2f} "
            f"loss_streak={event.consecutive_losses}\n"
            "State: awaiting operator decision."
        )
        await self._send_text(text)

    async def _hourly_loop(self) -> None:
        while not self._stopped.is_set():
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=3600)
            except TimeoutError:
                await self.send_status("hourly status")

    async def _poll_loop(self) -> None:
        while not self._stopped.is_set():
            try:
                updates = await self._api("getUpdates", {"timeout": 25, "offset": self._offset})
                for item in updates.get("result", []):
                    self._offset = max(self._offset, int(item.get("update_id", 0)) + 1)
                    msg = item.get("message") or {}
                    text = str(msg.get("text") or "").strip().lower()
                    chat = msg.get("chat") or {}
                    if str(chat.get("id", "")) != str(self.chat_id):
                        continue
                    if text.startswith("/status"):
                        await self.send_status("manual status")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("telegram poll failed: %s", e)
                await asyncio.sleep(10)

    async def _send_text(self, text: str) -> None:
        try:
            await self._api("sendMessage", {"chat_id": self.chat_id, "text": text})
        except Exception as e:
            log.warning("telegram send failed: %s", e)

    async def _api(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("telegram client not started")
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok", False):
            raise RuntimeError(f"telegram {method} failed: {data}")
        return data

    def _status_text(self, prefix: str) -> str:
        snap = self.state.snapshot()
        guards = snap["guards"]
        live_rows = [s for s in snap["symbols"] if s.get("execution_mode") == "live"]
        lines = [
            f"{prefix}",
            f"mode={snap['mode']} public_ws={'ok' if snap['binance_connected'] else 'down'} private_ws={'ok' if snap['binance_private_connected'] else 'down'}",
            f"daily_pnl=${guards['daily_pnl']:.2f} 12h_pnl=${guards['pnl_12h']:.2f} orders={guards['daily_orders']}",
            f"live_symbols={len(live_rows)}/{guards['max_live_symbols']}",
        ]
        if guards["emergency_stopped"]:
            lines.append(f"EMERGENCY: {guards['emergency_reason']}")
        now = time.time()
        for s in live_rows[:10]:
            cooldown = ""
            if s.get("cooldown_until", 0) > now:
                cooldown = f" cooldown={int((s['cooldown_until'] - now) / 60)}m"
            lines.append(
                f"{s['symbol']} {s.get('live_state')} pnl12=${s.get('symbol_realized_pnl_12h', 0):.2f} "
                f"trades={s.get('live_trade_count', 0)} w/l={s.get('live_wins', 0)}/{s.get('live_losses', 0)}"
                f"{cooldown}"
            )
            if s.get("block_reason"):
                lines.append(f"  block={s['block_reason']}")
        return "\n".join(lines)
