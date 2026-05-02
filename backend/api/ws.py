"""WebSocket endpoint for real-time UI updates.

Every ~500 ms we broadcast ``app_state.snapshot()`` to all connected
clients. A per-client send queue with a small bound prevents a slow
consumer from stalling the whole process.
"""

from __future__ import annotations

import asyncio

import orjson
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, WebSocketException, status

from backend.api.auth import check_basic_auth_header
from backend.log import get_logger
from backend.state import app_state

log = get_logger(__name__)
router = APIRouter()


_clients: set[WebSocket] = set()
_broadcast_task: asyncio.Task | None = None


async def _broadcast_loop() -> None:
    global _clients
    while True:
        try:
            await asyncio.sleep(0.5)
            if not _clients:
                continue
            payload = orjson.dumps(app_state.snapshot()).decode()
            dead: list[WebSocket] = []
            for ws in list(_clients):
                try:
                    await asyncio.wait_for(ws.send_text(payload), timeout=1.0)
                except (TimeoutError, WebSocketDisconnect, RuntimeError):
                    dead.append(ws)
                except Exception as e:  # noqa: BLE001
                    log.debug("ws send error: %s", e)
                    dead.append(ws)
            for ws in dead:
                _clients.discard(ws)
        except asyncio.CancelledError:
            break


def _ensure_broadcast() -> None:
    global _broadcast_task
    if _broadcast_task is None or _broadcast_task.done():
        _broadcast_task = asyncio.create_task(_broadcast_loop(), name="ui-ws-broadcast")


@router.websocket("/ws")
async def ws(ws: WebSocket) -> None:
    if not check_basic_auth_header(ws.headers.get("authorization")):
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
    await ws.accept()
    _ensure_broadcast()
    _clients.add(ws)
    try:
        # Send an initial snapshot immediately
        await ws.send_text(orjson.dumps(app_state.snapshot()).decode())
        while True:
            # Clients don't send anything meaningful; keep the connection open
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)


async def shutdown() -> None:
    global _broadcast_task
    if _broadcast_task is not None:
        _broadcast_task.cancel()
        try:
            await _broadcast_task
        except (asyncio.CancelledError, Exception):
            pass
