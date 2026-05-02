import asyncio

import orjson
import pytest

from backend.exchanges.binance_private_ws import BinancePrivateWS


class FakeRest:
    def __init__(self):
        self.started = 0
        self.keepalives = 0
        self.closed = 0

    async def start_user_data_stream(self):
        self.started += 1
        return "LISTEN_KEY"

    async def keepalive_user_data_stream(self, listen_key):
        assert listen_key == "LISTEN_KEY"
        self.keepalives += 1

    async def close_user_data_stream(self, listen_key):
        assert listen_key == "LISTEN_KEY"
        self.closed += 1


class FakeWebSocket:
    def __init__(self, messages):
        self.messages = list(messages)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def recv(self):
        if self.messages:
            return self.messages.pop(0)
        await asyncio.sleep(10)


@pytest.mark.asyncio
async def test_private_ws_creates_and_closes_listen_key():
    rest = FakeRest()

    async def ws_connect(*args, **kwargs):
        return FakeWebSocket([])

    client = BinancePrivateWS(rest_client=rest, ws_connect=ws_connect, keepalive_interval_s=3600)
    await client.start()
    await asyncio.sleep(0)
    await client.stop()

    assert rest.started == 1
    assert rest.closed == 1
    assert client.listen_key == "LISTEN_KEY"


@pytest.mark.asyncio
async def test_private_ws_emits_order_and_account_updates():
    rest = FakeRest()
    order_updates = []
    account_updates = []

    client = BinancePrivateWS(
        rest_client=rest,
        on_order_update=lambda data: order_updates.append(data),
        on_account_update=lambda data: account_updates.append(data),
    )

    client._handle_message(orjson.dumps({"e": "ORDER_TRADE_UPDATE", "o": {"s": "UBUSDT"}}).decode())
    client._handle_message(orjson.dumps({"e": "ACCOUNT_UPDATE", "a": {"m": "ORDER"}}).decode())

    assert order_updates == [{"e": "ORDER_TRADE_UPDATE", "o": {"s": "UBUSDT"}}]
    assert account_updates == [{"e": "ACCOUNT_UPDATE", "a": {"m": "ORDER"}}]


@pytest.mark.asyncio
async def test_private_ws_reports_connected_state_from_read_loop():
    rest = FakeRest()
    states = []
    msg = orjson.dumps({"e": "ACCOUNT_UPDATE", "a": {}}).decode()

    async def ws_connect(*args, **kwargs):
        return FakeWebSocket([msg])

    client = BinancePrivateWS(
        rest_client=rest,
        ws_connect=ws_connect,
        on_state=lambda connected, ts: states.append(connected),
        keepalive_interval_s=3600,
    )

    await client.start()
    await asyncio.sleep(0.05)
    await client.stop()

    assert True in states
    assert states[-1] is False
