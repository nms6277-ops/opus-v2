from backend.config import Settings
from backend.exchanges.binance_rest import BinanceRest
from backend.exchanges.binance_ws import BinanceWS


def test_binance_timeouts_default_to_slow_tls_safe_values():
    cfg = Settings(_env_file=None)

    assert cfg.binance_connect_timeout_s >= 12.0
    assert cfg.binance_ws_open_timeout_s >= 12.0


def test_binance_rest_uses_configured_connect_timeout(monkeypatch):
    monkeypatch.setattr("backend.exchanges.binance_rest.settings.binance_connect_timeout_s", 17.0)

    client = BinanceRest(base="https://example.invalid")

    assert client._client.timeout.connect == 17.0


def test_binance_ws_uses_configured_open_timeout(monkeypatch):
    monkeypatch.setattr("backend.exchanges.binance_ws.settings.binance_ws_open_timeout_s", 18.0)
    ws = BinanceWS(lambda _s, _d: None, lambda _s, _d: None, lambda _s, _d: None)

    assert ws._connect_kwargs()["open_timeout"] == 18.0
