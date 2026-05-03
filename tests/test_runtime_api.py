from dataclasses import asdict

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import rest
from backend.settings_store import RuntimeSettings, SettingsLimitError


class FakeRuntime:
    def __init__(self):
        self.runtime_settings = RuntimeSettings()
        self.mode_calls = []
        self.disabled = []
        self.emergency = []
        self.selected_model = None

    def runtime_settings_dict(self):
        return asdict(self.runtime_settings)

    async def set_runtime_settings(self, updates):
        if updates.get("active_notional_usd", 0) > 20.0:
            raise SettingsLimitError("active_notional_usd=50.0 exceeds OPUS_HARD_MAX_NOTIONAL_USD=20.0")
        current = asdict(self.runtime_settings)
        current.update(updates)
        self.runtime_settings = RuntimeSettings(**current)
        return self.runtime_settings_dict()

    async def set_symbol_execution_mode(self, symbol, execution_mode):
        self.mode_calls.append((symbol, execution_mode))
        return {"symbol": symbol, "execution_mode": execution_mode}

    async def disable_symbol(self, symbol, reason):
        self.disabled.append((symbol, reason))
        return {"symbol": symbol, "disabled": True, "reason": reason}

    async def emergency_stop(self, reason):
        self.emergency.append(reason)
        return {"ok": True, "reason": reason}

    def model_status(self):
        return {"enabled": True, "horizons": ["1s"], "model_dir": "models/agnostic/global"}

    def available_models(self):
        return [
            {
                "label": "agnostic/global",
                "path": "models/agnostic/global",
                "horizons": ["1s"],
                "active": True,
            }
        ]

    async def set_model_dir(self, model_dir):
        self.selected_model = model_dir
        return {"enabled": True, "horizons": ["1s", "2s"], "model_dir": model_dir}


class FakeExchangeWS:
    def __init__(self):
        self.started = []
        self.stopped = False
        self.added = []
        self.removed = []

    async def start(self, symbols):
        self.started.append(list(symbols))

    async def stop(self):
        self.stopped = True

    async def add_symbols(self, symbols):
        self.added.append(list(symbols))

    async def remove_symbols(self, symbols):
        self.removed.append(list(symbols))


def _client(fake_runtime, monkeypatch):
    monkeypatch.setattr(rest, "get_runtime", lambda: fake_runtime)
    app = FastAPI()
    app.include_router(rest.router)
    return TestClient(app)


def test_get_runtime_settings(monkeypatch):
    fake = FakeRuntime()
    client = _client(fake, monkeypatch)

    response = client.get("/api/settings")

    assert response.status_code == 200
    assert response.json()["active_notional_usd"] == 20.0


def test_post_runtime_settings(monkeypatch):
    fake = FakeRuntime()
    client = _client(fake, monkeypatch)

    response = client.post("/api/settings", json={"active_notional_usd": 19.0})

    assert response.status_code == 200
    assert response.json()["active_notional_usd"] == 19.0


def test_post_runtime_settings_reports_hard_cap_violation(monkeypatch):
    fake = FakeRuntime()
    client = _client(fake, monkeypatch)

    response = client.post("/api/settings", json={"active_notional_usd": 50.0})

    assert response.status_code == 400
    assert "OPUS_HARD_MAX_NOTIONAL_USD" in response.json()["detail"]


def test_post_guards_clamps_to_hard_caps(monkeypatch):
    """`/api/guards` must not let a misconfigured UI widen the safety envelope."""
    from backend.config import settings
    from backend.state import app_state

    fake = FakeRuntime()
    client = _client(fake, monkeypatch)

    # Try to set every limit far above its hard cap.
    response = client.post(
        "/api/guards",
        json={
            "daily_loss_limit_usd": 999.0,
            "loss_12h_limit_usd": 999.0,
            "symbol_loss_limit_usd": 999.0,
            "max_position_usd": 999.0,
            "max_live_symbols": 999,
            "max_orders_per_min": 100_000,
        },
    )

    assert response.status_code == 200
    applied = response.json()["applied"]
    assert applied["daily_loss_limit_usd"] == settings.hard_daily_loss_usd
    assert applied["loss_12h_limit_usd"] == settings.hard_12h_loss_usd
    assert applied["symbol_loss_limit_usd"] == settings.hard_symbol_loss_usd
    assert applied["max_position_usd"] == settings.hard_max_notional_usd
    assert applied["max_live_symbols"] == settings.hard_max_live_symbols
    assert applied["max_orders_per_min"] == settings.hard_max_orders_per_min
    # And the global state is consistent with what was applied.
    assert app_state.guards.daily_loss_limit_usd == settings.hard_daily_loss_usd
    assert app_state.guards.max_position_usd == settings.hard_max_notional_usd


def test_set_symbol_execution_mode(monkeypatch):
    fake = FakeRuntime()
    client = _client(fake, monkeypatch)

    response = client.post("/api/watchlist/mode", json={"symbol": "ubusdt", "execution_mode": "live"})

    assert response.status_code == 200
    assert response.json() == {"symbol": "UBUSDT", "execution_mode": "live"}
    assert fake.mode_calls == [("UBUSDT", "live")]


def test_disable_symbol(monkeypatch):
    fake = FakeRuntime()
    client = _client(fake, monkeypatch)

    response = client.post("/api/watchlist/disable", json={"symbol": "UBUSDT", "reason": "bad sample"})

    assert response.status_code == 200
    assert response.json()["disabled"] is True
    assert fake.disabled == [("UBUSDT", "bad sample")]


def test_emergency_stop(monkeypatch):
    fake = FakeRuntime()
    client = _client(fake, monkeypatch)

    response = client.post("/api/emergency/stop", json={"reason": "operator"})

    assert response.status_code == 200
    assert fake.emergency == ["operator"]


def test_get_models(monkeypatch):
    fake = FakeRuntime()
    client = _client(fake, monkeypatch)

    response = client.get("/api/models")

    assert response.status_code == 200
    assert response.json()["active_model"] == "models/agnostic/global"
    assert response.json()["models"][0]["label"] == "agnostic/global"


def test_select_model(monkeypatch):
    fake = FakeRuntime()
    client = _client(fake, monkeypatch)

    response = client.post("/api/models/select", json={"model_dir": "models/agnostic_next/global"})

    assert response.status_code == 200
    assert response.json()["model_dir"] == "models/agnostic_next/global"
    assert fake.selected_model == "models/agnostic_next/global"
