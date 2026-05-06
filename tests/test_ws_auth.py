import base64

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.api import ws as ws_api
from backend.config import settings


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(ws_api.router)
    return TestClient(app)


def _basic(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {token}"


def test_ws_accepts_when_auth_disabled(monkeypatch):
    monkeypatch.setattr(settings, "ui_user", "")
    monkeypatch.setattr(settings, "ui_password", "")

    with _client().websocket_connect("/ws") as websocket:
        first = websocket.receive_json()
        assert first["mode"] in {"collect", "paper", "live"}


def test_ws_rejects_missing_auth_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "ui_user", "operator")
    monkeypatch.setattr(settings, "ui_password", "secret")

    with pytest.raises(WebSocketDisconnect):
        with _client().websocket_connect("/ws"):
            pass


def test_ws_accepts_valid_basic_auth(monkeypatch):
    monkeypatch.setattr(settings, "ui_user", "operator")
    monkeypatch.setattr(settings, "ui_password", "secret")

    headers = {"authorization": _basic("operator", "secret")}
    with _client().websocket_connect("/ws", headers=headers) as websocket:
        first = websocket.receive_json()
        assert first["mode"] in {"collect", "paper", "live"}


def test_ws_rejects_empty_password_when_only_user_configured(monkeypatch):
    """Regression: half-configured auth (user set, password empty) used to
    accept any empty-password request because ``compare_digest("", "")``
    is True. We now treat half-configured as misconfigured and refuse all
    credentials until both are set."""

    monkeypatch.setattr(settings, "ui_user", "operator")
    monkeypatch.setattr(settings, "ui_password", "")

    headers = {"authorization": _basic("operator", "")}
    with pytest.raises(WebSocketDisconnect):
        with _client().websocket_connect("/ws", headers=headers):
            pass


def test_ws_rejects_when_only_password_configured(monkeypatch):
    """Symmetric guard: only password set, user empty must also reject."""

    monkeypatch.setattr(settings, "ui_user", "")
    monkeypatch.setattr(settings, "ui_password", "secret")

    headers = {"authorization": _basic("", "secret")}
    with pytest.raises(WebSocketDisconnect):
        with _client().websocket_connect("/ws", headers=headers):
            pass
