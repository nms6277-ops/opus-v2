"""Shared UI authentication helpers for REST and WebSocket endpoints."""

from __future__ import annotations

import base64
import secrets

from backend.config import settings


def ui_auth_enabled() -> bool:
    return bool(settings.ui_user or settings.ui_password)


def check_basic_credentials(username: str, password: str) -> bool:
    return secrets.compare_digest(username, settings.ui_user) and secrets.compare_digest(
        password, settings.ui_password
    )


def check_basic_auth_header(authorization: str | None) -> bool:
    """Validate an HTTP Basic Authorization header for UI access."""

    if not ui_auth_enabled():
        return True
    if not authorization:
        return False
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "basic" or not token:
        return False
    try:
        decoded = base64.b64decode(token).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    username, sep, password = decoded.partition(":")
    if sep != ":":
        return False
    return check_basic_credentials(username, password)
