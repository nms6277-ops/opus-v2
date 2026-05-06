"""Shared UI authentication helpers for REST and WebSocket endpoints."""

from __future__ import annotations

import base64
import secrets

from backend.config import settings


def ui_auth_enabled() -> bool:
    """Return True if Basic auth should be enforced.

    Auth is enforced whenever EITHER ``OPUS_UI_USER`` or ``OPUS_UI_PASSWORD``
    is set. A half-configured pair is intentionally treated as
    "auth on but mis-configured": :func:`check_basic_credentials` will
    refuse every request, so the operator notices immediately and fixes
    ``.env`` instead of silently running an unprotected UI.

    The only configuration that disables auth is leaving BOTH variables
    empty.
    """
    return bool(settings.ui_user) or bool(settings.ui_password)


def check_basic_credentials(username: str, password: str) -> bool:
    """Constant-time check of supplied basic-auth credentials.

    Returns False if either configured value is empty so a half-configured
    setup (only username, or only password) cannot be bypassed via the
    matching half plus an empty other half — ``secrets.compare_digest("", "")``
    returns True, which would otherwise let an attacker who guessed the
    username log in with no password at all.
    """
    if not settings.ui_user or not settings.ui_password:
        return False
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
