"""REST endpoints consumed by the UI.

All endpoints are behind optional HTTP Basic auth
(``OPUS_UI_USER`` / ``OPUS_UI_PASSWORD`` in .env).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from backend.api.auth import check_basic_credentials
from backend.config import Mode, settings
from backend.runtime import get_runtime
from backend.safety import guards as guards_mod
from backend.settings_store import SettingsLimitError
from backend.state import app_state

router = APIRouter(prefix="/api")

_basic = HTTPBasic(auto_error=False)


def _auth(creds: Annotated[HTTPBasicCredentials | None, Depends(_basic)]) -> None:
    if not settings.ui_user and not settings.ui_password:
        return
    if creds is None:
        raise HTTPException(status_code=401, detail="auth required")
    if not check_basic_credentials(creds.username, creds.password):
        raise HTTPException(status_code=401, detail="bad credentials")


# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------
class AddSymbolReq(BaseModel):
    symbol: str = Field(..., min_length=3, max_length=20)
    position_size_usd: float = Field(..., gt=0.0, le=1_000_000.0)


class SetModeReq(BaseModel):
    mode: Mode


class SetGuardReq(BaseModel):
    daily_loss_limit_usd: float | None = Field(None, ge=0.0)
    loss_12h_limit_usd: float | None = Field(None, ge=0.0)
    symbol_loss_limit_usd: float | None = Field(None, ge=0.0)
    max_position_usd: float | None = Field(None, ge=0.0)
    max_live_symbols: int | None = Field(None, ge=1)
    max_orders_per_min: int | None = Field(None, ge=1)


class RuntimeSettingsReq(BaseModel):
    leverage: int | None = Field(None, ge=1)
    max_live_symbols: int | None = Field(None, ge=1)
    daily_loss_limit_usd: float | None = Field(None, ge=0.0)
    loss_12h_limit_usd: float | None = Field(None, ge=0.0)
    symbol_loss_limit_usd: float | None = Field(None, ge=0.0)
    probation_notional_usd: float | None = Field(None, ge=0.0)
    active_notional_usd: float | None = Field(None, ge=0.0)
    min_expected_gross_bp: float | None = Field(None, ge=0.0)
    global_profit_giveback_pct: float | None = Field(None, ge=0.0, le=1.0)
    symbol_profit_giveback_pct: float | None = Field(None, ge=0.0, le=1.0)
    loss_streak_limit: int | None = Field(None, ge=1)
    rolling_guard_trades: int | None = Field(None, ge=1)
    rolling_min_win_rate: float | None = Field(None, ge=0.0, le=1.0)
    rolling_min_loss_net_bp: float | None = Field(None, ge=0.0)
    rolling_min_drawdown_pct: float | None = Field(None, ge=0.0, le=1.0)
    global_guard_min_trades: int | None = Field(None, ge=1)
    symbol_guard_min_trades: int | None = Field(None, ge=1)
    probation_trades: int | None = Field(None, ge=1)
    cooldown_hours: int | None = Field(None, ge=1)


class SetSymbolModeReq(BaseModel):
    symbol: str = Field(..., min_length=3, max_length=20)
    execution_mode: str = Field(..., pattern="^(paper|live)$")


class DisableSymbolReq(BaseModel):
    symbol: str = Field(..., min_length=3, max_length=20)
    reason: str = ""


class EmergencyStopReq(BaseModel):
    reason: str = "operator"


class SelectModelReq(BaseModel):
    model_dir: str = Field(..., min_length=1, max_length=500)


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@router.get("/status")
def status_(_auth: Annotated[None, Depends(_auth)]) -> dict:
    return app_state.snapshot()


@router.post("/mode")
async def set_mode(body: SetModeReq, _auth: Annotated[None, Depends(_auth)]) -> dict:
    await get_runtime().set_mode(body.mode)
    return {"ok": True, "mode": body.mode.value}


@router.get("/watchlist")
def watchlist(_auth: Annotated[None, Depends(_auth)]) -> list[dict]:
    return [s.to_dict() for s in app_state.symbols.values()]


@router.post("/watchlist/add")
async def add_symbol(body: AddSymbolReq, _auth: Annotated[None, Depends(_auth)]) -> dict:
    if body.position_size_usd > app_state.guards.max_position_usd:
        raise HTTPException(
            status_code=400,
            detail=(
                f"position_size_usd {body.position_size_usd} exceeds "
                f"max_position_usd guard {app_state.guards.max_position_usd}"
            ),
        )
    await get_runtime().add_symbol(body.symbol, body.position_size_usd)
    return {"ok": True, "symbol": body.symbol.upper()}


@router.post("/watchlist/remove")
async def remove_symbol(body: dict, _auth: Annotated[None, Depends(_auth)]) -> dict:
    symbol = str(body.get("symbol", "")).upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")
    await get_runtime().remove_symbol(symbol)
    return {"ok": True, "symbol": symbol}


@router.post("/guards")
def set_guards(body: SetGuardReq, _auth: Annotated[None, Depends(_auth)]) -> dict:
    """Update safety limits, clamped to .env hard caps.

    The hard caps in ``backend/config.py`` (``OPUS_HARD_*`` env vars)
    define the maximum permitted values. UI-editable limits may only
    lower these, never exceed them. Anything above the cap is silently
    clamped down to the cap so a misconfigured UI cannot widen the
    safety envelope.
    """
    g = app_state.guards
    if body.daily_loss_limit_usd is not None:
        g.daily_loss_limit_usd = min(body.daily_loss_limit_usd, settings.hard_daily_loss_usd)
    if body.loss_12h_limit_usd is not None:
        g.loss_12h_limit_usd = min(body.loss_12h_limit_usd, settings.hard_12h_loss_usd)
    if body.symbol_loss_limit_usd is not None:
        g.symbol_loss_limit_usd = min(body.symbol_loss_limit_usd, settings.hard_symbol_loss_usd)
    if body.max_position_usd is not None:
        g.max_position_usd = min(body.max_position_usd, settings.hard_max_notional_usd)
    if body.max_live_symbols is not None:
        g.max_live_symbols = min(body.max_live_symbols, settings.hard_max_live_symbols)
    if body.max_orders_per_min is not None:
        g.max_orders_per_min = min(body.max_orders_per_min, settings.hard_max_orders_per_min)
    return {
        "ok": True,
        "applied": {
            "daily_loss_limit_usd": g.daily_loss_limit_usd,
            "loss_12h_limit_usd": g.loss_12h_limit_usd,
            "symbol_loss_limit_usd": g.symbol_loss_limit_usd,
            "max_position_usd": g.max_position_usd,
            "max_live_symbols": g.max_live_symbols,
            "max_orders_per_min": g.max_orders_per_min,
        },
    }


@router.get("/settings")
def runtime_settings(_auth: Annotated[None, Depends(_auth)]) -> dict:
    rt = get_runtime()
    if hasattr(rt, "runtime_settings_dict"):
        return rt.runtime_settings_dict()
    return {}


@router.get("/models")
def models(_auth: Annotated[None, Depends(_auth)]) -> dict:
    rt = get_runtime()
    status = (
        rt.model_status()
        if hasattr(rt, "model_status")
        else {
            "enabled": False,
            "horizons": [],
            "model_dir": str(settings.model_dir),
        }
    )
    available = rt.available_models() if hasattr(rt, "available_models") else []
    return {
        "active_model": status["model_dir"],
        "status": status,
        "models": available,
    }


@router.post("/models/select")
async def select_model(body: SelectModelReq, _auth: Annotated[None, Depends(_auth)]) -> dict:
    try:
        return await get_runtime().set_model_dir(body.model_dir)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/settings")
async def set_runtime_settings(body: RuntimeSettingsReq, _auth: Annotated[None, Depends(_auth)]) -> dict:
    try:
        return await get_runtime().set_runtime_settings(body.model_dump(exclude_none=True))
    except SettingsLimitError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/watchlist/mode")
async def set_symbol_mode(body: SetSymbolModeReq, _auth: Annotated[None, Depends(_auth)]) -> dict:
    try:
        return await get_runtime().set_symbol_execution_mode(body.symbol.upper(), body.execution_mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/watchlist/disable")
async def disable_symbol(body: DisableSymbolReq, _auth: Annotated[None, Depends(_auth)]) -> dict:
    try:
        return await get_runtime().disable_symbol(body.symbol.upper(), body.reason)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/emergency/stop")
async def emergency_stop(body: EmergencyStopReq, _auth: Annotated[None, Depends(_auth)]) -> dict:
    return await get_runtime().emergency_stop(body.reason)


@router.post("/emergency/clear")
def emergency_clear(_auth: Annotated[None, Depends(_auth)]) -> dict:
    guards_mod.clear_emergency(app_state.guards)
    return {"ok": True}


@router.get("/trader")
def trader_stats(_auth: Annotated[None, Depends(_auth)]) -> dict:
    """Snapshot of paper-trader state: open positions + daily PnL summary."""
    rt = get_runtime()
    trader = getattr(rt, "_trader", None)
    if trader is not None and hasattr(trader, "stats_dict"):
        return trader.stats_dict()
    return {
        "open_positions": [],
        "daily_trades": 0,
        "daily_wins": 0,
        "daily_losses": 0,
        "daily_win_rate": 0.0,
        "daily_pnl_usd": 0.0,
        "horizon": settings.trade_horizon,
        "conf_threshold": settings.trade_conf_threshold,
        "notional_usd": settings.trade_notional_usd,
        "allowed_symbols": list(settings.trade_symbols),
    }


@router.get("/predictor")
def predictor_status(_auth: Annotated[None, Depends(_auth)]) -> dict:
    """Predictor status: which horizons are loaded and which symbols seen."""
    rt = get_runtime()
    if hasattr(rt, "model_status"):
        return rt.model_status()
    pred = getattr(rt, "_predictor", None)
    if pred is None:
        return {"enabled": False, "horizons": [], "model_dir": str(settings.model_dir)}
    return {
        "enabled": pred.enabled,
        "horizons": pred.horizons,
        "model_dir": str(settings.model_dir),
    }
