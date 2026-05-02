"""Runtime configuration loaded from `.env`."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Mode(str, Enum):
    """Top-level operating mode of the bot."""

    COLLECT = "collect"
    PAPER = "paper"
    LIVE = "live"


class Settings(BaseSettings):
    """All configuration is prefixed with ``OPUS_`` in the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="OPUS_",
        case_sensitive=False,
        extra="ignore",
    )

    # Mode
    mode: Mode = Mode.COLLECT
    runtime_profile: str = "local"

    # Server
    host: str = "127.0.0.1"
    port: int = 8081
    ui_user: str = ""
    ui_password: str = ""

    # Storage
    data_dir: Path = Path("./data")
    logs_dir: Path = Path("./logs")
    parquet_rotation_min: int = 60

    # Collector
    snapshot_interval_ms: int = 250
    lob_depth: int = 200
    snapshot_depth: int = 40

    # Raw logs (tick-level). Both optional; disk budget is the trade-off.
    collect_raw_depth: bool = False    # every depthUpdate, ~1-3 MB/symbol/hr after zstd
    collect_raw_trades: bool = False   # every aggTrade, ~0.5-5 MB/symbol/hr after zstd

    # Price-bucket aggregations (in bp, +/- from mid) written into each snapshot row.
    # Tick-size-independent features that work on any symbol.
    bucket_bps: Annotated[tuple[int, ...], NoDecode] = (5, 10, 25, 50)

    # Exchanges.
    #
    # Binance split their WS infrastructure (see "Important WebSocket Change
    # Notice"): high-frequency depth goes to /public, regular feeds (aggTrade,
    # markPrice, kline, ticker) go to /market, user data to /private. We open
    # two separate connections to reduce per-connection load and jitter, as
    # recommended by Binance.
    binance_ws_public: str = "wss://fstream.binance.com/public/stream"
    binance_ws_market: str = "wss://fstream.binance.com/market/stream"
    binance_ws_private: str = "wss://fstream.binance.com/private/ws"
    # Legacy combined URL — kept for backwards compatibility and tests.
    binance_ws: str = "wss://fstream.binance.com/stream"
    binance_rest: str = "https://fapi.binance.com"
    binance_connect_timeout_s: float = Field(15.0, ge=1.0)
    binance_ws_open_timeout_s: float = Field(15.0, ge=1.0)
    bybit_ws: str = "wss://stream.bybit.com/v5/public/linear"
    bybit_rest: str = "https://api.bybit.com"
    enable_bybit: bool = False

    # Binance API keys (live only)
    binance_api_key: str = ""
    binance_api_secret: str = ""

    # Safety guards (also editable at runtime via UI, these are initial values)
    daily_loss_limit_usd: float = Field(5.0, ge=0.0)
    max_position_usd: float = Field(50.0, ge=0.0)
    max_live_symbols: int = Field(1, ge=1)
    max_orders_per_min: int = Field(60, ge=1)
    ws_stale_ms: int = Field(1500, ge=500)

    # VPS/live hard caps. UI-editable runtime settings may only lower these,
    # never exceed them.
    hard_daily_loss_usd: float = Field(2.0, ge=0.0)
    hard_12h_loss_usd: float = Field(2.0, ge=0.0)
    hard_symbol_loss_usd: float = Field(0.30, ge=0.0)
    hard_max_live_symbols: int = Field(5, ge=1)
    hard_max_notional_usd: float = Field(20.0, ge=0.0)
    hard_max_leverage: int = Field(10, ge=1)

    # Telegram notifications.
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ML / paper trader
    #
    # Path to the trained model bundle (output of `python -m backend.ml.train`).
    # The directory must contain h{horizon}/{model.lgb,meta.json} subfolders.
    model_dir: Path = Path("./models/global")

    # Which trained horizon's predictions drive the paper trader. Must match a
    # trained horizon name (e.g. "1s", "5s", "30s").
    trade_horizon: str = "5s"

    # Subset of symbols the paper trader is *allowed* to trade. Comma-separated
    # in the env var (e.g. "BTCUSDT,ETHUSDT,AIOTUSDT"). Empty/unset means trade
    # any symbol the runtime is currently subscribed to. This is intentionally
    # decoupled from the data-collection symbol set — you can collect on 9
    # symbols but only trade 3.
    trade_symbols: Annotated[tuple[str, ...], NoDecode] = ()

    # Confidence is `P(UP) - P(DOWN)` in [-1, 1]. Trade only when the absolute
    # confidence exceeds this threshold.
    trade_conf_threshold: float = Field(0.10, ge=0.0, le=1.0)

    # Notional size of every paper trade (USD). Fixed for MVP-1.
    trade_notional_usd: float = Field(10.0, ge=0.0)

    # Round-trip taker fee (basis points), used by the paper trader to compute
    # net PnL. Binance Futures: 4 bp/side, so 8 bp round-trip.
    taker_fee_bp: float = Field(4.0, ge=0.0)

    # Maker fee (bp). Binance Futures: 2 bp default, can go negative for VIP.
    maker_fee_bp: float = Field(2.0)

    # Stop-loss in basis points (price moves against us). Set high to disable.
    trade_stop_loss_bp: float = Field(50.0, ge=0.0)

    @field_validator("trade_symbols", mode="before")
    @classmethod
    def _split_trade_symbols(cls, v):
        """Parse a comma-separated env string into a tuple of upper-case symbols."""
        if v is None or v == "":
            return ()
        if isinstance(v, str):
            return tuple(s.strip().upper() for s in v.split(",") if s.strip())
        if isinstance(v, (list, tuple)):
            return tuple(str(s).strip().upper() for s in v if str(s).strip())
        return ()

    @field_validator("bucket_bps", mode="before")
    @classmethod
    def _split_bucket_bps(cls, v):
        if isinstance(v, str):
            return tuple(int(x) for x in v.split(",") if x.strip())
        return v


settings = Settings()
