"""Central runtime orchestrator.

Holds per-symbol:
  - OrderBook (bids/asks + recent trades)
  - SnapshotWriter (parquet)
  - Snapshot task (periodic feature writer)

Plus the two exchange WS clients and the active trader.

The API layer talks to the Runtime (add/remove symbols, switch mode).
The Runtime is the only component that *mutates* per-symbol state.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import orjson

from backend.adaptive_sdk import (
    AdaptiveAnalyticsSDK,
)
from backend.adaptive_sdk import (
    BookSnapshot as SDKBookSnapshot,
)
from backend.adaptive_sdk import (
    GlobalConfig as SDKGlobalConfig,
)
from backend.adaptive_sdk import (
    TradeTick as SDKTradeTick,
)
from backend.collector.lob import OrderBook, Trade
from backend.collector.trade_log import TradeLogWriter
from backend.collector.writer import SnapshotWriter
from backend.config import Mode, settings
from backend.exchanges.binance_private_ws import BinancePrivateWS
from backend.exchanges.binance_rest import get_client as binance_rest
from backend.exchanges.binance_ws import BinanceWS
from backend.exchanges.bybit_ws import BybitWS
from backend.features import snapshot as snap_mod
from backend.log import get_logger
from backend.ml.inference import Predictor
from backend.model_registry import discover_model_bundles
from backend.settings_store import RuntimeSettings, SettingsHardCaps, SettingsStore
from backend.state import AppState, SymbolStats, app_state
from backend.telegram import TelegramNotifier
from backend.traders.base import Trader
from backend.traders.live import LiveTrader
from backend.traders.paper import PaperTrader

log = get_logger(__name__)


_SNAPSHOT_EXCHANGE_CLOCK_MAX_SKEW_MS = 10_000
_RUNTIME_MODEL_FILE = "runtime_model.json"


def _load_runtime_model_dir(data_dir: Path, default: Path) -> Path:
    path = data_dir / _RUNTIME_MODEL_FILE
    if not path.is_file():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default
    model_dir = data.get("model_dir") if isinstance(data, dict) else None
    if not model_dir:
        return default
    saved = Path(str(model_dir))
    if not saved.is_absolute():
        saved = Path.cwd() / saved
    return saved


def _save_runtime_model_dir(data_dir: Path, model_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / _RUNTIME_MODEL_FILE
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"model_dir": str(model_dir)}, indent=2), encoding="utf-8")
    tmp.replace(path)


def _snapshot_now_ms(ob: OrderBook, local_now_ms: int) -> int:
    """Use exchange event time for coherent trade windows when local clock leads WS data."""
    exchange_now_ms = int(ob.last_update_ts_ms or 0)
    if ob.recent_trades:
        exchange_now_ms = max(exchange_now_ms, int(ob.recent_trades[-1].ts_ms))
    if exchange_now_ms <= 0:
        return local_now_ms
    if abs(local_now_ms - exchange_now_ms) > _SNAPSHOT_EXCHANGE_CLOCK_MAX_SKEW_MS:
        return local_now_ms
    return exchange_now_ms


def _fast_levels_json(levels: list) -> str:
    """Serialize depth levels [["price","qty"], ...] to a compact JSON str."""
    try:
        return orjson.dumps(levels).decode("ascii")
    except Exception:
        return "[]"


@dataclass
class SymbolCtx:
    symbol: str
    ob: OrderBook
    writer: SnapshotWriter
    raw_depth_writer: SnapshotWriter | None = None
    raw_trade_writer: SnapshotWriter | None = None
    snapshot_task: asyncio.Task | None = None
    writer_flush_task: asyncio.Task | None = None
    raw_depth_flush_task: asyncio.Task | None = None
    raw_trade_flush_task: asyncio.Task | None = None

    # Bybit-side light state (best bid/ask only — for cross-exchange features)
    by_best_bid: float = 0.0
    by_best_ask: float = 0.0

    # SDK is registered lazily on first trade; this flag avoids the dict
    # lookup on every tick once registered.
    sdk_registered: bool = False


def _sdk_state_to_columns(state) -> dict[str, float]:
    """Map an :class:`adaptive_sdk.SymbolState` to a flat column dict.

    Returns columns prefixed with ``sdk_`` so they're easy to filter at the
    feature-selection layer. Always returns the same set of keys (zero-valued
    when the SDK is still warming up) so the parquet schema stays stable.
    """
    return {
        "sdk_is_ready": 1.0 if state.is_ready else 0.0,
        "sdk_vpin": float(state.vpin),
        "sdk_buy_flow_z": float(state.buy_exhaustion_z),
        "sdk_sell_flow_z": float(state.sell_exhaustion_z),
        "sdk_realized_vol": float(state.realized_vol),
        "sdk_buckets_filled": float(state.buckets_filled),
        "sdk_pending_signals": float(state.pending_signals_count),
    }


class Runtime:
    def __init__(self, state: AppState | None = None) -> None:
        self.state = state or app_state
        self.symbols: dict[str, SymbolCtx] = {}
        self.model_dir = _load_runtime_model_dir(settings.data_dir, Path(settings.model_dir))
        self.settings_store = SettingsStore(
            path=settings.data_dir / "runtime_settings.json",
            hard_caps=SettingsHardCaps(
                hard_max_leverage=settings.hard_max_leverage,
                hard_max_live_symbols=settings.hard_max_live_symbols,
                hard_daily_loss_usd=settings.hard_daily_loss_usd,
                hard_12h_loss_usd=settings.hard_12h_loss_usd,
                hard_symbol_loss_usd=settings.hard_symbol_loss_usd,
                hard_notional_usd=settings.hard_max_notional_usd,
            ),
        )
        self.runtime_settings = self.settings_store.load()

        self.binance_ws = BinanceWS(
            on_book=self._on_binance_depth,
            on_trade=self._on_binance_trade,
            on_resync=self._on_binance_resync,
            on_state=self._on_binance_state,
        )
        self.bybit_ws = BybitWS(
            on_book=self._on_bybit_book,
            on_trade=self._on_bybit_trade,
            on_state=self._on_bybit_state,
        )
        self.binance_private_ws: BinancePrivateWS | None = None

        self._trader: Trader | None = None
        self._predictor: Predictor | None = None
        self._trade_log: TradeLogWriter | None = None
        self._telegram = TelegramNotifier(self.state)
        self._trade_log_task: asyncio.Task | None = None
        self._ws_watchdog_task: asyncio.Task | None = None
        self._stopped = False

        # Adaptive analytics SDK (VPIN, exhaustion, OBI). Pure side-channel:
        # never gates trading; only emits 'sdk_*' columns into snapshots.
        # Stays None when settings.enable_adaptive_sdk is False.
        self._sdk: AdaptiveAnalyticsSDK | None = (
            AdaptiveAnalyticsSDK(SDKGlobalConfig()) if settings.enable_adaptive_sdk else None
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        log.info("runtime: starting in mode=%s", self.state.mode.value)
        await self.binance_ws.start([])
        await self._maybe_start_bybit()
        await self._telegram.start()
        if settings.binance_api_key and settings.binance_api_secret:
            self.binance_private_ws = BinancePrivateWS(
                on_order_update=self._on_binance_order_update,
                on_account_update=self._on_binance_account_update,
                on_state=self._on_binance_private_state,
            )
            await self.binance_private_ws.start()
        else:
            log.info("runtime: Binance private WS disabled (API keys not configured)")

        # ML inference is read-only and safe to load once even in COLLECT mode;
        # if no model is configured the predictor stays disabled.
        try:
            self._predictor = Predictor(model_dir=self.model_dir)
        except Exception as e:
            log.warning("runtime: failed to construct predictor: %s", e)
            self._predictor = None

        # One trade-log writer for the whole runtime; shared across traders.
        self._trade_log = TradeLogWriter(data_dir=settings.data_dir)
        self._trade_log_task = asyncio.create_task(
            self._trade_log.periodic_flush(30.0), name="trade-log-flush"
        )

        self._trader = self._make_trader(self.state.mode)
        await self._trader.start()
        self._ws_watchdog_task = asyncio.create_task(self._ws_watchdog(), name="ws-watchdog")

    async def stop(self) -> None:
        self._stopped = True
        log.info("runtime: stopping")
        if self._trader is not None:
            await self._trader.stop()
        if self.binance_private_ws is not None:
            await self.binance_private_ws.stop()
        await self._telegram.stop()
        await self.binance_ws.stop()
        await self._maybe_stop_bybit()
        if self._ws_watchdog_task is not None:
            self._ws_watchdog_task.cancel()
        if self._trade_log_task is not None:
            self._trade_log_task.cancel()
            try:
                await self._trade_log_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._trade_log is not None:
            await self._trade_log.close()
        for ctx in list(self.symbols.values()):
            await self._stop_symbol_tasks(ctx)
            await ctx.writer.close()
        await binance_rest().close()

    def _make_trader(self, mode: Mode) -> Trader:
        if mode == Mode.LIVE:
            return LiveTrader(
                self.state,
                predictor=self._predictor,
                runtime_settings=self.runtime_settings,
                on_risk_event=self._on_risk_event,
            )
        # COLLECT and PAPER both use PaperTrader; in COLLECT mode the snapshot
        # loop just doesn't call ``on_snapshot``.
        return PaperTrader(
            self.state,
            predictor=self._predictor,
            trade_log=self._trade_log,
            runtime_settings=self.runtime_settings,
            on_risk_event=self._on_risk_event,
        )

    def model_status(self) -> dict:
        pred = self._predictor
        return {
            "enabled": bool(pred is not None and getattr(pred, "enabled", False)),
            "horizons": list(getattr(pred, "horizons", [])) if pred is not None else [],
            "model_dir": str(self.model_dir),
        }

    def available_models(self) -> list[dict]:
        active = self.model_dir.resolve()
        return [
            b.to_dict(active=(b.path.resolve() == active)) for b in discover_model_bundles(Path("./models"))
        ]

    def _has_open_positions(self) -> bool:
        trader = self._trader
        if trader is None:
            return False
        positions = getattr(trader, "_positions", None)
        if positions:
            return True
        open_symbols = getattr(trader, "_open_symbols", None)
        if open_symbols:
            return True
        return any(abs(float(s.position_base or 0.0)) > 0.0 for s in self.state.symbols.values())

    async def set_model_dir(self, model_dir: str | Path) -> dict:
        if self._has_open_positions():
            raise ValueError("cannot switch model while trader has open positions")

        next_dir = Path(model_dir)
        if not next_dir.is_absolute():
            next_dir = Path.cwd() / next_dir
        if not next_dir.is_dir():
            raise ValueError(f"model_dir does not exist: {model_dir}")

        next_predictor = Predictor(model_dir=next_dir)
        if not next_predictor.enabled:
            raise ValueError(f"model_dir has no loadable horizons: {model_dir}")

        self.model_dir = next_dir
        self._predictor = next_predictor
        if self._trader is not None and hasattr(self._trader, "predictor"):
            self._trader.predictor = next_predictor
        _save_runtime_model_dir(settings.data_dir, self.model_dir)
        log.info("runtime: switched model_dir -> %s horizons=%s", self.model_dir, next_predictor.horizons)
        return self.model_status()

    async def set_mode(self, mode: Mode) -> None:
        if mode == self.state.mode:
            return
        log.info("runtime: mode %s -> %s", self.state.mode.value, mode.value)
        if self._trader is not None:
            await self._trader.stop()
        self.state.mode = mode
        self._trader = self._make_trader(mode)
        await self._trader.start()

    # ------------------------------------------------------------------
    # Symbol management
    # ------------------------------------------------------------------
    async def add_symbol(self, symbol: str, position_size_usd: float) -> None:
        symbol = symbol.upper()
        if symbol in self.symbols:
            stats = self.state.symbols[symbol]
            stats.position_size_usd = position_size_usd
            return
        log.info(
            "runtime: add symbol %s size=$%.2f raw_depth=%s raw_trades=%s",
            symbol,
            position_size_usd,
            settings.collect_raw_depth,
            settings.collect_raw_trades,
        )
        ob = OrderBook(symbol=symbol, max_depth=settings.lob_depth)
        writer = SnapshotWriter(
            data_dir=settings.data_dir,
            symbol=symbol,
            rotation_min=settings.parquet_rotation_min,
        )
        ctx = SymbolCtx(symbol=symbol, ob=ob, writer=writer)
        if self._sdk is not None:
            try:
                self._sdk.register_symbol(symbol)
                ctx.sdk_registered = True
            except Exception as e:
                log.warning("runtime: SDK register %s failed: %s", symbol, e)
        ctx.writer_flush_task = asyncio.create_task(writer.periodic_flush(10.0), name=f"writer-{symbol}")
        if settings.collect_raw_depth:
            raw_d = SnapshotWriter(
                data_dir=settings.data_dir,
                symbol=symbol,
                rotation_min=settings.parquet_rotation_min,
                subdir="raw/depth",
                batch_rows=1000,
            )
            ctx.raw_depth_writer = raw_d
            ctx.raw_depth_flush_task = asyncio.create_task(
                raw_d.periodic_flush(10.0), name=f"raw-depth-{symbol}"
            )
        if settings.collect_raw_trades:
            raw_t = SnapshotWriter(
                data_dir=settings.data_dir,
                symbol=symbol,
                rotation_min=settings.parquet_rotation_min,
                subdir="raw/trades",
                batch_rows=2000,
            )
            ctx.raw_trade_writer = raw_t
            ctx.raw_trade_flush_task = asyncio.create_task(
                raw_t.periodic_flush(10.0), name=f"raw-trades-{symbol}"
            )
        ctx.snapshot_task = asyncio.create_task(self._snapshot_loop(ctx), name=f"snap-{symbol}")
        self.symbols[symbol] = ctx

        now = time.time()
        self.state.symbols[symbol] = SymbolStats(
            symbol=symbol,
            position_size_usd=position_size_usd,
            started_at=now,
            added_at=now,
        )

        await self.binance_ws.add_symbols([symbol])
        await self._maybe_add_bybit_symbols([symbol])

    def runtime_settings_dict(self) -> dict:
        return {
            **asdict(self.runtime_settings),
            "hard_caps": asdict(self.settings_store.hard_caps),
        }

    async def set_runtime_settings(self, updates: dict) -> dict:
        current = asdict(self.runtime_settings)
        current.update({k: v for k, v in updates.items() if v is not None and k in current})
        self.runtime_settings = self.settings_store.save(RuntimeSettings(**current))
        g = self.state.guards
        g.daily_loss_limit_usd = self.runtime_settings.daily_loss_limit_usd
        g.loss_12h_limit_usd = self.runtime_settings.loss_12h_limit_usd
        g.symbol_loss_limit_usd = self.runtime_settings.symbol_loss_limit_usd
        g.max_live_symbols = self.runtime_settings.max_live_symbols
        if isinstance(self._trader, LiveTrader):
            self._trader.runtime_settings = self.runtime_settings
        return self.runtime_settings_dict()

    async def set_symbol_execution_mode(self, symbol: str, execution_mode: str) -> dict:
        symbol = symbol.upper()
        stats = self.state.symbols.get(symbol)
        if stats is None:
            raise ValueError(f"symbol {symbol} is not in watchlist")
        if execution_mode == "live":
            stats.execution_mode = "live"
            stats.live_state = "probation_live"
            stats.live_trade_count = 0
            stats.live_wins = 0
            stats.live_losses = 0
            stats.consecutive_losses = 0
            stats.current_notional_usd = self.runtime_settings.probation_notional_usd
            stats.block_reason = ""
        elif execution_mode == "paper":
            stats.execution_mode = "paper"
            stats.live_state = "paper"
            stats.current_notional_usd = 0.0
            stats.block_reason = ""
        else:
            raise ValueError("execution_mode must be paper or live")
        return {"symbol": symbol, "execution_mode": stats.execution_mode}

    async def disable_symbol(self, symbol: str, reason: str = "") -> dict:
        symbol = symbol.upper()
        stats = self.state.symbols.get(symbol)
        if stats is None:
            raise ValueError(f"symbol {symbol} is not in watchlist")
        stats.execution_mode = "paper"
        stats.live_state = "disabled"
        stats.current_notional_usd = 0.0
        stats.block_reason = reason or "disabled by operator"
        if isinstance(self._trader, LiveTrader):
            await self._trader.flatten_symbol(symbol)
        elif self._trader is not None:
            await self._trader.cancel_all(symbol)
        return {"symbol": symbol, "disabled": True, "reason": stats.block_reason}

    async def emergency_stop(self, reason: str = "operator") -> dict:
        from backend.safety import guards as guards_mod

        guards_mod.trip_emergency(self.state.guards, reason)
        if isinstance(self._trader, LiveTrader):
            await self._trader.emergency_flatten()
        elif self._trader is not None:
            for sym in list(self.symbols):
                await self._trader.cancel_all(sym)
        return {"ok": True, "reason": reason}

    async def _on_risk_event(self, event) -> None:
        log.warning("runtime: risk event %s %s", event.scope, event.reason)
        await self._telegram.send_risk_event(event)

    async def remove_symbol(self, symbol: str) -> None:
        symbol = symbol.upper()
        if symbol not in self.symbols:
            return
        log.info("runtime: remove symbol %s", symbol)
        ctx = self.symbols.pop(symbol)
        await self._stop_symbol_tasks(ctx)
        await ctx.writer.close()
        # SDK has no unregister_symbol; leave the per-symbol context in
        # memory (small) so re-add_symbol resumes its warmup state.

        if self._trader is not None:
            await self._trader.cancel_all(symbol)

        self.state.symbols.pop(symbol, None)
        await self.binance_ws.remove_symbols([symbol])
        await self._maybe_remove_bybit_symbols([symbol])

    async def _maybe_start_bybit(self) -> None:
        if settings.enable_bybit:
            await self.bybit_ws.start([])
        else:
            log.info("runtime: Bybit WS disabled (OPUS_ENABLE_BYBIT=false)")
            self.state.bybit_connected = False
            self.state.bybit_last_msg_ts = 0.0

    async def _maybe_stop_bybit(self) -> None:
        if settings.enable_bybit:
            await self.bybit_ws.stop()

    async def _maybe_add_bybit_symbols(self, symbols: list[str]) -> None:
        if settings.enable_bybit:
            await self.bybit_ws.add_symbols(symbols)

    async def _maybe_remove_bybit_symbols(self, symbols: list[str]) -> None:
        if settings.enable_bybit:
            await self.bybit_ws.remove_symbols(symbols)

    async def _stop_symbol_tasks(self, ctx: SymbolCtx) -> None:
        for t in (
            ctx.snapshot_task,
            ctx.writer_flush_task,
            ctx.raw_depth_flush_task,
            ctx.raw_trade_flush_task,
        ):
            if t is not None:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        if ctx.raw_depth_writer is not None:
            await ctx.raw_depth_writer.close()
        if ctx.raw_trade_writer is not None:
            await ctx.raw_trade_writer.close()

    # ------------------------------------------------------------------
    # Snapshot loop
    # ------------------------------------------------------------------
    async def _snapshot_loop(self, ctx: SymbolCtx) -> None:
        interval = settings.snapshot_interval_ms / 1000.0
        while True:
            try:
                await asyncio.sleep(interval)
                sdk_cols: dict[str, float] | None = None
                if self._sdk is not None and ctx.sdk_registered:
                    try:
                        sdk_state = self._sdk.get_state(ctx.symbol)
                        sdk_cols = _sdk_state_to_columns(sdk_state)
                    except Exception as e:
                        log.debug("runtime: SDK get_state %s failed: %s", ctx.symbol, e)
                snap = snap_mod.build(
                    ctx.ob,
                    depth=settings.snapshot_depth,
                    trade_window_ms=1000,
                    now_ms=_snapshot_now_ms(ctx.ob, int(time.time() * 1000)),
                    bucket_bps=settings.bucket_bps,
                    band_bps_max=settings.band_bps_max,
                    band_bps_step=settings.band_bps_step,
                    sdk_columns=sdk_cols,
                )
                if snap is None:
                    continue
                row = snap.to_row()
                await ctx.writer.append(row)

                stats = self.state.symbols.get(ctx.symbol)
                if stats is not None:
                    stats.snapshots_written += 1
                    stats.last_snapshot_ts = time.time()
                    stats.best_bid = snap.best_bid
                    stats.best_ask = snap.best_ask
                    stats.spread_bp = snap.spread_bp
                    stats.microprice = snap.microprice

                # Hand the snapshot to the predictor / paper trader. In PAPER
                # mode we hand it to the trader, which internally calls
                # ``Predictor.predict()`` (which itself pushes the row into
                # the history buffer). In other modes we still want the
                # buffer to warm up so we call ``update()`` directly.
                # Doing both at once would push every row twice and corrupt
                # all derived features (lag/rolling/OFI) that depend on the
                # buffer contents.
                if self.state.mode == Mode.PAPER and isinstance(self._trader, PaperTrader):
                    try:
                        await self._trader.on_snapshot(ctx.symbol, row)
                    except Exception as e:
                        log.error("trader on_snapshot %s failed: %s", ctx.symbol, e)
                elif self.state.mode == Mode.LIVE and isinstance(self._trader, LiveTrader):
                    try:
                        await self._trader.on_snapshot(ctx.symbol, row)
                    except Exception as e:
                        log.error("live trader on_snapshot %s failed: %s", ctx.symbol, e)
                elif self._predictor is not None and self._predictor.enabled:
                    try:
                        self._predictor.update(ctx.symbol, row)
                    except Exception as e:
                        log.error("predictor update %s failed: %s", ctx.symbol, e)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("snapshot_loop %s error: %s", ctx.symbol, e)

    # ------------------------------------------------------------------
    # Binance callbacks
    # ------------------------------------------------------------------
    def _on_binance_depth(self, symbol: str, data: dict) -> None:
        ctx = self.symbols.get(symbol)
        if ctx is None:
            return
        ok = ctx.ob.apply_diff(data)
        stats = self.state.symbols.get(symbol)
        if stats is not None:
            stats.ws_last_msg_ts = time.time()
        if not ok and stats is not None:
            stats.ws_sequence_gaps = ctx.ob.sequence_gaps
            asyncio.create_task(self.binance_ws.request_resync(symbol))
        elif ok and self._trader is not None:
            asyncio.create_task(self._trader.on_book_update(symbol))
        # If the book is still unready (buffering pre-sync or after a gap
        # whose resync we already requested but didn't land yet), nudge
        # a resync. Cheap: the in-flight guard inside request_resync
        # collapses duplicates into one REST call.
        if not ctx.ob.ready:
            asyncio.create_task(self.binance_ws.request_resync(symbol))
        # Raw depth log: store every update (even those buffered) as a single
        # parquet row. bids/asks are kept as JSON strings to fit in a single
        # column and keep schema stable regardless of level counts.
        if ctx.raw_depth_writer is not None:
            try:
                bids = data.get("b", [])
                asks = data.get("a", [])
                asyncio.create_task(
                    ctx.raw_depth_writer.append(
                        {
                            "ts_ms": int(data.get("E", time.time() * 1000)),
                            "symbol": symbol,
                            "U": int(data.get("U", 0)),
                            "u": int(data.get("u", 0)),
                            "pu": int(data.get("pu", 0)),
                            "n_bids": len(bids),
                            "n_asks": len(asks),
                            "bids_json": _fast_levels_json(bids),
                            "asks_json": _fast_levels_json(asks),
                        }
                    )
                )
            except Exception:
                pass

    def _on_binance_trade(self, symbol: str, data: dict) -> None:
        ctx = self.symbols.get(symbol)
        if ctx is None:
            return
        try:
            ts = int(data.get("T", data.get("E", time.time() * 1000)))
            price = float(data["p"])
            qty = float(data["q"])
            is_buyer_maker = bool(data.get("m", False))
        except (KeyError, ValueError, TypeError):
            return
        ctx.ob.add_trade(Trade(ts_ms=ts, price=price, qty=qty, is_buyer_maker=is_buyer_maker))
        if self._sdk is not None and ctx.sdk_registered:
            try:
                self._sdk.on_trade(
                    SDKTradeTick(
                        symbol=symbol,
                        price=price,
                        quantity=qty,
                        is_buyer_maker=is_buyer_maker,
                        timestamp=ts / 1000.0,
                    )
                )
                # Push best-bid / best-ask volume so the SDK's OBI confidence
                # multiplier has a value (otherwise it stays at 1.0). Cheap.
                if ctx.ob.bids and ctx.ob.asks:
                    bb_p, bb_q = next(iter(sorted(ctx.ob.bids.items(), key=lambda kv: -kv[0])))
                    ba_p, ba_q = next(iter(sorted(ctx.ob.asks.items(), key=lambda kv: kv[0])))
                    self._sdk.on_book_update(
                        SDKBookSnapshot(
                            symbol=symbol,
                            best_bid=float(bb_p),
                            best_ask=float(ba_p),
                            bid_vol=float(bb_q),
                            ask_vol=float(ba_q),
                            timestamp=ts / 1000.0,
                        )
                    )
            except Exception as e:
                log.debug("runtime: SDK on_trade %s failed: %s", symbol, e)
        stats = self.state.symbols.get(symbol)
        if stats is not None:
            stats.last_trade_price = price
            stats.last_trade_qty = qty
            stats.ws_last_msg_ts = time.time()
        if ctx.raw_trade_writer is not None:
            try:
                asyncio.create_task(
                    ctx.raw_trade_writer.append(
                        {
                            "ts_ms": ts,
                            "symbol": symbol,
                            "price": price,
                            "qty": qty,
                            "is_buyer_maker": is_buyer_maker,
                            "agg_id": int(data.get("a", 0)),
                            "first_id": int(data.get("f", 0)),
                            "last_id": int(data.get("l", 0)),
                        }
                    )
                )
            except Exception:
                pass

    def _on_binance_resync(self, symbol: str, snap: dict) -> None:
        ctx = self.symbols.get(symbol)
        if ctx is None:
            return
        log.info(
            "runtime: binance resync %s (lastUpdateId=%s, %d bids / %d asks)",
            symbol,
            snap.get("lastUpdateId"),
            len(snap.get("bids", [])),
            len(snap.get("asks", [])),
        )
        was_ready = ctx.ob.ready
        ctx.ob.apply_snapshot(snap)
        ctx.ob.flush_buffer()
        if ctx.ob.ready and not was_ready:
            log.info(
                "runtime: %s book READY (last_update_id=%d, %d bids / %d asks in memory, gaps=%d)",
                symbol,
                ctx.ob.last_update_id,
                len(ctx.ob.bids),
                len(ctx.ob.asks),
                ctx.ob.sequence_gaps,
            )
        stats = self.state.symbols.get(symbol)
        if stats is not None:
            stats.ws_reconnects += 1

    def _on_binance_state(self, connected: bool, ts: float) -> None:
        self.state.binance_connected = connected
        self.state.binance_last_msg_ts = ts

    def _on_binance_private_state(self, connected: bool, ts: float) -> None:
        self.state.binance_private_connected = connected
        self.state.binance_private_last_msg_ts = ts

    def _on_binance_order_update(self, data: dict) -> None:
        # LiveTrader reconciliation is wired in the live-execution task.
        log.debug("runtime: Binance order update %s", data.get("o", {}).get("s", ""))

    def _on_binance_account_update(self, data: dict) -> None:
        # LiveTrader reconciliation is wired in the live-execution task.
        log.debug("runtime: Binance account update reason=%s", data.get("a", {}).get("m", ""))

    # ------------------------------------------------------------------
    # Bybit callbacks (read-only; used for cross-exchange features later)
    # ------------------------------------------------------------------
    def _on_bybit_book(self, symbol: str, data: dict) -> None:
        ctx = self.symbols.get(symbol.upper())
        if ctx is None:
            return
        try:
            payload = data.get("data") or {}
            bids = payload.get("b") or []
            asks = payload.get("a") or []
            if bids and bids[0]:
                ctx.by_best_bid = float(bids[0][0])
            if asks and asks[0]:
                ctx.by_best_ask = float(asks[0][0])
        except (IndexError, KeyError, ValueError, TypeError):
            return

    def _on_bybit_trade(self, symbol: str, data: dict) -> None:
        # For now we just use the signal as a heartbeat; rich feature
        # computation will come later.
        return

    def _on_bybit_state(self, connected: bool, ts: float) -> None:
        self.state.bybit_connected = connected
        self.state.bybit_last_msg_ts = ts

    # ------------------------------------------------------------------
    # Watchdog: trip kill switch on long WS silence (LIVE mode only)
    # ------------------------------------------------------------------
    async def _ws_watchdog(self) -> None:
        from backend.safety import guards as guards_mod

        while not self._stopped:
            try:
                await asyncio.sleep(0.5)
                if self.state.mode != Mode.LIVE:
                    continue
                if not self.state.binance_last_msg_ts:
                    continue
                age_ms = (time.time() - self.state.binance_last_msg_ts) * 1000.0
                if age_ms > self.state.guards.ws_stale_ms:
                    guards_mod.trip_emergency(
                        self.state.guards,
                        f"binance ws stale ({age_ms:.0f}ms)",
                    )
                    if self._trader is not None:
                        # Network kill switch: cancel orders AND flatten any
                        # open live positions via reduce-only MARKETs.
                        # cancel_all only deletes resting orders — it does
                        # NOT close positions, so an open live trade would
                        # otherwise stay exposed until horizon timeout.
                        if isinstance(self._trader, LiveTrader):
                            try:
                                await self._trader.emergency_flatten()
                            except Exception as e:
                                log.error("ws_watchdog: emergency_flatten failed: %s", e)
                        else:
                            for sym in list(self.symbols):
                                try:
                                    await self._trader.cancel_all(sym)
                                except Exception as e:
                                    log.error("ws_watchdog: cancel_all %s failed: %s", sym, e)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("ws_watchdog error: %s", e)


# Singleton runtime (one per process)
_runtime: Runtime | None = None


def get_runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        _runtime = Runtime()
    return _runtime


async def shutdown_runtime() -> None:
    global _runtime
    if _runtime is not None:
        await _runtime.stop()
        _runtime = None
