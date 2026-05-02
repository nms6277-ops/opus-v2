"""Binance USDT-M Futures symbol filter helpers."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Any


@dataclass(frozen=True)
class SymbolFilters:
    tick_size: float
    step_size: float
    min_qty: float
    min_notional: float
    price_precision: int
    quantity_precision: int


def _decimal_places(value: str) -> int:
    d = Decimal(value).normalize()
    return max(0, -d.as_tuple().exponent)


def parse_symbol_filters(symbol_info: dict[str, Any]) -> SymbolFilters:
    """Parse Binance exchangeInfo symbol filters and fail closed if incomplete."""

    by_type = {f.get("filterType"): f for f in symbol_info.get("filters", [])}
    price_filter = by_type.get("PRICE_FILTER")
    lot_filter = by_type.get("LOT_SIZE") or by_type.get("MARKET_LOT_SIZE")
    notional_filter = by_type.get("MIN_NOTIONAL") or by_type.get("NOTIONAL")
    if not price_filter or not lot_filter or not notional_filter:
        raise ValueError(f"missing required filters for {symbol_info.get('symbol', '<unknown>')}")

    tick_str = str(price_filter.get("tickSize", "0"))
    step_str = str(lot_filter.get("stepSize", "0"))
    min_qty_str = str(lot_filter.get("minQty", "0"))
    min_notional_str = str(notional_filter.get("notional", notional_filter.get("minNotional", "0")))

    tick_size = float(tick_str)
    step_size = float(step_str)
    min_qty = float(min_qty_str)
    min_notional = float(min_notional_str)
    if tick_size <= 0 or step_size <= 0 or min_qty < 0 or min_notional <= 0:
        raise ValueError(f"invalid filters for {symbol_info.get('symbol', '<unknown>')}")

    return SymbolFilters(
        tick_size=tick_size,
        step_size=step_size,
        min_qty=min_qty,
        min_notional=min_notional,
        price_precision=int(symbol_info.get("pricePrecision", _decimal_places(tick_str))),
        quantity_precision=int(symbol_info.get("quantityPrecision", _decimal_places(step_str))),
    )


def round_qty_down(qty: float, filters: SymbolFilters) -> float:
    """Round quantity down to the exchange step size."""

    step = Decimal(str(filters.step_size))
    q = Decimal(str(qty))
    rounded = (q / step).to_integral_value(rounding=ROUND_DOWN) * step
    return float(round(rounded, filters.quantity_precision))


def round_price(price: float, filters: SymbolFilters) -> float:
    """Round price to the nearest tick size."""

    tick = Decimal(str(filters.tick_size))
    p = Decimal(str(price))
    rounded = (p / tick).to_integral_value(rounding=ROUND_HALF_UP) * tick
    return float(round(rounded, filters.price_precision))


def validate_notional(*, price: float, qty: float, filters: SymbolFilters) -> None:
    notional = price * qty
    if qty < filters.min_qty:
        raise ValueError(f"quantity {qty} below min qty {filters.min_qty}")
    if notional < filters.min_notional:
        raise ValueError(f"notional {notional:.8f} below min notional {filters.min_notional:.8f}")
