import pytest

from backend.exchanges.binance_filters import (
    SymbolFilters,
    parse_symbol_filters,
    round_price,
    round_qty_down,
    validate_notional,
)


def _exchange_info_symbol():
    return {
        "symbol": "UBUSDT",
        "pricePrecision": 5,
        "quantityPrecision": 0,
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.00010"},
            {"filterType": "LOT_SIZE", "minQty": "1", "stepSize": "1"},
            {"filterType": "MIN_NOTIONAL", "notional": "5"},
        ],
    }


def test_parse_symbol_filters():
    filters = parse_symbol_filters(_exchange_info_symbol())
    assert filters.tick_size == 0.0001
    assert filters.step_size == 1.0
    assert filters.min_qty == 1.0
    assert filters.min_notional == 5.0
    assert filters.price_precision == 5
    assert filters.quantity_precision == 0


def test_quantity_rounds_down_to_step_size():
    filters = SymbolFilters(
        tick_size=0.0001,
        step_size=0.1,
        min_qty=0.1,
        min_notional=5.0,
        price_precision=5,
        quantity_precision=1,
    )
    assert round_qty_down(12.349, filters) == 12.3


def test_price_rounds_to_tick_size():
    filters = SymbolFilters(
        tick_size=0.0005,
        step_size=1.0,
        min_qty=1.0,
        min_notional=5.0,
        price_precision=4,
        quantity_precision=0,
    )
    assert round_price(1.23476, filters) == 1.235


def test_min_notional_is_enforced():
    filters = parse_symbol_filters(_exchange_info_symbol())
    validate_notional(price=1.0, qty=5.0, filters=filters)
    with pytest.raises(ValueError, match="min notional"):
        validate_notional(price=1.0, qty=4.0, filters=filters)


def test_missing_required_filters_fail_closed():
    with pytest.raises(ValueError, match="missing"):
        parse_symbol_filters({"symbol": "BADUSDT", "filters": []})
