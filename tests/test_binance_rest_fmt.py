"""Unit tests for the decimal formatter used on Binance order fields."""

from backend.exchanges.binance_rest import _fmt_decimal


def test_fmt_decimal_avoids_scientific_notation_for_small_values():
    # These are representative micro-cap prices where Python's default
    # ``f"{x}"`` switches to scientific notation, which Binance rejects.
    assert "e" not in _fmt_decimal(0.00000995).lower()
    assert "e" not in _fmt_decimal(9.95e-06).lower()
    assert _fmt_decimal(0.00001194) == "0.00001194"
    assert _fmt_decimal(0.000008) == "0.000008"


def test_fmt_decimal_keeps_human_readable_for_normal_values():
    assert _fmt_decimal(1.2) == "1.2"
    assert _fmt_decimal(0.0) == "0"
    assert _fmt_decimal(10) == "10"
    # Trailing zeros get stripped but the integer part is preserved.
    assert _fmt_decimal(1.20) == "1.2"


def test_fmt_decimal_rounds_to_eight_places_by_default():
    # Beyond 8 decimal places the value is truncated at the format step.
    out = _fmt_decimal(0.123456789012345)
    assert "." in out
    decimals = out.split(".")[1]
    assert len(decimals) <= 8
