from backend.collector.lob import OrderBook, Trade
from backend.runtime import _snapshot_now_ms


def test_snapshot_now_uses_recent_exchange_time_when_local_clock_is_ahead():
    ob = OrderBook(symbol="UBUSDT")
    ob.last_update_ts_ms = 10_000
    ob.add_trade(Trade(ts_ms=10_200, price=0.1, qty=1.0, is_buyer_maker=False))

    assert _snapshot_now_ms(ob, local_now_ms=13_200) == 10_200


def test_snapshot_now_falls_back_to_local_time_without_exchange_events():
    ob = OrderBook(symbol="UBUSDT")

    assert _snapshot_now_ms(ob, local_now_ms=13_200) == 13_200
