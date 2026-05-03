# adaptive_sdk (vendored)

Real-time order-flow analytics SDK. Vendored from
[nms6277-ops/heatmap-sdk](https://github.com/nms6277-ops/heatmap-sdk)
(`heatmap-sdk.zip`, `adaptive_sdk/` directory, version `1.1.0`).

## What this provides

| Module | Class | Purpose |
|---|---|---|
| `vpin.py` | `TrueVPINEngine` | Volume-time buckets, `TRADE_SIGN`/`BVC` classification, trade splitting; rolling **VPIN** (Volume-synchronised Probability of INformed trading, Easley/Lopez de Prado/O'Hara 2012) |
| `exhaustion.py` | `ExhaustionDetector` | Per-second aggressive-buy / aggressive-sell **flow Z-scores**, time-windowed price extrema, rolling realized volatility |
| `mab.py` | `ThompsonSamplingMAB` | 3-arm Gaussian Thompson Sampling (Murphy 2007 NIG conjugate) for choosing entry threshold from post-trade outcomes |
| `sdk.py` | `AdaptiveAnalyticsSDK` | Facade: per-symbol isolation, pending-signal registry, `on_trade` / `on_book_update` / `get_state` / `report_outcome` |

Zero deps beyond `numpy` (only used for the MAB RNG); normal CDF is `math.erf`.

## How opus uses it

The runtime feeds every Binance trade tick into the SDK and every top-of-book
update into `on_book_update`. At each snapshot tick (every 100 ms), we sample
`SymbolState` and append SDK-derived columns to the snapshot row:

```
sdk_vpin                 — current VPIN value
sdk_buy_flow_z           — per-second aggressive-buy flow Z-score
sdk_sell_flow_z          — per-second aggressive-sell flow Z-score
sdk_realized_vol         — rolling realized volatility (log-returns)
sdk_buckets_filled       — number of volume buckets filled (warm-up gate)
sdk_is_ready             — VPIN/Z scores are usable (warm-up complete)
```

These flow into the LightGBM feature set alongside the price-band book
features. Because the SDK consumes trades (not snapshots), the same code path
runs identically in live trading and in offline replay from the trade log.

## Why vendored vs. submodule

We control the upgrade cycle (sed-rewrite `from adaptive_sdk` ->
`from backend.adaptive_sdk` was the only required change). Tests live in
`tests/adaptive_sdk/`. To pull a newer version from upstream:

```bash
# unpack the new heatmap-sdk.zip
unzip -o heatmap-sdk.zip -d /tmp/hs
rsync -a --delete /tmp/hs/adaptive_sdk/ backend/adaptive_sdk/  # keep this README
rsync -a --delete /tmp/hs/tests_adaptive_sdk/ tests/adaptive_sdk/
sed -i 's|from adaptive_sdk\b|from backend.adaptive_sdk|g' tests/adaptive_sdk/*.py
ruff format backend/adaptive_sdk tests/adaptive_sdk
```

Calibration note from upstream README: `vpin_mid=0.30 / vpin_high=0.50` are
liquid-spot-crypto heuristics. Microcap altcoins MUST be calibrated per-symbol
from their own VPIN distribution (see `app/entry_filter.py` in the heatmap-sdk
upstream for the `AdaptiveVpinRegime` quantile-based version).
