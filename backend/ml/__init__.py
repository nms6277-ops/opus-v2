"""ML pipeline for opus MVP-1.

Modules:
- ``dataset``  : load + concatenate parquet snapshots, time-aware split
- ``labels``   : build 3-class direction targets at multiple horizons
- ``features`` : derive predictive features from raw snapshot columns
- ``train``    : LightGBM training (per-horizon, with Optuna search)
- ``eval``     : metrics + simulated paper-trading PnL on the test slice

The data collector (``backend.collector``) writes one parquet per
``{symbol}/{YYYY-MM-DD}/{HH}.parquet`` under ``OPUS_DATA_DIR/snapshots``.
The ML pipeline reads those files (read-only) and never blocks the
collector. All training is offline.
"""
