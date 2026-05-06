# Results

A consolidated record of what we have actually trained, backtested, and
forward-tested in `opus-v2`. This is the empirical evidence behind the
recommendations in `docs/ROADMAP.md`.

The bottom line is up front so anyone reading this in 6 months can act
on it immediately:

> **Short-horizon direction prediction with 100 ms time-bar features,
> single-day training window, taker execution on 4-bp-spread micro-cap
> altcoins is not profitable.** Spread + commission (~5–10 bp) eats any
> edge the model can extract on average. Two of seven symbols looked
> profitable on the held-out test split (LABUSDT +7.04 bp, UBUSDT
> +6.74 bp on the 15 s horizon) but **collapsed on a fresh out-of-sample
> 6-hour window** (LABUSDT -15.67 bp / hit 0.407, UBUSDT -4.15 bp /
> hit 0.611). The numbers we believed were edge were a mix of selection
> bias, regime fit, and a single-day training window that does not
> generalise forward.

What we did NOT prove false:
- The system is correct (no bugs visible in execution, accounting, or
  guards). The infrastructure is reusable for any next iteration.
- Microstructure signal exists. UBUSDT held above 0.5 hit rate on the
  15 s horizon out-of-sample; UBUSDT `maker_zero` was the only positive
  bp number on the fresh 6 h slice. The amount of signal is just not
  large enough to clear taker round-trip cost on a 4 bp spread.

---

## 1. Data and setup

| Item | Value |
|---|---|
| Snapshot cadence | 100 ms (v2 default; was 250 ms in v1) |
| Snapshot row size | ~100 numeric columns: top-of-book, top-N depth, ±50 bp / 5 bp band buckets, 7 adaptive_sdk columns |
| Training window | 24 h (rolling tail) |
| Symbols collected | BIOUSDT, BRUSDT, BUSDT, LABUSDT, SKYAIUSDT, TSTUSDT, UBUSDT |
| Total rows / symbol | ~864 000 / day (24 h × 36 000/h) |
| Horizons trained | 2 s, 5 s, 15 s |
| Label mode | `cost_mode="taker"` (spread-aware: row is UP/DOWN only if the round-trip beats spread + commission + 3 bp safety margin) |
| Train/val/test split | 70/15/15 by time, no shuffle |
| Backtest threshold | top-5% of `|P(UP) - P(DOWN)|` → ~250 trades / 1 h slice |
| Taker round-trip cost in bp | spread (~3–4 bp) + 2 × 4 bp commission = ~10–12 bp |

---

## 2. Phase 3, round 1 — global model on all 7 symbols

**Setup.** One LightGBM multi-class classifier per horizon, with
`symbol` added as a categorical feature. Trained on 24 h pooled across
all 7 symbols.

**Backtest summary** (test slice, top-5% trades):

| Horizon | n_trades | taker.avg_net_bp | taker.hit | maker_zero.avg_net_bp | sharpe(taker) |
|---|---|---|---|---|---|
| 2 s | 1238 | **-9.03** | 0.510 | +1.86 | -0.49 |
| 5 s | 1234 | **-7.51** | 0.592 | +3.24 | -0.37 |
| 15 s | 1217 | **-7.69** | 0.648 | +3.33 | -0.25 |

**Verdict.** All three horizons negative on taker. The model captures
some directional information (hit rate above 0.5 on 5 s and 15 s) but
the move size is below the round-trip cost. `maker_zero` is positive
on every horizon but those numbers are a theoretical upper bound (queue
position and adverse selection are unmodelled — see §6).

**Decision.** Switch to per-symbol training. Pooling 7 micro-caps with
different microstructure regimes was muddling the signal.

---

## 3. Phase 3, round 2 — per-symbol models, in-sample (test split)

**Setup.** One LightGBM bundle per (symbol, horizon), 24 h training
window per symbol, evaluated on each symbol's own held-out test slice
(`--window 1h`).

**Backtest summary on the 15 s horizon:**

| Symbol | n_trades | taker.avg_net_bp | hit | maker_zero.avg_net_bp | sharpe(taker) | Verdict |
|---|---|---|---|---|---|---|
| LABUSDT | 243 | **+7.04** | 0.749 | +16.16 | +0.29 | candidate |
| UBUSDT | 258 | **+6.74** | 0.783 | +17.95 | +0.13 | candidate |
| BIOUSDT | 244 | -0.09 | 0.884 | +9.58 | -0.01 | maker-only |
| BRUSDT | 248 | -8.95 | 0.696 | +1.60 | -0.54 | not deployable |
| BUSDT | 247 | -12.16 | 0.484 | +0.12 | -0.31 | not deployable |
| SKYAIUSDT | 247 | -12.98 | 0.417 | -1.04 | -0.18 | not deployable |
| TSTUSDT | 250 | -31.89 | **0.260** | -20.23 | -0.79 | anti-signal |

Other horizons were uniformly worse:

- **2 s**: every symbol negative on taker (-7 to -20 bp); hit rates
  cluster around 0.42–0.62. Spread cost dominates.
- **5 s**: only LABUSDT positive (+0.15 bp); UBUSDT showed an
  anti-predictive 0.378 hit rate, suggesting mean-reversion dominates
  this horizon while h15s captures the directional component.

**Selection-bias check.** With 7 independent symbols × 3 horizons, by
chance alone we would expect a few positive bars. The empirical
distribution (2 clearly positive, 1 break-even, 3 negative, 1
anti-signal) is consistent with a mix of real microstructure
heterogeneity and selection effects. The contrast in TSTUSDT (h2s/h5s
hit ≈ 0.5–0.6 but h15s hit 0.260) is itself a sanity check: the
architecture is reading **something** from the book, just not the same
something on every symbol.

**Decision.** Forward-test the two candidates (LABUSDT and UBUSDT) on
data the model never saw. Promotion to paper or live without a forward
test would have been hope, not evidence.

---

## 4. Phase 3, round 3 — out-of-sample on a fresh 6 h slice

**Setup.** Without retraining, replay the same LABUSDT and UBUSDT
models on **6 hours of data collected after** the training window. This
is the cheapest substitute for a paper-mode forward test: the model has
literally never observed any of these timestamps.

**LABUSDT — fresh 6 h slice:**

| Horizon | n_trades | taker.avg_net_bp | hit | maker_zero.avg_net_bp |
|---|---|---|---|---|
| 2 s | 993 | -10.04 | 0.471 | -0.90 |
| 5 s | 994 | -11.15 | 0.379 | -2.08 |
| 15 s | 987 | **-15.67** | **0.407** | **-6.39** |

**UBUSDT — fresh 6 h slice:**

| Horizon | n_trades | taker.avg_net_bp | hit | maker_zero.avg_net_bp |
|---|---|---|---|---|
| 2 s | 993 | -9.55 | 0.568 | +0.48 |
| 5 s | 992 | -10.74 | 0.451 | -0.43 |
| 15 s | 990 | **-4.15** | **0.611** | **+5.93** |

**Comparison vs. in-sample test split:**

| Symbol | h15s taker (test split) | h15s taker (fresh 6 h) | Δ | hit (test) | hit (fresh) |
|---|---|---|---|---|---|
| LABUSDT | +7.04 bp | -15.67 bp | **-22.7 bp** | 0.749 | **0.407** |
| UBUSDT | +6.74 bp | -4.15 bp | -10.9 bp | 0.783 | 0.611 |

**Read.**

- **LABUSDT collapsed.** Hit rate dropped from 0.749 to 0.407 — the
  model now picks the wrong direction more often than a coin flip on
  fresh data. The +7 bp on the test split was either (a) a regime that
  ended within hours, (b) statistical luck on a 243-trade sample, or
  (c) subtle leakage we could not find. Either way the signal does not
  exist on out-of-sample 15 s windows.
- **UBUSDT degraded but not dead.** Hit 0.611 is still above 0.5, and
  `maker_zero` is the only positive bp number across both symbols on
  the fresh slice. There is *some* signal, but it is below the spread
  cost in taker mode. To trade it on taker we would need either a wider
  edge (different target) or a different cost regime (different venue,
  rebated maker, larger/more liquid symbol).

**Verdict.** The direction-prediction-on-time-bars-in-taker approach
does not work on these symbols at this data window. Stop the road
forward.

---

## 5. Symbol-level observations worth recording

- **TSTUSDT, h15s.** Hit rate 0.260 — anti-predictive. On h2s/h5s the
  model is roughly random (hit 0.49–0.60). This is consistent with a
  newly-listed coin where 15 s moves are dominated by news/pump flow
  rather than book microstructure; the model trained on its own past
  hour reads the book correctly for 5 s noise but mis-reads anything
  longer. *Useful as a signal that the architecture does read the book —
  it just gets fed a regime where the book does not predict 15 s
  direction.*
- **BIOUSDT.** Hit rate 0.884 on h15s but taker net 0.00 bp. The
  classifier gets the direction right almost every time, but the
  directional move is exactly the size of the spread + commission — many
  small wins offset by a few larger losses. In a no-spread regime
  (`maker_zero` +9.58 bp) the model is excellent. This is the textbook
  case for maker mode if and only if the maker fill assumptions can be
  validated outside backtest.
- **BRUSDT.** Hit 0.696 / taker -8.95 bp. Direction is right ~70%, but
  the wins are smaller than the losses; net negative after costs.
- **SKYAIUSDT, BUSDT.** Hit ≈ 0.42–0.48, every horizon negative. The
  microstructure is either too thin or has a different generating
  process than what 24 h of band-bucket features capture.

---

## 6. Why the maker_* numbers are not believable

The backtest emits four columns per horizon: `taker`, `maker_best`,
`maker_zero`, `maker_cross`. **Only the `taker` column corresponds to
what `paper.py` / `live.py` actually executes.** The maker columns are
implemented as "if the limit at our quoted price filled, the P&L would
be the mid-to-mid return minus the maker fee." They are useful as a
ceiling but they overstate any realistic maker outcome because:

1. **Queue position is unmodelled.** Public depth diffs do not tell us
   where in the queue a hypothetical limit would sit. The default
   "bid hit at any t+1 ≤ bid_t" treats us as front-of-queue every
   time. Real-world fill rate at front-of-queue on micro-caps is
   typically 1–3 %, not 5 %.
2. **Cancellations and re-quotes are unmodelled.** Real market makers
   cancel and re-post 10–50 times per second. We hold a limit for a
   full snapshot interval.
3. **Adverse selection is not priced.** A maker fills exactly when an
   informed taker comes for the other side. Backtest fills are
   selection-blind; live maker fills are biased toward "we got hit
   right before the move went against us."

**A `maker_zero +16 bp` line in our backtest does not mean "+16 bp on
real money." It means: if we could perfectly post-only with a 5 %
fill rate at zero adverse selection, the model would have edged that
much.** Validation of any maker strategy must be done in paper-mode or
live, not backtest. This is recorded at the top of `ROADMAP.md` so it is
not forgotten the next time the +16 bp number tempts us.

---

## 7. What we actually built and verified (independent of the ML result)

The infrastructure work landed in PR #2 stands on its own:

- **Spread-aware labelling** (`backend/ml/labels.py`): replaces the v1
  mid-to-mid label with a taker round-trip P&L thresholded against
  per-symbol spread + commission + 3 bp margin. Forces the model to fire
  only on rows where a real round-trip beats cost. This was the right
  call: it surfaces the truth (`taker.avg_net_bp` numbers) instead of
  hiding it behind mid-to-mid magic.
- **100 ms snapshots + ±50 bp / 5 bp wide-band buckets**
  (`backend/features/snapshot.py`): tick-size-independent features so a
  model trained on AIOTUSDT generalises to NEIROUSDT or BTCUSDT without
  retraining the column layout.
- **Vendored `adaptive_sdk`** (`backend/adaptive_sdk/`): VPIN,
  exhaustion detector, Thompson MAB. Their state is sampled into the
  parquet rows as `sdk_*` columns per snapshot.
- **Per-symbol live state machine** (`backend/traders/live.py`,
  `live_state.py`, `safety/regime.py`): `paper / probation_live /
  active_live` with rolling-window regime guards (win rate, net bp,
  drawdown, loss streak). Trips drop a symbol back to `paper` and start
  a cooldown.
- **Hard-cap clamp at boot** (`backend/state.py::Guards.__post_init__`):
  guards initialise from `min(soft, hard)` so even the boot envelope is
  bounded; the UI cannot raise above hard caps.
- **Memory-stable training**
  (`backend/ml/train.py::_xy`): pre-allocated C-order float32 buffer
  filled column-by-column; avoids the 2× transient peak from Polars
  Fortran-ordered `to_numpy` + `np.ascontiguousarray` that OOM'd on a
  16 GB Windows box on a 2 M-row dataset.
- **Exit-path predict failure does not flag the symbol**
  (`backend/traders/live.py::_prediction(reject_on_failure=...)`): a
  predictor exception during exit monitoring no longer flashes a
  misleading "predict failed" UI block reason. Stop-loss and timeout
  still fire on `pred=None`.
- **Backtest summary persistence** (`backend/ml/backtest.py`): JSON
  reports per horizon and an aggregated summary saved next to the
  models so historical comparisons are reproducible.

---

## 8. How to reproduce the experiments

(Paths are the operator's Windows box for these results; substitute
your data dir.)

### 8.1 Train per-symbol on 24 h

```powershell
cd D:\opus-v3
.\.venv\Scripts\python.exe -m backend.ml.train `
    --symbols BIOUSDT BRUSDT BUSDT LABUSDT SKYAIUSDT TSTUSDT UBUSDT `
    --mode per-symbol `
    --window 24h `
    --models-dir models_100msSDK
```

Output: `models_100msSDK\per_symbol\<SYM>\h{H}\model.lgb` (+ `meta.json`,
`eval.json`).

### 8.2 In-sample backtest on test slice

```powershell
foreach ($s in @("BIOUSDT","BRUSDT","BUSDT","LABUSDT","SKYAIUSDT","TSTUSDT","UBUSDT")) {
    Write-Host "=== $s ===" -ForegroundColor Cyan
    .\.venv\Scripts\python.exe -m backend.ml.backtest `
        --symbols $s `
        --window 1h `
        --models-dir "models_100msSDK\per_symbol\$s"
}
```

Read `taker avg_net=...bp` per horizon from each summary.

### 8.3 Forward test on a fresh window

After collecting 6+ extra hours of data, re-run the backtest with the
**same models** (do not retrain) and `--window 6h`. The label column is
recomputed from the new parquet rows; the model sees timestamps it
never saw at training time. This is the cheapest valid out-of-sample
test we can run without spinning up a paper-mode bot.

```powershell
.\.venv\Scripts\python.exe -m backend.ml.backtest `
    --symbols LABUSDT `
    --window 6h `
    --models-dir "models_100msSDK\per_symbol\LABUSDT"
```

### 8.4 Inspect a snapshot manually

To debug a single parquet row (book + bands + adaptive_sdk +
trade-window stats), open the file in Python:

```python
import polars as pl
df = pl.read_parquet("path/to/HH.parquet")
print(df.columns)
print(df.tail(1).transpose())
```

The expected schema is documented in `backend/features/snapshot.py`.

---

## 9. Things we deliberately did NOT do (yet)

- **Paper-mode forward test on VPS for 24-48 h.** Recommended next
  step had we continued; obsoleted by the 6 h cold replay in §4 which
  already collapsed.
- **Maker-mode architecture.** Post-only LIMIT with queue management,
  partial fills, fallback MARKET on timeout. Backtest would have lied
  (see §6); only paper / live can validate. Not done.
- **Optuna hyperparameter search.** All numbers above use the fixed
  default LightGBM config. Tuning typically buys 0.5–2 % AUC; would not
  flip the verdict.
- **Volume / dollar bars.** Time-bar aggregation mixes active and idle
  regimes; dollar bars normalise by activity and tend to produce more
  stationary distributions. See `docs/ROADMAP.md` §1 for sizing math.
- **Event-prediction targets** (vacuum / sweep / mean-reversion).
  Direction prediction was the wrong target for our cost regime;
  predicting structural events (and trading them with limit orders or
  with much larger expected moves) is plausibly the right shape. See
  `docs/ROADMAP.md` §2 for an honest analysis of which of those three
  ideas are worth implementing in our setup.
