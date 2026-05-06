# Roadmap

Where to take `opus-v2` from here. Each direction below is sized
honestly: what it is, why it might help, what work it requires, what
risks it does NOT remove, and a personal call on probability of paying
out. **This document is the answer to the question "we paused; if we
ever come back, what's worth trying?"**

Read `docs/RESULTS.md` first — it documents what we tried and what
broke. Several of the recommendations here are direct responses to
specific failure modes recorded there.

---

## 0. The cardinal rule (do not lose this)

> **Maker performance cannot be backtested honestly with the data we
> have.** Queue position, partial fills, cancellations, re-quotes and
> adverse selection are all unmodelled in `backend/ml/backtest.py`. The
> `maker_best / maker_zero / maker_cross` columns are theoretical
> ceilings, useful for sizing the gap to paper-mode reality, **never
> useful as deployment evidence on their own**. Any maker strategy must
> be validated with a paper-mode forward test (no real money but real
> book ticks) for at least 24-48 h before live promotion.

If you ignore this rule you will look at a `maker_zero +16 bp` cell and
talk yourself into a "+16 bp strategy" that bleeds money in production.

---

## 1. Volume / Dollar bars

**The idea.** Replace the 100 ms time-bar pipeline with bars closed
when a fixed dollar amount has traded. Time bars mix active and idle
periods (Asia low-vol vs. US news spike); dollar bars normalise by
activity, return distributions become more stationary, and ML models
typically learn faster on them. Reference: López de Prado, *Advances in
Financial ML*, ch. 2.

**Sizing for our scale.** LABUSDT 24 h volume ≈ $1.1 B → ≈$12.7 K/sec
average. Dollar-bar density we can expect:

| Bar size | Bars/sec | Bars/24 h | Bars/5 days |
|---|---|---|---|
| $1 000 | ~12.7 | ~1.1 M | ~5.5 M |
| $10 000 | ~1.27 | ~110 K | ~550 K |
| $50 000 | ~0.25 | ~22 K | ~110 K |

For comparison, our 100 ms time-bar pipeline produces ~864 K rows / 24 h.
Dollar bars at $10 K give a comparable count with much better
stationarity properties.

**Why this might help us specifically.** The fresh-6-hour collapse on
LABUSDT (`docs/RESULTS.md` §4) was partly an activity-regime mismatch:
the test split happened to overlap with one regime; the fresh 6 h was a
different regime. Dollar bars sample uniformly *per unit of activity*,
so each bar is equally informative; the model is less likely to overfit
the activity profile of the training window.

**Implementation cost (estimate).**
- New bar generator in `backend/collector/` triggered off the trade
  stream rather than a periodic timer. ~2 days.
- Rewrite `backend/ml/dataset.py` to load bar-events instead of
  fixed-rate snapshots; `backend/ml/labels.py` to express horizons in
  bar-counts (`+N_bars`) instead of milliseconds. ~1-2 days.
- Adaptive SDK is already event-driven, no change required.

**What this does NOT solve.** Spread cost on taker is unchanged; the
fundamental "edge < spread" problem remains. Dollar bars improve the
*quality* of signal extraction, not its *magnitude*.

**Probability of paying out (alone): 15-25 %.** Bars are necessary
infrastructure for several other directions; on their own they probably
move our taker numbers from "-7 bp" toward "-2 to +1 bp" but don't
flip the sign decisively.

---

## 2. Event-prediction targets (vs. direction prediction)

The deeper critique of round 1-3 is that we trained the model to
predict **price direction**. That is a bad target for two reasons:

1. **Cost regime.** A 2-5 bp directional move can't beat a 5-7 bp
   round-trip cost; only larger and rarer moves clear the bar.
2. **Information regime.** Direction at 15 s on a micro-cap is mostly
   driven by exogenous flow (news, listing pumps, whales), which is
   not in the order book. We have an edge in **microstructure**, which
   is *book behaviour*, not *price behaviour*.

Three event-prediction targets are worth recording:

### 2.1 Liquidity Vacuum Prediction

**Target.** Within the next 5 s, the qty on best-bid or best-ask drops
below 10 % of its trailing MA AND the spread widens by more than X
ticks.

**Predictability.** Plausibly real. Liquidity removal is auto-correlated
across MMs — when one pulls, others follow.

**Implementation.** ~1-2 days for a new label in `labels.py` (binary,
"vacuum within +5 s window") + a binary classifier in `train.py`. All
inputs are already in our snapshot rows.

**Why we should NOT prioritise this.** The natural execution is "post a
limit at the wide-spread level expecting crowd panic." That is exactly
what a real market maker *avoids*: filling into a vacuum is filling on
adverse selection. To *take* a vacuum profitably you need sub-ms
latency to beat HFT firms with co-location, which we do not have on a
2 GB / 2 vCPU VPS. The bp number in backtest will look great; the live
result will be punishing.

**Probability of paying out: 10-15 %.** Exclude unless we radically
upgrade execution venue and latency budget.

### 2.2 Aggressive Sweep Detection

**Target.** A cluster of market-buy ticks at >10× the rolling average
of buy volume, in the direction of a visible wall in the book; the
wall thins as the cluster fires; predict that the wall is about to be
swept and chase with a market order.

**Predictability.** Real but ephemeral.

**Implementation.** ~2-3 days. We have aggregated buy/sell flow per
100 ms (`buy_volume_win`, `sell_volume_win`) and band-bucket depth at
+5 / +10 / +25 bp. Wall-thinning across snapshots is a `diff` on the
bucket columns.

**Why we should NOT prioritise this.** This is a classic momentum-chase
target where co-located firms see the wall thinning 100 ms before any
VPS does. The backtest does not model latency, so it overstates the
edge. Live-trading the same signal puts us at the back of every
aggression event we trigger on.

**Probability of paying out: 5-10 %.** Skip.

### 2.3 Mean-Reversion after Toxic Flow

**Target.** A large market order moves the price by >X bp on Y volume,
but the depth at +5 to +25 bp behind the move did NOT shrink (real
support intact). Predict that price reverts at least 50 % of the move
within the next 5-15 s.

**Predictability.** Real. This is the textbook
information-vs-noise-flow distinction; the depth response to flow is
the discriminator. Our `sdk_vpin` column is already a noisy proxy for
"flow is toxic" and is in every snapshot row.

**Implementation.** ~2-3 days for new conditional labels in `labels.py`
(`y_revert_h{H} = 1` iff (|move at t-1| > X bp) AND (depth at +5 bp
unchanged) AND (price reverts ≥50 % within H seconds)) + retraining
the classifier with these as targets.

**Why this is the most realistic of the three.**
- **Latency-tolerant.** Reversion plays out over seconds; 100-300 ms
  decision latency is fine.
- **Compatible with taker.** The trade is "post-event entry into a
  reverting move." If the expected reversion is 10-20 bp (not just
  5 bp), it can clear taker round-trip cost.
- **Compatible with maker** if we ever build that path. A limit on the
  retrace is a natural execution.
- **The signal is real**: information-flow vs. noise-flow distinction
  is one of the longest-validated microstructure asymmetries
  (Easley-O'Hara, Cont).

**What this does NOT solve.**
- Sample sparsity. Toxic-flow events with depth confirmation occur
  maybe 100-500 times / day per symbol, not 850 K. We need 5-7+ days
  of collected data per symbol before this can train.
- Out-of-sample validation. Like every other target, "looks great on
  test split" must be confirmed on a held-out fresh window before
  paper, before live.

**Probability of paying out: 30-40 %.** This is the one direction in
the event-prediction family that justifies the effort. *If* we come
back to opus-v2, this is where to start.

---

## 3. More data / longer training window

**The idea.** 24 h is probably below the minimum stationarity scale
for these symbols. Microcap microstructure changes intra-day (Asia /
EU / US sessions, news, listings, whale flow). Re-train every 6-12 h
on a rolling 7-14 day window.

**Why this might help us specifically.** The LABUSDT collapse from
`hit=0.749` (test split) to `hit=0.407` (fresh 6 h) is the exact shape
you would expect from regime overfitting on a too-short training
window. A wider window would either (a) average across regimes and
flatten the test-split number toward zero (truth) or (b) actually capture
multiple regimes and let the model learn cross-regime invariants.

**Implementation cost.**
- Change nothing in code; just collect 7-14 days and run the existing
  pipeline. Free engineering time, costs disk + collector uptime.
- *Optional follow-up*: rolling-retrain harness with cron + atomic
  model swap. ~1-2 days when we want it.

**Probability of paying out: 20-30 %.** Cheap, mandatory, but does not
on its own change the fundamental cost-vs-edge math.

**This is the strict prerequisite for §2.3 above.** Event-target labels
are too sparse on 24 h windows.

---

## 4. Larger / more liquid symbols

**The idea.** Move from 4 bp-spread micro-caps (LABUSDT, UBUSDT, ...) to
0.5–1 bp-spread mid-caps (SOLUSDT, SUIUSDT, NEARUSDT, ...) or majors
(BTCUSDT, ETHUSDT). Round-trip cost drops from ~10 bp to ~2-3 bp.

**Trade-off.**
- **Pro:** Even a 3-5 bp directional edge clears the cost barrier on
  more liquid symbols. Fill quality is dramatically better. Latency
  tolerance is much higher because spreads are stable.
- **Con:** The microstructure edge is also smaller. Majors are
  dominated by HFT MM activity; a 100 ms VPS bot is not the marginal
  trader. A model that found 7 bp on LAB might find 0 bp on BTC.
- **Con:** `OPUS_TRAIN_SYMBOL_BLOCKLIST` excludes BTC/ETH by default
  precisely because their <1 bp spreads would distort our spread-aware
  labelling. Lifting that requires re-tuning the threshold heuristic
  in `labels.py`.

**Implementation cost.** Configuration only — add the symbols to
collection, override the blocklist for training, retrain. ~0.5 days.

**Probability of paying out: 15-25 %.** Worth a single-symbol pilot
(e.g. SOLUSDT or DOGEUSDT) before any architectural change.

---

## 5. Maker-mode architecture (post-only LIMIT)

**The idea.** Replace the MARKET orders in `backend/traders/live.py`
with post-only LIMIT orders, with a queue-management layer (re-quote
on tick changes), partial-fill handling, and a MARKET fallback after a
configurable timeout. Capture the spread + maker rebate instead of
paying the spread + taker fee.

**Why we did NOT do this in PR #2.**
- **Backtest cannot validate it.** See §0. There is no honest way to
  show that "maker_zero +16 bp" in our backtest survives queue / fill /
  adverse-selection costs. Any deployment must be paper or live.
- **It is real engineering.** Estimate ~3-5 days code + 8-10 new tests:
  - Post-only LIMIT path in `binance_rest.py` and `live.py`
    (already partly there for TP/SL but not for entries)
  - Queue tracker per symbol (track our remaining qty + position in
    queue heuristically)
  - Partial-fill aggregator (`live.py` currently treats fills as
    atomic)
  - Cancel-and-replace policy (re-quote on tick / time / book change)
  - MARKET fallback after a timeout
  - Slippage accounting for the partial-then-fallback case
  - Tests for adverse-selection scenarios

**When it would be worth doing.**
1. We have 5-7+ days of data and the per-symbol model on the longer
   window still shows meaningfully positive `maker_*` numbers, AND
2. We have a paper-mode forward test plan ready to run for 24-48 h
   immediately after the implementation, AND
3. We are willing to live-pilot with $6-10 notional and tight regime
   guards even if paper looks merely break-even (because the backtest
   numbers aren't trustworthy and paper is the first signal we believe).

**Probability of paying out: 25-40 %.** Higher than vanilla taker on
the same models because we capture spread instead of paying it; lower
than naively reading our backtest because real fill quality on
micro-caps is brutal.

---

## 6. Alternative strategies (different problem entirely)

For completeness — these are NOT extensions of the current ML pipeline
but distinct projects that share infrastructure (collector, exchanges,
guards, UI):

- **Funding rate arbitrage.** Hold delta-neutral perp + spot positions
  through funding payments. Predictable cash flow, no microstructure
  ML required. Capacity is small per symbol but stacks well.
- **Liquidation hunting.** Detect imminent liquidation cascades from
  open-interest + funding + price velocity, position into the cascade.
  Dirty, requires careful risk sizing.
- **Statistical arbitrage between pairs.** Cointegration + ML on the
  spread. Slower-moving, longer holding period. Backtest reasonably
  honest.
- **Cross-exchange arbitrage (Binance ↔ Bybit).** Latency-bound; we
  already collect Bybit. Probably below our latency floor for hard
  arb but viable for slower stat-arb on the spread.

These are listed for completeness, not recommended. Each is a 3-6 week
project with its own data, safety, and execution requirements.

---

## 7. Recommended sequence (if we resume)

If the project is reactivated, the cheapest credible path to a
yes/no answer on "is there a real edge here" is:

1. **Collect 7-14 days of data.** ~free engineering, requires
   collector uptime. Prerequisite for everything below. (§3)
2. **Re-run per-symbol training on the longer window** with the
   existing time-bar pipeline. If h15s taker turns positive on the
   longer window AND survives a fresh-window forward test, we have a
   candidate without doing anything new. ~1 day. (§3)
3. **Add dollar bars** as a parallel pipeline (do not replace
   time-bars). Compare side-by-side on the same symbols. ~3-5 days.
   (§1)
4. **Add the mean-reversion-after-toxic-flow target.** Train and
   forward-test with the 7-14 day data. ~3 days. (§2.3)
5. **Pilot a more liquid symbol** (SOLUSDT or DOGEUSDT) end-to-end. If
   the cost regime alone is enough to flip the sign, that is a useful
   signal even before the maker work. ~0.5 days. (§4)
6. **Only if 4+5 are positive on forward tests**, build maker-mode and
   validate in paper before any live promotion. ~5 days. (§0, §5)

This sequence costs ~2-3 weeks of engineering and ~2 weeks of data
collection. Each stage produces evidence that argues either for the
next stage or for terminating the project.

---

## 8. What is permanently true regardless of direction

These are the parts of `opus-v2` that we believe are correct and
useful regardless of which direction is taken next:

- The collector and LOB reconstruction (futures sync rule, sequence-gap
  handling).
- Spread-aware labelling (`cost_mode="taker"`).
- Adaptive SDK columns (`sdk_*`) in every snapshot.
- The full safety stack: pre-trade `Guards` + post-trade `regime` +
  hard-cap clamping at boot + per-symbol live state machine.
- The training memory pattern (column-by-column C-order build) for
  Polars → numpy → LightGBM on Windows.
- Backtest output schema (one JSON per horizon + a summary file in
  `{models_dir}/backtest/`).

When the next experiment runs, it should layer on top of this
infrastructure rather than rebuild it.
