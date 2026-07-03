# Pre-registration — the 2024 cross-cycle generalization test

**Status: committed BEFORE fetching or scoring any 2024 data.** This document
fixes the hypotheses, the frozen decision thresholds, and the predicted outcome
in advance, so the 2024 result is a genuine out-of-sample test rather than a
number found by searching. Commit hash of this file is the timestamp of record.

## Why 2024

Every operational evaluation year in the project so far (2013–2017) is **solar
cycle 24** — the same cycle the deployed model trained on (SWAN-SF 2010–2012).
2024 is **cycle 25 near maximum**: a different cycle, a fresh active-region
population, ~616 M+ flares, and — critically — a HEK/SWPC AR-attribution rate of
~0.94, well above our 0.8 label-quality gate (unlike 2023's 0.15, which the gate
excludes). 2024 is therefore the first *true cross-cycle* generalization test and
the honest replacement for what 2023 was meant to show.

## Frozen decision thresholds (do not re-fit on 2024)

All four are the values already committed to the repo; every one was fit on
benchmark validation or an *earlier* operational year, never on 2024. Source:
`research/results/multiperiod_rescore.json` (`by_operating_point`) and
`MODEL_CARD.md` §1.

| Operating point | Threshold | Objective / origin |
|---|---|---|
| `balanced` (F1 default) | **0.104** | max F1 on benchmark validation |
| `high_precision` | **0.107** | precision-weighted on benchmark validation |
| `high_recall` (= benchmark-val TSS) | **0.031** | max TSS on benchmark validation |
| `operational` (deployed) | **0.021** | TSS-objective, recalibrated on live 2013 |

## Hypotheses and predicted outcomes

- **H1 — the F1 default degrades again across the cycle boundary.** Predict
  `balanced` (0.104) live TSS on 2024 in the **0.35–0.55** band — well below the
  SWAN-SF benchmark TSS 0.77, consistent with the 0.35 (2014) / 0.53 (2017) it
  posted within cycle 24.
- **H2 — the TSS-objective thresholds transfer across the cycle boundary.**
  Predict `high_recall` (0.031) and `operational` (0.021) live TSS on 2024 in the
  **0.72–0.85** band, matching their within-cycle transfer (0.82–0.84 on unseen
  cycle-24 years).
- **H3 — the benchmark still overstates.** Predict the benchmark-minus-operational
  TSS gap at the deployed threshold stays **positive** (> 0) on 2024.

A result that contradicts H1/H2 (e.g. the F1 default *also* transfers to 2024, or
the TSS point collapses cross-cycle) is a real, reportable falsification — that is
the point of registering the prediction first.

## Protocol — run exactly once

1. **Gate first (fail-closed).** Measure 2024 label attribution and record it:
   `python scripts/label_attribution.py 2024 2024` → add the rate to config
   `scorecard.label_attribution_by_year`. Proceed only if ≥ 0.8; otherwise 2024 is
   excluded and this test is void (report that honestly).
2. **Build the dataset once** (definitive science series, as for every training
   build): `python -m solarflare.sharpdata --start 2024-01-01 --end 2025-01-01 --out data/sharp_live/dataset_2024.npz`
3. **Score once** with the deployed model at the four frozen thresholds above —
   no threshold search, no retraining. Extend `research/exp1_multiperiod_rescore.py`
   with a `2024_cross_cycle` period; cluster-bootstrap CIs (whole active regions),
   same protocol as every other period.
4. **Report** the four TSS values with CIs against the three predictions above,
   pass or fail, in RESULTS.md — before touching the manuscript.

Determinism: fixed seed 42 for bootstrap; JSOC/HEK are historical queries and
return identical records for identical date ranges, so the single run is
reproducible.

## What is NOT frozen

Nothing about 2024 data has been examined at commit time. If any threshold above
is later found to have been silently changed, this pre-registration is void.
