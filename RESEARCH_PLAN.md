# Research Plan — Helios / Benchmark-vs-Reality in Solar-Flare Forecasting

**Category:** Physics & Astronomy
**Project type:** Research (hypothesis-driven), with a real-time engineering artifact as the instrument.

---

## Research question
**Do standard benchmark scores overstate the real-time operational skill of ML
solar-flare forecasts, and does training on live satellite data close the gap?**

## Hypotheses
- **H₁ (primary):** A flare-forecasting model scored on live, out-of-sample
  JSOC/SDO data will show **significantly lower TSS** than on the SWAN-SF
  benchmark, because of distribution shift between the curated benchmark and raw
  operational data. **Re-selecting the decision threshold (for the
  base-rate-robust TSS objective) recovers a measurable fraction of the lost
  skill.**
- **H₀ (null):** Benchmark TSS and live operational TSS are statistically
  equivalent — no significant benchmark inflation.

## Why it matters
Solar flares threaten power grids, satellites, aviation, and communications.
Published and competition ML flare forecasts routinely report **accuracy (90–99%)**
on curated/balanced datasets. Accuracy is meaningless for a ~1–5% base-rate event,
and benchmark scores may not reflect real operational skill. If forecasts are
trusted operationally, the honesty of their reported skill is a safety question.

## Variables
| Type | Variable |
|---|---|
| **Independent** | Evaluation regime (SWAN-SF benchmark vs. live JSOC); calibration/training source (SWAN-SF vs. live JSOC) |
| **Dependent** | Skill: **TSS** (primary), HSS, recall, precision |
| **Controlled** | Same 17 SHARP features, 12 h→24 h window, M+ ≥ threshold label, model architecture, random seed, leakage-free chronological splits |

## Method (4 experiments)
1. **Establish the gap** — score the SWAN-SF-trained model on (a) SWAN-SF test
   partition and (b) live JSOC data across **multiple independent periods**
   (2014 solar max, 2015 declining, 2017 declining/X-flare era; candidate years
   must pass the label-quality gate below).
2. **Diagnose the cause** — compare SHARP feature distributions (curated SWAN-SF
   vs. raw JSOC); KS-tests to show distribution shift is real.
3. **The fix** — recalibrate thresholds + isotonic-calibrate on a live-JSOC
   *validation* period; measure live TSS on a *separate* held-out live period.
4. **Robustness** — repeat across solar-cycle phases; bootstrap 95% CIs on TSS.

## Controls & rigor
- Leakage-free **region-disjoint** chronological splits — a whole active region
  lives in exactly one of train/val/test (never shuffled). The storm model
  additionally enforces a 5-day temporal gap between splits.
- **Fail-closed label-quality gate** — a year is scored only with a measured
  HEK AR-attribution rate ≥ 0.8 (2023 = 0.15 → excluded; unmeasured years are
  excluded until measured).
- TSS/HSS, not accuracy (accuracy is degenerate at low base rate).
- Out-of-sample by design (test periods years outside training).
- Bootstrap confidence intervals; the gap must exceed CIs to reject H₀.
- Fully reproducible: fixed seeds, versioned data, all code in this repo.

## Preliminary evidence (already collected)
Live-JSOC 2014 Q1 (3,503 windows, 149 M+ flares, out-of-sample; model trained
2010-05 → 2012-03):

| | TSS | Recall | Precision |
|---|---|---|---|
| SWAN-SF benchmark (reported) | **0.772** | 0.805 | 0.286 |
| Live JSOC @ default threshold | **0.350** | 0.416 | 0.219 |
| Live JSOC @ high-recall | 0.640 | 0.859 | 0.148 |

→ Benchmark overstates live default-threshold skill by ~2×. Consistent with H₁;
awaiting multi-period confirmation (Exp 1) and the calibration fix (Exp 3).

## Expected result / contribution
"Standard benchmarks overstate operational solar-flare forecasting skill by
~2×; the gap is distribution shift and is partially correctable by operational
calibration" — a reproducible method for honest space-weather forecast evaluation.

## Status / to-do
- [x] Preliminary gap measured (2014 Q1).
- [x] **Exp 1: multi-period live re-scores (2014 / 2015 / 2017) — DONE.** Benchmark
      TSS 0.77 vs live default TSS 0.35 / 0.64 / 0.53; benchmark above the live
      point estimate in all three periods and above the cluster-bootstrap 95% CI
      in 2014/2017. (2023 excluded: label attribution 0.15.) Gap is
      condition-dependent.
- [x] Cluster-bootstrap 95% CIs on TSS (whole active regions resampled) — DONE.
- [x] Accuracy-illusion demonstration (zero-skill model vs accuracy) — DONE (paper §7).
- [x] **Exp 2: feature-distribution KS-tests — DONE.** Raw JSOC 2011 vs 2015: all
      17/17 SHARP features significantly shifted (median KS D 0.19); operational
      data has ~2% missing values vs ~0.2% in the benchmark era. Evidences the
      distribution-shift mechanism. (TODO: obtain raw SWAN-SF tarballs to isolate
      the curation component from the era component.)
- [x] **Exp 3: 2×2 (benchmark- vs live-trained × benchmark vs operational test) — DONE.**
      Benchmarks overstate (gap 0.083 peak TSS); training on live data does NOT
      close the gap (H₁ᵦ refuted) → gap is an evaluation-set property. Reproducible
      via `python -m solarflare.scorecard`. (TODO: choose peak-TSS threshold on a
      validation split, not the test set, to remove mild optimism.)
- [x] **Exp 4 (the validated fix, decomposed) — DONE.** Three frozen thresholds
      tested on unseen 2015/2017: default (val-F1) 0.64→0.835 and 0.53→0.841
      (paired gains +0.19 [0.06,0.63] and +0.31 [−0.18,0.38]). The benchmark's
      own val-TSS threshold performs ≈ identically to the live-recalibrated one
      (live increment ≤ +0.02) — the fix is the threshold OBJECTIVE (TSS vs
      F1), not live data (`research/exp4_recalibration.py`, Fig 7, paper §8).
      Deployed: `sharp_live.operating_point: operational`.
- [ ] More windows per phase to tighten CIs further (optional).
- [x] **Second-architecture replication (LightGBM) — DONE.** Identical protocol on
      the rebuilt SWAN-SF p1 set (73,492 samples, exact match to the deployed
      model's training data): benchmark TSS 0.917 vs live 0.51–0.74 at the
      like-for-like threshold — positive gap in every period, every operating
      point. Gap is architecture-independent (`research/exp5_second_model.py`).
- [x] **Physics interpretation — DONE.** RF feature importance: top 5 params
      (TOTUSJH, TOTUSJZ, R_VALUE, USFLUX, TOTPOT) = 72% of decisions. PCA: PC1 = 33%
      (AR size/energy axis). Key insight: most-important features = most-shifted
      features, explaining operational degradation. (Paper §10.)
- [x] **Board figures — DONE (6 figures in `figures/`).**
- [ ] Verified citations + author name(s).
- [x] **Scorecard optimism fixed in code** — thresholds now chosen on a held-out
      validation split, not the test set (`scorecard.py` `_val_selected_tss`).
      TODO (team): re-run `python -m solarflare.scorecard` where the .npz datasets
      live to refresh skill_scorecard.json + dashboard + Figure 2.
