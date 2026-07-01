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
  operational data. **Recalibrating / retraining on live data recovers a
  measurable fraction of the lost skill.**
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
   (solar-max 2014, quiet 2018–19, solar-max 2023–24).
2. **Diagnose the cause** — compare SHARP feature distributions (curated SWAN-SF
   vs. raw JSOC); KS-tests to show distribution shift is real.
3. **The fix** — recalibrate thresholds + isotonic-calibrate on a live-JSOC
   *validation* period; measure live TSS on a *separate* held-out live period.
4. **Robustness** — repeat across solar-cycle phases; bootstrap 95% CIs on TSS.

## Controls & rigor
- Leakage-free chronological splits with multi-day gaps (never shuffled).
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
- [x] **Exp 1: multi-period live re-scores (2014 / 2015 / 2023) — DONE.** Benchmark
      TSS 0.77 vs live default TSS 0.35 / 0.64 / 0.33; H₀ rejected in all three
      periods (benchmark above upper 95% CI). Gap is condition-dependent.
- [x] Bootstrap 95% CIs on TSS (part of Exp 4) — DONE.
- [x] Accuracy-illusion demonstration (zero-skill model vs accuracy) — DONE (paper §7).
- [ ] Exp 2: feature-distribution comparison + KS-tests (diagnose distribution shift).
- [ ] Exp 3: recalibrate/retrain on live JSOC; before/after on held-out live period.
- [ ] Exp 4 (remainder): more windows per phase to tighten CIs.
- [ ] Physics interpretation: PCA / feature-importance of SHARP features.
- [ ] Final figures (skill-vs-period plot) + verified citations + author names.
