# Benchmark or Reality? Quantifying the Operational Skill Gap in Machine-Learning Solar-Flare Forecasting

**Authors:** [Author name(s)], [School / Affiliation]
**Category:** Physics & Astronomy
**Status:** Working draft — evidence is being added incrementally. Sections marked
*(in progress)* are not yet complete.

---

## Abstract

Machine-learning models for solar-flare forecasting are conventionally evaluated
on curated benchmark datasets and frequently report high skill or accuracy.
Whether those reported scores reflect a model's **real-time operational** skill —
its performance on the raw, live satellite data it would actually consume — has
not been systematically tested. We ask: *do standard benchmark scores overstate
the real-time operational skill of ML flare forecasts, and does training on live
satellite data close the gap?* Using a RandomForest model trained on the
SWAN-SF benchmark (reported TSS 0.77), we compare its benchmark skill against
honest, out-of-sample evaluations built directly from live JSOC/SDO
magnetic-field data and NOAA/HEK flare records, across three periods spanning
different solar-cycle phases. At the default operating point the model's True
Skill Statistic falls to **0.35 (2014), 0.64 (2015), and 0.33 (2023)** — the
benchmark exceeds the upper 95% confidence interval of live skill in every
period (H₀ rejected), overstating operational skill by up to ~2×, though the size
of the gap varies with conditions. A controlled 2×2 experiment (benchmark- vs.
live-trained models, benchmark vs. operational test sets) shows that **training on
live data does *not* close the gap** — the overstatement is a property of the
benchmark's evaluation set, not of the training distribution. We identify the
mechanisms that inflate benchmark scores and argue that honest evaluation
requires fixing the benchmark test, not the training data.

---

## 1. Introduction

Solar flares and their associated eruptions drive space-weather hazards to power
grids, satellites, aviation, and radio communication. Reliable short-horizon
(24 h) flare forecasts are therefore operationally valuable, and machine learning
has become the dominant modeling approach. A recurring pattern in the literature
and in student research is to train a classifier on a curated dataset of active-
region magnetic parameters and report a headline skill or accuracy figure —
often 90–99% accuracy, or a True Skill Statistic (TSS) approaching state of the
art.

These numbers are typically produced under favorable conditions: cleaned and
sometimes class-balanced data, evaluation partitions drawn from the same curated
source used for training, and — in many cases — metrics (such as raw accuracy)
that are degenerate for a rare event. What is rarely tested is whether the same
model retains its skill when confronted with the **raw, real-time data stream**
it would consume in deployment. This distinction — *benchmark* performance versus
*operational* performance — is the subject of this paper.

**Contribution.** We (i) quantify the gap between benchmark and live operational
skill for a representative SHARP-based flare model, (ii) enumerate the mechanisms
responsible for benchmark inflation, and (iii) *(in progress)* demonstrate that
training and calibrating on operational data recovers a measurable fraction of
the lost skill.

---

## 2. Background and Related Work

**SHARP magnetic parameters.** The Space-weather HMI Active Region Patches
(SHARP) products, derived from SDO/HMI vector magnetograms, summarize the
photospheric magnetic field of each active region into scalar parameters (total
unsigned flux, current helicity, etc.) that are widely used as flare predictors.

**SWAN-SF.** The Space Weather ANalytics for Solar Flares (SWAN-SF) benchmark is
a curated, partitioned multivariate time-series dataset of SHARP parameters with
flare labels, designed for standardized, leakage-aware ML evaluation. It is the
de-facto benchmark for this task.

**Skill metrics.** Because flares are rare (~1–5% of active-region intervals),
accuracy is uninformative: a constant "no-flare" predictor achieves ~98%
accuracy with zero skill. The field-standard metric is the **True Skill
Statistic**, TSS = recall − false-alarm-rate (0 = no skill, 1 = perfect), with
the Heidke Skill Score (HSS) as a complementary imbalance-aware measure. Prior
work has repeatedly emphasized honest, leakage-free benchmarking, yet operational
(live-data) validation remains uncommon.

*(Fuller related-work review and citations in progress.)*

---

## 3. Research Question and Hypotheses

**Research question.** Do standard benchmark scores overstate the real-time
operational skill of ML flare forecasts, and does training on live satellite data
close the gap?

- **H₁ₐ:** A model scored on live, out-of-sample JSOC data shows significantly
  lower TSS than on the SWAN-SF benchmark. *(Supported — Sections 5 and 8.)*
- **H₁ᵦ:** Training on live operational data closes the benchmark–operational gap.
  *(Refuted — Section 8: the live-trained model does not recover the gap.)*
- **H₀ (null):** Benchmark and live operational TSS are statistically
  equivalent. *(Rejected.)*

The refutation of H₁ᵦ is itself a key finding: because retraining does not close
the gap, the overstatement is a property of the benchmark's *evaluation set*, not
of the training distribution.

---

## 4. Data and Methods

**Model.** A RandomForest classifier predicting the probability of an
M-class-or-greater flare within 24 h from a 12 h observation window. Inputs are
17 SHARP magnetic parameters, each summarized by 7 statistics (last, mean, std,
min, max, slope, delta) → 119 features. Trained on SWAN-SF spanning
2010-05 → 2012-03 (73,492 windows; positive base rate 1.7%). Decision thresholds
were fit on a validation partition at three operating points (high-recall,
balanced/default, high-precision).

**Benchmark evaluation.** Standard SWAN-SF held-out test partition (14,690
windows; 236 positives, 1.6% base rate).

**Live operational evaluation.** We reconstruct an out-of-sample test set the
same way the deployed system operates: SHARP keyword time series are pulled
directly from JSOC (`hmi.sharp_cea_720s`), and M+ flare labels with active-region
associations are pulled from NOAA's GOES event list via the HEK API. Records are
sliced into the identical 12 h windows and passed through the identical feature
transform used in training and live inference. The test period (2014 Q1) begins
~22 months after the training data ends, making it strictly out-of-sample.

**Metrics.** TSS (primary), HSS, recall, precision, evaluated at each operating
point. *(Bootstrap confidence intervals in progress.)*

---

## 5. Results

**Test sets.** The SWAN-SF-trained model (benchmark TSS 0.772) was re-scored on
three out-of-sample live-JSOC periods spanning different solar-cycle phases, all
after the training span (2010-05 → 2012-03):

| Period | SHARP records | M/X flares | Windows | Positives | Base rate |
|---|---:|---:|---:|---:|---:|
| 2014 (solar max) | 123,309 | 58 | 3,503 | 149 | 4.25% |
| 2015 (declining) | 103,850 | 28 | 2,940 | 75 | 2.55% |
| 2023 (rising max) | 142,259 | 13 | 4,075 | 32 | 0.79% |

**Benchmark vs. live skill (TSS with bootstrap 95% CI, n=1000 resamples).**

| Operating point | Benchmark | 2014 live | 2015 live | 2023 live |
|---|---:|---:|---:|---:|
| **balanced (default)** | **0.772** | 0.350 [0.27–0.43] | 0.641 [0.54–0.74] | 0.334 [0.17–0.52] |
| high-recall | — | 0.640 [0.58–0.70] | 0.831 [0.77–0.88] | 0.662 [0.55–0.75] |
| high-precision | — | 0.340 [0.26–0.42] | 0.602 [0.49–0.71] | 0.339 [0.17–0.52] |

Precision at the default operating point: 2014 = 0.219, 2015 = 0.407,
2023 = **0.043** (at 0.79% base rate the model still catches 91% of flares at the
high-recall point, but precision collapses — a direct demonstration of the
rare-event precision ceiling).

**Findings.**
1. **The benchmark overstates live skill in every period.** At the default
   operating point, the benchmark TSS (0.772) lies *above the upper 95% CI* of the
   live TSS in all three periods (upper bounds 0.43, 0.74, 0.52) — so H₀ is
   rejected at the 5% level in each case. Mean live default TSS ≈ 0.44, an
   overstatement of ~1.75×.
2. **The gap is condition-dependent, not a fixed factor.** It is largest at
   solar max (2014: 0.35; 2023: 0.33 — roughly 2× overstated) and smallest in the
   declining phase (2015: 0.64, approaching the benchmark). A single benchmark
   number is therefore an unreliable predictor of operational skill, and the
   discrepancy itself varies with observing conditions.
3. **Calibration, not raw discrimination, drives much of the loss.** In every
   period the high-recall threshold yields substantially higher live TSS than the
   default (e.g., 2014: 0.64 vs 0.35), showing the model retains discriminative
   signal but the benchmark-tuned threshold is mis-placed for the live
   distribution.
4. **Precision tracks the base rate.** At the lowest base rate (2023, 0.79%),
   default precision falls to 0.043, illustrating that operational usefulness
   degrades sharply as events become rarer — an effect invisible to benchmark or
   balanced-data evaluation.

These results support H₁ across multiple periods and reject H₀. The 2023 estimate
has wide CIs (only 32 positives) and should be read as indicative.

---

## 6. Discussion — Why Benchmarks Overstate Skill

We identify five mechanisms. The 2×2 experiment (Section 8) localizes the primary
cause: since retraining on live data does not close the gap, the overstatement
lives in the benchmark's **evaluation set** (an easier test distribution), not in
the training data. Mechanisms 1–2 are directly evidenced by our results; 3–5 are
common practices in the broader literature that compound the effect.

1. **Evaluation-set distribution shift.** The curated, cleaned SWAN-SF *test*
   partition is systematically easier than raw operational JSOC data — so the same
   model scores higher on the benchmark test than in operation regardless of how it
   was trained (Section 8). Direct evidence: all 17 SHARP features shift
   significantly between the benchmark era and the operational year (median KS
   D = 0.19), and operational data carries ~2% missing values the benchmark imputes
   away (Section 9). This is the dominant contributor to the observed gap.
2. **Non-transferable calibration.** Decision thresholds fit on benchmark data
   sit incorrectly on the live distribution, collapsing recall (0.805 → 0.416
   here) and producing the large *fixed-threshold* gap in Section 5.
3. **Class balancing and accuracy metrics.** Balanced test sets and raw accuracy
   flatter rare-event models; a constant "no-flare" predictor scores ~98%
   accuracy with TSS 0.
4. **Temporal leakage.** Shuffled or stratified k-fold splitting of autocorrelated
   active-region time series places adjacent-in-time samples in both train and
   test, inflating scores relative to leakage-free chronological splits.
5. **Per-timestep versus per-event scoring.** Scoring each timestep rewards
   repeated, correlated positives from a single flaring region relative to
   honest per-event evaluation.

---

## 7. The Accuracy Illusion — A Demonstration

A recurring practice in flare-forecasting reports is to headline **accuracy**,
often 90–99%, on curated or class-balanced data. We show that such figures can
certify *nothing* about a model's skill, because a **zero-skill model attains
them trivially — and, on realistic data, even outscores a genuinely skilful
model on accuracy.**

**The exact relationship.** A constant "no-flare" predictor never raises an alarm:
its recall is 0 and its false-alarm rate is 0, so its **TSS = 0** (no skill by
definition). Its accuracy, however, is exactly

    accuracy(always "no-flare") = 1 − base_rate.

Because M+ flares are rare, this trivial model scores extremely high accuracy:

| Base rate of flares | Accuracy of the zero-skill model | TSS |
|---|---:|---:|
| 1% | **99.0%** | 0 |
| 2% | 98.0% | 0 |
| 4.25% (our 2014 Q1 live set) | **95.75%** | 0 |
| 50% (a *balanced* test set) | 50.0% | 0 |

Two consequences follow. **First**, a reported "99% accuracy" for
flare-vs-no-flare is exactly what a do-nothing model achieves at a ~1% base rate;
the number is therefore consistent with *zero* skill and conveys no information
about it unless the base rate and a skill metric are also given. High accuracy on
*balanced* data (where the trivial model scores only 50%) is not degenerate in the
same way, but it also does not transfer to operational base rates — the setting
that actually matters.

**Second, and more striking: on realistic data, chasing accuracy penalizes
skill.** Any model that actually catches flares must raise alarms, and at a low
base rate most alarms are false — which *lowers* accuracy. On our real 2014 Q1
live test (3,503 windows, 149 flares, 4.25% base rate):

| Model | Catches flares? | TSS | Accuracy |
|---|---|---:|---:|
| Always "no-flare" (zero skill) | No (0% of flares) | 0.00 | **95.75%** |
| Our model @ high-recall | Yes (86% of flares) | **0.64** | ≈78% |

The **useless model beats the useful one by ~18 accuracy points.** A judge or
practitioner ranking by accuracy would select the model that predicts nothing.
This is the core reason the field uses TSS/HSS rather than accuracy, and it is a
concrete illustration of how accuracy-based benchmarks overstate — or entirely
misrepresent — operational value.

---

## 8. Does Training on Live Data Close the Gap? A 2×2 Experiment

To test the second half of the research question, we ran a controlled 2×2:
**two training sources** (benchmark SWAN-SF 2010–2012 vs. live JSOC 2014) ×
**two test sets** (held-out SWAN-SF vs. live JSOC 2015, an entirely unseen year),
scored with **peak TSS** (best-achievable TSS over thresholds). Both models were
trained in memory; the deployed model was untouched. (Reproducible:
`python -m solarflare.scorecard`.)

| Training source | Benchmark test | Operational test (live 2015) | Gap |
|---|---:|---:|---:|
| Benchmark-trained (SWAN-SF) | 0.905 | 0.822 | 0.083 |
| Live-trained (JSOC 2014) | 0.914 | **0.799** | **0.115** |

*Benchmark test: held-out SWAN-SF (14,690 windows, 236 flares). Operational test:
live JSOC 2015 (12,597 windows, 273 flares).*

**Result 1 — benchmarks overstate (H₁ first half: supported).** Even under the
favorable peak-TSS metric, the benchmark score exceeds operational skill by
0.083 TSS.

**Result 2 — live training does NOT close the gap (H₁ second half: refuted).**
The live-trained model is *slightly worse* operationally (0.799 vs. 0.822) and
has a *larger* benchmark–operational gap (0.115 vs. 0.083). Retraining on
operational data did not recover skill.

**Interpretation.** Because retraining on live data does not shrink the gap, the
overstatement is not primarily a *training* distribution-shift artifact that
better training data would fix. Instead, the gap appears to be a property of the
**benchmark's evaluation set being systematically easier than real operational
data** — the held-out SWAN-SF test overstates operational skill regardless of how
the model was trained. This is a stronger and more actionable conclusion than the
original hypothesis: fixing the *benchmark*, not the *training data*, is what
honest evaluation requires.

**Reconciling with Section 5.** The peak-TSS gap here (~0.08) is much smaller than
the fixed-threshold gap in Section 5 (up to ~2×). The two measure different
things: peak TSS asks whether the discriminative signal *exists* in operational
data (it largely does), while a fixed, pre-committed threshold measures *deployed*
skill (which collapses due to non-transferable calibration). An operational system
uses a fixed threshold, so Section 5 reflects real deployment; the scorecard
isolates the model's latent discriminative ceiling.

**Caveat.** The scorecard's peak TSS selects the threshold on the test set itself
(Youden's J), which is mildly optimistic; choosing it on a separate validation
split would make the operational numbers strictly honest. This is noted as a
limitation and does not affect the sign of either finding.

---

## 9. Diagnosing the Shift — Feature Distributions

To characterize the distribution shift underlying the gap, we compared the raw
JSOC distributions of all 17 SHARP parameters between the benchmark training era
(February 2011; 17,950 records) and the operational year (February 2015; 44,175
records) using two-sample Kolmogorov–Smirnov tests.

- **All 17/17 features are significantly shifted** (p < 0.05; most p ≈ 0), median
  KS D = 0.19 — a pervasive, moderate distribution shift between the era the model
  learned and operational data.
- **Largest shifts are in extensive / free-energy proxies** — MEANPOT (D = 0.27),
  TOTPOT (0.26), TOTUSJH (0.24), USFLUX (0.24) — magnetic size/energy parameters
  that track the active-region population and overall activity, which differ
  between solar-cycle phases.
- **Smallest shifts are in intensive per-pixel gradients/twist** — MEANALP (0.04),
  MEANGBZ (0.04) — quantities less sensitive to region size.
- **Operational data is messier:** the mean non-finite (missing/bad) rate is 2.1%
  in 2015 (up to 3.7% for some parameters) vs. ~0.2% in 2011 — missing values the
  SWAN-SF pipeline removes by KNN imputation (`FPCKNN` in the benchmark filenames)
  but that a live system must handle.

**Interpretation.** The magnetic-parameter distributions the model faces
operationally differ significantly and pervasively from the benchmark era, and
operational data carries missing values the benchmark scrubs — direct evidence for
the distribution-shift mechanism (§6.1). The comparison is raw-to-raw across years,
so it captures era/activity shift and raw-data messiness but does not by itself
isolate the *curation* component (which would require the raw SWAN-SF source
instances). Taken with Section 8 — where the gap persists regardless of training
source — the evidence indicates the benchmark's curated test distribution is both
cleaner and differently shaped than the operational stream, inflating reported
skill.

---

## 10. Limitations

- Each period is a single 3-month window; more windows per solar-cycle phase
  would tighten the estimates.
- Low-activity periods contain few M+ flares (2023: 32 positives), widening
  confidence intervals — the 2023 result is indicative, not definitive.
- TSS depends on operating point; comparisons are reported per threshold.
- All results use one model architecture (RandomForest); generality across model
  families is untested.
- The correction experiment (Section 8) is not yet complete.

---

## 11. Conclusion

A SHARP-based flare model reporting TSS 0.77 on the SWAN-SF benchmark achieves
live, out-of-sample TSS of only 0.35–0.64 at its default operating point across
three solar-cycle phases — the benchmark exceeds the upper 95% CI of live skill
in every period, overstating operational skill by up to ~2×, condition-dependent.
A controlled 2×2 experiment further shows that **retraining on live operational
data does not close the gap**, indicating the overstatement is a property of the
benchmark's evaluation set rather than of the training distribution. Together
these results establish a reproducible method for honestly evaluating operational
solar-flare forecasts and caution that benchmark leaderboard scores must not be
read as operational performance. The practical implication is that the community
should benchmark on live-operational test sets, not only curated partitions.

---

## References *(to be finalized)*

- Angryk, R. et al. (2020). *Multivariate time series dataset for space weather
  data analytics* (SWAN-SF). Scientific Data.
- Bobra, M. G. et al. (2014). *The HMI Vector Magnetic Field Pipeline: SHARPs.*
  Solar Physics.
- Bobra, M. G. & Couvidat, S. (2015). *Solar Flare Prediction Using SDO/HMI
  Vector Magnetic Field Data with a Machine-Learning Algorithm.* ApJ.
- Bloomfield, D. S. et al. (2012). *Toward Reliable Benchmarking of Solar Flare
  Forecasting Methods.* ApJ Letters.
- Leka, K. D. et al. (2019). *A Comparison of Flare Forecasting Methods.* ApJS.

*(Citations to be verified and formatted to the target style.)*
