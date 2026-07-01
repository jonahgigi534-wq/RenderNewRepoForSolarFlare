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
SWAN-SF benchmark, we compare its reported benchmark skill against an honest,
out-of-sample evaluation built directly from live JSOC/SDO magnetic-field data
and NOAA/HEK flare records. In a first held-out test (2014 Q1), the model's True
Skill Statistic (TSS) falls from **0.77 on the benchmark to 0.35 on live data at
its default operating point** — an overstatement of roughly 2×. We identify five
mechanisms that inflate benchmark scores and outline a correction based on
recalibration and training on operational data. *(Multi-period confirmation and
the correction experiment are in progress.)*

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

- **H₁ (primary):** A model scored on live, out-of-sample JSOC data shows
  significantly lower TSS than on the SWAN-SF benchmark, owing to distribution
  shift between curated and raw operational data; recalibrating or retraining on
  live data recovers a measurable fraction of the lost skill.
- **H₀ (null):** Benchmark and live operational TSS are statistically
  equivalent.

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

## 5. Preliminary Results

**Test set.** 2014 Q1 live pull: 123,309 SHARP records and 58 M/X flares across
15 active regions, yielding 3,503 labeled windows with 149 positives (4.25% base
rate).

**Benchmark vs. live skill.**

| Evaluation | TSS | Recall | Precision |
|---|---:|---:|---:|
| SWAN-SF benchmark (reported) | **0.772** | 0.805 | 0.286 |
| Live JSOC — default (balanced) threshold | **0.350** | 0.416 | 0.219 |
| Live JSOC — high-recall threshold | 0.640 | 0.859 | 0.148 |
| Live JSOC — high-precision threshold | 0.340 | 0.403 | 0.221 |

At the model's default operating point, live TSS (0.350) is **less than half**
the benchmark TSS (0.772). Recall collapses most sharply (0.805 → 0.416),
consistent with a decision threshold that is mis-placed for the live data
distribution: at a lower (high-recall) threshold the model recovers TSS 0.640 on
the same live data, indicating that much of the discriminative signal survives
but the *calibration* does not transfer from benchmark to operation.

These results support H₁ for a single period. Multi-period confirmation across
solar-cycle phases is required before rejecting H₀. *(In progress.)*

---

## 6. Discussion — Why Benchmarks Overstate Skill

We identify five mechanisms. The first two are directly evidenced by the results
above; the remaining three are common practices in the broader literature that
compound the effect.

1. **Distribution shift (train–serve skew).** Benchmarks use curated, cleaned
   magnetic parameters; deployment uses raw JSOC values. Identical feature names,
   different distributions — a model tuned on the clean version underperforms on
   the raw stream. This is the dominant contributor to the observed gap.
2. **Non-transferable calibration.** Decision thresholds fit on benchmark data
   sit incorrectly on the shifted live distribution, collapsing recall (0.805 →
   0.416 here).
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

## 7. Correction — Training and Calibrating on Live Data *(in progress)*

Planned experiment: recalibrate thresholds (and isotonic-calibrate probabilities)
on a live-JSOC *validation* period, then measure live TSS on a separate held-out
live period; and, separately, retrain on live-JSOC-derived data to remove the
distribution shift at the source. Predicted outcome: live default-threshold TSS
rises from ~0.35 toward the ~0.64 the model already reaches at a better-placed
threshold, at improved precision.

---

## 8. Limitations

- Results to date cover a **single** 3-month period (2014 Q1); the gap must be
  confirmed across multiple solar-cycle phases.
- Solar-minimum periods contain few M+ flares, limiting statistical power there.
- TSS depends on operating point; comparisons are reported per threshold.
- The correction experiments are not yet complete.

---

## 9. Conclusion *(preliminary)*

A SHARP-based flare model reporting TSS 0.77 on the SWAN-SF benchmark achieves
TSS 0.35 on out-of-sample live satellite data at its default operating point — a
~2× overstatement — with evidence pointing to distribution shift and
non-transferable calibration as primary causes. If confirmed across periods and
paired with an effective correction, this establishes a reproducible method for
honestly evaluating — and improving — operational solar-flare forecasts.

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
