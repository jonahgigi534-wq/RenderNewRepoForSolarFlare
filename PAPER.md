# Benchmark or Reality? Quantifying the Operational Skill Gap in Machine-Learning Solar-Flare Forecasting

**Authors:** Jonathan Gigi, Alfred Antony, Aidan George — Cypress Woods High School
**Category:** Physics & Astronomy
**Status:** Working draft. All four experiments are complete; remaining items are
final citations, the scorecard regeneration with validation-selected thresholds,
and author details. Every result is reproducible from the scripts in
[`research/`](research/README.md).

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
skill for a representative SHARP-based flare model, (ii) enumerate and evidence
the mechanisms responsible for benchmark inflation, and (iii) show via a
controlled 2×2 experiment that *retraining* on operational data does **not**
close the gap, while *recalibrating* the operating threshold on operational data
recovers most of the lost skill.

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
  *(Refuted for single-year retraining — Section 8; multi-year training narrows
  the gap but not statistically significantly.)*
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
transform used in training and live inference. All three test periods (2014,
2015, 2023) begin at least ~22 months after the training data ends, making them
strictly out-of-sample.

**Metrics.** TSS (primary), HSS, recall, precision, evaluated at each operating
point, with percentile-bootstrap 95% confidence intervals on TSS (n = 1,000
resamples, fixed seed).

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

**Benchmark vs. live skill (TSS with bootstrap 95% CI, n=1000 resamples; Figure 1).**

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

**Architecture replication — the gap is not a RandomForest quirk.** A **LightGBM**
gradient-boosted model trained under the identical protocol (same SWAN-SF p1
training set, region-disjoint chronological split, thresholds tuned on validation
only; `research/exp5_second_model.py`) shows the same pattern. At its high-recall
operating point (threshold 0.056, the like-for-like comparison since its balanced
threshold degenerated), LightGBM scores **TSS 0.917 on the benchmark test** but
only **0.559 [0.48–0.63] (2014), 0.743 [0.65–0.83] (2015), and 0.513 [0.34–0.66]
(2023)** on live data — a positive benchmark-vs-live gap in every period, at every
operating point, mirroring the RandomForest. Two architectures on the flare task,
plus the storm model in a different physical domain (Section 8), now show the
same optimism: the gap is a property of the **evaluation regime**, not of any one
model family. Notably, LightGBM's validation-tuned balanced threshold also failed
to transfer *within* the benchmark itself (validation TSS 0.553 → test 0.318),
an independent demonstration that threshold calibration, not discrimination, is
the fragile component.

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
live test (3,503 windows, 149 flares, 4.25% base rate; Figure 4):

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
live JSOC 2015 (12,597 windows, 273 flares). See Figure 2.*

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

**Most of the lost skill is recoverable — by calibration, not retraining.** The
large default-threshold drop (0.77 → 0.35) is dominated by *miscalibration*, not by
loss of discriminative ability: at a properly chosen threshold the model's
operational skill rises to **TSS ≈ 0.82** (peak; ≈0.79 mean under the regenerated
frozen-threshold protocol across four years — see the robust regeneration below). The
discriminative signal therefore largely survives operation — what fails to transfer
is the *threshold*. Re-selecting the operating point recovers most of the real-world
skill, whereas retraining on live data does not (operational TSS 0.822 → 0.799).
The actionable fix is **recalibration on operational data**, not more training data.

**The fix, validated out-of-sample.** To rule out hindsight, we tested
recalibration as a deployable *procedure* (`research/exp4_recalibration.py`): the
decision threshold was re-selected on live 2014 Q1 only (0.035, vs. the
benchmark-tuned default 0.104), **frozen**, and then evaluated on two periods it
never saw:

| Unseen period | TSS @ default 0.104 | TSS @ frozen recalibrated 0.035 | Gain |
|---|---:|---:|---:|
| 2015 (declining) | 0.641 [0.53–0.75] | **0.835 [0.77–0.89]** | +0.19 |
| 2023 (rising max) | 0.334 [0.17–0.52] | **0.661 [0.54–0.76]** | +0.33 |

The bootstrap 95% CIs do not overlap in either period — the transferred threshold
significantly beats the benchmark-tuned default on data years removed from its
calibration period, and on 2015 the recalibrated *live* skill (0.835) exceeds the
model's own *benchmark* score (0.77). The trade-off is explicit: the recalibrated
point is recall-heavy (2015: recall 0.93 at precision 0.20, vs. 0.67/0.41 at the
default), so operators trade more false alarms for far fewer missed flares.
Operational recalibration is thus a validated, deployable correction — completing
the answer to the research question: benchmarks overstate, retraining does not
help, **recalibration does.**

**Robust regeneration (v2).** The scorecard has since been regenerated with a
statistically hardened protocol (`python -m solarflare.reproduce`, RESULTS.md):
**four** operational years (2013/2015/2016/2017), **cluster** bootstrap resampling
whole active regions (windows of one region are correlated; i.i.d. resampling
would understate uncertainty), and **frozen validation-tuned thresholds** alongside
peak TSS. Both findings survive and sharpen: the benchmark-trained gap is
**+0.078 peak (95% CI +0.006..+0.306) and +0.085 frozen (CI +0.007..+0.309)** —
positive in every tested year — while single-year live training *widens* the gap
(+0.109). One refinement emerges: training on **multiple** live years
(2011+2012+2014) narrows the gap to +0.046, though the difference's CI includes
zero — so multi-year operational training *may* partially help, but is not
statistically established. The same protocol also reproduces the optimism on the
**geomagnetic-storm model** (peak gap +0.165, CI +0.006..+0.372) — a different
feature space and physical domain — indicating the effect is not a flare-model
quirk.

---

## 9. Diagnosing the Shift — Feature Distributions

To characterize the distribution shift underlying the gap, we compared the raw
JSOC distributions of all 17 SHARP parameters between the benchmark training era
(February 2011; 17,950 records) and the operational year (February 2015; 44,175
records) using two-sample Kolmogorov–Smirnov tests (Figure 3).

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

## 10. Physics Interpretation — What the Model Learned

**Feature importance (Figure 5).** Aggregating the RandomForest's importances over
each parameter's 7 summary statistics, five parameters account for **72% of the
model's decisions**: TOTUSJH (0.20, total unsigned current helicity), TOTUSJZ
(0.16, total unsigned vertical current), R_VALUE (0.12, flux near the
polarity-inversion line), USFLUX (0.12, total unsigned flux ≈ region size), and
TOTPOT (0.11, total magnetic free energy). These are precisely the physically
motivated flare predictors: large, current-carrying active regions with strong
polarity-inversion-line fields are the flare-productive ones — consistent with the
established flare-forecasting literature.

**PCA (Figure 6).** A principal-component analysis of the 119-feature space shows
PC1 alone explains **33%** of the variance and is dominated by the extensive
current/flux parameters (TOTUSJZ, TOTUSJH, SAVNCPP, ABSNJZH, USFLUX) — i.e. an
**"active-region size and magnetic-energy" axis.** PC1–3 capture 52%, and 15 of
119 components are needed for 90%, indicating a moderately redundant but not
trivially low-dimensional feature space. (Prior competition work reached a similar
"dominant factor ≈ region size" conclusion via factor analysis; we recover it
transparently and note it is one axis of several.)

**The connecting insight.** The parameters the model relies on most (TOTUSJH,
TOTUSJZ, USFLUX, TOTPOT) are the *same* parameters that shift most between
benchmark and operational data (Section 9: MEANPOT, TOTPOT, TOTUSJH, USFLUX,
TOTUSJZ have the largest KS D). **The model leans hardest on exactly the inputs
that are least stable across regimes** — a direct, physical explanation for why its
operational skill degrades: its most-trusted magnetic predictors are the ones whose
distributions differ most between the curated benchmark and live operation.

---

## 11. Limitations

- Each period is a single 3-month window; more windows per solar-cycle phase
  would tighten the estimates.
- Low-activity periods contain few M+ flares (2023: 32 positives), widening
  confidence intervals — the 2023 result is indicative, not definitive.
- TSS depends on operating point; comparisons are reported per threshold.
- The gap now replicates across two flare-model architectures (RandomForest,
  LightGBM) and a storm model in a separate feature space; broader families
  (e.g., deep sequence models) remain untested.
- Multi-year live training narrows the gap (+0.046 vs +0.078) but the difference's
  CI includes zero — H₁ᵦ's refutation is firm for single-year retraining and
  open for larger operational training sets.
- The correction experiment (Section 8) is not yet complete.

---

## 12. Conclusion

A SHARP-based flare model reporting TSS 0.77 on the SWAN-SF benchmark achieves
live, out-of-sample TSS of only 0.35–0.64 at its default operating point across
three solar-cycle phases — the benchmark exceeds the upper 95% CI of live skill
in every period, overstating operational skill by up to ~2×, condition-dependent.
A controlled 2×2 experiment further shows that **retraining on live operational
data does not close the gap**, indicating the overstatement is a property of the
benchmark's evaluation set rather than of the training distribution — while a
threshold **recalibrated** on one operational period and frozen recovers TSS to
0.66–0.84 on unseen years. The gap **replicates on a second architecture**
(LightGBM) and on a geomagnetic-storm model in a separate feature space, and it
survives a hardened protocol (four operational years, cluster bootstrap, frozen
thresholds). Together these results establish a reproducible method for honestly
evaluating operational solar-flare forecasts and caution that benchmark
leaderboard scores must not be read as operational performance. The practical
implication is that the community should benchmark on live-operational test
sets, not only curated partitions.

---

## Figures

All figures are in `figures/` and reproducible from the experiment scripts.

- **Figure 1** (`fig1_multiperiod.png`) — Benchmark vs. live TSS across 2014/2015/2023, with 95% CIs (Exp 1, §5).
- **Figure 2** (`fig2_scorecard_2x2.png`) — The 2×2: training on live data does not close the gap (Exp 3, §8).
- **Figure 3** (`fig3_distribution_shift.png`) — KS distribution shift across all 17 SHARP features (Exp 2, §9).
- **Figure 4** (`fig4_accuracy_illusion.png`) — A zero-skill model "wins" on accuracy (§7).
- **Figure 5** (`fig5_feature_importance.png`) — RandomForest feature importance per SHARP parameter (§10).
- **Figure 6** (`fig6_pca_scree.png`) — PCA scree; PC1 = active-region size/energy axis (§10).
- **Figure 7** (`fig7_recalibration_fix.png`) — The validated fix: a threshold frozen on 2014 beats the default on unseen 2015/2023 (§8).

---

## References

1. Angryk, R. A., Martens, P. C., Aydin, B., et al. (2020). Multivariate time
   series dataset for space weather data analytics. *Scientific Data*, **7**, 227.
   doi:10.1038/s41597-020-0548-x
2. Bobra, M. G., Sun, X., Hoeksema, J. T., et al. (2014). The Helioseismic and
   Magnetic Imager (HMI) Vector Magnetic Field Pipeline: SHARPs — Space-Weather
   HMI Active Region Patches. *Solar Physics*, **289**, 3549–3578.
   doi:10.1007/s11207-014-0529-3
3. Bobra, M. G., & Couvidat, S. (2015). Solar Flare Prediction Using SDO/HMI
   Vector Magnetic Field Data with a Machine-Learning Algorithm. *The
   Astrophysical Journal*, **798**(2), 135. doi:10.1088/0004-637X/798/2/135
4. Bloomfield, D. S., Higgins, P. A., McAteer, R. T. J., & Gallagher, P. T.
   (2012). Toward Reliable Benchmarking of Solar Flare Forecasting Methods. *The
   Astrophysical Journal Letters*, **747**(2), L41.
   doi:10.1088/2041-8205/747/2/L41
5. Leka, K. D., Park, S.-H., Kusano, K., et al. (2019). A Comparison of Flare
   Forecasting Methods. II. Benchmarks, Metrics, and Performance Results for
   Operational Solar Flare Forecasting Systems. *The Astrophysical Journal
   Supplement Series*, **243**(2), 36. doi:10.3847/1538-4365/ab2e12
