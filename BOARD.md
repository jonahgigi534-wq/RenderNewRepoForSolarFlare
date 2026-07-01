# ISEF Tri-Fold Board Layout

A judge spends ~3 minutes and reads the **center panel first**. Put the strongest
result there. Everything below is sized for a poster: short, bold, numbers-forward.

---

## Header (spans all three panels, top)

**Title:** Benchmark or Reality? Quantifying the Operational Skill Gap in
Machine-Learning Solar-Flare Forecasting
**Subtitle / one-line takeaway (large, under the title):**
> *Benchmark scores overstate real solar-flare forecasting skill by up to 2× — and retraining can't fix it.*

Author name(s), school, category (Physics & Astronomy).

---

## LEFT PANEL — "Why & How"

**1. Problem / Goal** *(3–4 lines)*
Solar flares threaten power grids, satellites, and aviation. ML flare forecasts
routinely report 90–99% accuracy — but does that reflect *real-time* skill on live
data? We test whether benchmark scores survive contact with operational reality.

**2. Research Question** *(boxed, bold)*
> Do standard benchmark scores overstate the real-time operational skill of ML
> flare forecasts, and does training on live satellite data close the gap?

**3. Hypotheses**
- H₁ₐ: benchmarks overstate live skill.
- H₁ᵦ: training on live data closes the gap.

**4. Methods** *(pipeline diagram or bullet list)*
- Model: RandomForest on 17 SHARP magnetic parameters (12 h → 24 h M+ forecast).
- Benchmark test: held-out SWAN-SF. Operational test: live JSOC/SDO + NOAA flares,
  scored the same way the deployed system runs.
- Metric: TSS (True Skill Statistic) — not accuracy (see center).

---

## CENTER PANEL — "The Findings" (the money panel)

**FIG 1 — the gap** *(largest figure on the board)*
Caption: *Benchmark TSS 0.77 vs. live 0.35–0.64 across three periods (95% CIs).
The benchmark beats the upper CI in every period.*

**FIG 2 — the 2×2** *(second-largest)*
Caption: *Training on live data does NOT close the gap → the overstatement is in
the benchmark's test set, not the training data.*

**FIG 7 — the fix works** *(the closer)*
Caption: *A threshold recalibrated on 2014 and frozen beats the benchmark-tuned
default on unseen 2015 (+0.19) and 2023 (+0.33) — CIs don't overlap. Recalibration
recovers the skill that retraining can't.*

**FIG 4 — the accuracy illusion**
Caption: *A zero-skill "always no-flare" model scores 96% accuracy (99% at a 1%
base rate) with zero skill — and beats a real model on accuracy. Accuracy lies.*

**Headline results box** *(bold, boxed)*
- Benchmark overstates operational skill by **up to 2×** (H₀ rejected, all periods).
- Retraining on live data **does not** close the gap (H₁ᵦ **refuted**).
- **Recalibrating on live data DOES** — validated on unseen years (+0.19/+0.33 TSS).
- The gap is **condition-dependent** (largest at solar max).

---

## RIGHT PANEL — "Why It Happens & So What"

**5. Diagnosis — distribution shift** *(FIG 3)*
Caption: *All 17 magnetic features shift between benchmark and operation (median
KS D 0.19); live data is also ~2% missing — mess the benchmark scrubs away.*

**6. Physics — what the model learned** *(FIG 5, optionally FIG 6)*
Caption: *5 parameters (current helicity, vertical current, PIL flux, total flux,
free energy) drive 72% of decisions.* **Key insight:** the model relies most on
the features that shift most — it trusts its least-stable inputs.

**7. Conclusion** *(boxed)*
> Benchmark leaderboard scores must not be read as operational performance. The gap
> lives in the *evaluation set*, so the fix is to benchmark on live-operational data
> — not to collect more training data.

**8. Broader impact / future work** *(2–3 lines)*
A reproducible recipe for honest space-weather forecast evaluation. Next: recalibrate
on live data, and extend to CMEs and geomagnetic storms.

---

## Figure usage
- **Headliners (must be on the board):** Fig 1, Fig 2, Fig 7.
- **Support:** Fig 3, Fig 4, Fig 5. (Fig 6 / PCA only if space allows.)
- Print figures large enough to read titles from ~3 ft.

## Design tips
- One accent color for "benchmark," one for "live" — keep it consistent with the figures (blue = benchmark, orange = live).
- Bold every number. Judges scan digits.
- White space > text. Cut any sentence that isn't a claim or a number.
- Have the 245-word abstract (ABSTRACT.md) printed and available, not necessarily on the board.
