# Project Timeline

A factual, dated record of the project's development, generated from the Git
commit history (`git log`) and model training metadata (`models/*.meta.json`).
Use this as the backbone for Form 1A's experimentation dates and as a data-book
appendix — every entry below is independently verifiable in the repository.

**Experimentation span to date:** 2026-06-27 → 2026-07-01 (ongoing).
*(Fill in start/end dates on Form 1A once your team's documented work
concludes — this file will keep growing as long as the project is active.)*

---

## 2026-06-27 — First models trained
- `flare_sharp_model` (SWAN-SF benchmark model) trained — 00:59 UTC.
- `storm_kp_model` (NASA OMNI geomagnetic-storm model) trained — 08:23 UTC.

## 2026-06-29 — Repository established
- Initial commit: Helios solar-flare & space-weather forecaster (baseline
  predictor, flux/NOAA tracks, dashboard).
- Local machine-file tracking configured.

## 2026-06-30 — Live SHARP model, deployment groundwork, first retrain
- Added the live SHARP flare model: a self-contained JSOC-trained pipeline
  (fetches live SDO/HMI magnetic data, runs independently of the SWAN-SF
  benchmark model).
- Shipped all three trained models via Git LFS so the deployment could carry
  them.
- Added Render deployment configuration (single-service web app).
- Retrained the live SHARP model on the SWAN-SF benchmark span, raising its
  test TSS from 0.54 to 0.77.
- Frontend refinements: US Central time display (DST-aware), calendar
  date-picker constrained to the real data-availability window
  (May 2010 – present).

## 2026-07-01 — Research phase: the benchmark-vs-reality investigation
This is the day the project's central research question was formulated and
tested end to end.

- **Research question locked and written up:** *Do standard benchmark scores
  overstate the real-time operational skill of ML flare forecasts, and does
  training on live satellite data close the gap?* Research plan and working
  paper drafted.
- **Experiment 1 (multi-period live re-score):** the benchmark-trained model
  scored on three independent, out-of-sample live-JSOC periods (2014, 2015,
  2023) with bootstrap 95% confidence intervals.
- **Experiment 2 (distribution-shift diagnosis):** Kolmogorov–Smirnov tests
  across all 17 SHARP magnetic parameters, benchmark era vs. operational year.
- **Experiment 3 (2×2 scorecard):** built and shipped `solarflare/scorecard.py`
  and the "Model Skill Scorecard" — training source × test set, testing
  whether live-data training closes the benchmark/operational gap.
- **Physics interpretation:** RandomForest feature-importance analysis and PCA
  of the SHARP feature space.
- **Scorecard correction:** identified and fixed an optimism bug (threshold
  selection on the test set) — replaced with validation-selected thresholds,
  the honest deployment-faithful measure.
- **Prediction-history log merged and extended:** locally issued forecasts
  (ids 7–9) appended; full history merged and sorted chronologically —
  includes a verified **X1.1-class flare HIT**.
- **Experiment 4 (validated recalibration fix):** a decision threshold
  calibrated on live 2014 data only, frozen, and tested on two years it never
  saw (2015, 2023) — confirming recalibration (not retraining) recovers most
  of the lost operational skill.
- **Statistical hardening (team):** the scorecard was rebuilt for robustness —
  four operational years (2013/2015/2016/2017), cluster-bootstrap confidence
  intervals (resampling whole active regions), frozen-threshold deployment
  scores, a reliability/calibration diagram, and a mechanism diagnosis
  (label-protocol audit, feature-importance divergence between models).
- **NOAA baseline comparison (team):** the deployed system benchmarked against
  NOAA SWPC's own archived official forecasts on identical days — competitive
  with, and in two of four tested years better than, the official standard.
- **Experiment 5 (architecture replication):** the original SWAN-SF benchmark
  data was downloaded from the Harvard Dataverse archive and parsed from
  scratch; a second, independent model architecture (LightGBM) was trained
  under the identical protocol and showed the same benchmark-vs-live gap —
  establishing that the finding is not specific to one model family.
- **References verified:** all cited literature checked against publisher
  records (DOI, volume, page).
- **Author attribution finalized:** Jonathan Gigi, Alfred Antony, Aidan George
  — Cypress Woods High School.
- **Full integration pass:** the NOAA comparison, label audit, and
  feature-importance divergence results folded into the written paper; all
  figures and tables reconciled to the final, hardened statistics.

---

## How to keep this file current
Regenerate the commit list at any time with:
```bash
git log --reverse --pretty=format:"%ad|%an|%s" --date=format:"%Y-%m-%d"
```
Add a new dated section whenever the team completes a work session, in your own
words, describing what was done and why — this file is a starting skeleton, not
a substitute for your personal data book/logbook entries.
