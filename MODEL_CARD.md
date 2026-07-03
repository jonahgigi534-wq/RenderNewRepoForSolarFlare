# Helios — Model Card

Honest documentation of every trained model this project ships. Format follows
the "Model Cards for Model Reporting" practice (Mitchell et al., 2019).

---

## 1. `flare_sharp_live_model.joblib` — the DEPLOYED flare model

| | |
|---|---|
| **Task** | P(M-class-or-greater flare within 24 h) per active region, from 12 h of SHARP magnetic-field history |
| **Architecture** | sklearn Pipeline: median impute → StandardScaler → RandomForest (isotonic-calibrated) — preprocessing is saved WITH the model |
| **Training data** | SWAN-SF partition 1 (Harvard Dataverse doi:10.7910/DVN/EBCFKM), 2010–2012, restricted to the 17 SHARP keywords JSOC serves live; 73,492 windows, 1,254 positives |
| **Evaluation** | region-disjoint chronological split; held-out test TSS **0.77** at the balanced operating point |
| **Operational skill** | measured on unseen JSOC years by `solarflare.scorecard` — expect the benchmark number to overstate it (that finding is the point of this project; see RESULTS.md) |
| **Runs live on** | JSOC `hmi.sharp_cea_720s_nrt` (near-real-time, ~1 h latency) via `drms` (`/api/sharp_live`); historical `?at=` demos use the definitive `hmi.sharp_cea_720s` |

**Intended use:** research/education dashboard guidance alongside official NOAA
SWPC products — never as the sole input to an operational decision.

**Operating points:** trained with high_recall / balanced / high_precision
(validation-tuned). `python -m solarflare.recalibrate` adds an **operational**
point — the threshold recalibrated on one year of live JSOC data (the
self-correcting-deployment feature; provenance saved in the artifact). Select
via `sharp_live.operating_point` in config.yaml.

**Deployed default (since 2026-07-02): `operational`** (threshold 0.021,
calibrated on 2013). Rationale: the project's own research shows the F1-tuned
"balanced" point collapses operationally (live TSS ~0.35 at solar max) because
F1's precision term is base-rate-sensitive, while a TSS-objective threshold
transfers across solar-cycle phases (live TSS 0.62/0.82/0.77 on 2014/2015/2017).
Trade-off: recall-heavy — more warnings at lower precision.

**Variant:** `flare_sharp_live_model_multiyear.joblib` — same pipeline trained
on JSOC 2011+2012+2014 (test TSS 0.83); runs live via
`/api/sharp_live?variant=multiyear` for side-by-side comparison. The default
deployed artifact is never modified by variant training or recalibration of a
variant.

**Series separation (deliberate):** training and dataset builds use the
definitive science series (`hmi.sharp_cea_720s`); live inference reads the NRT
series (`hmi.sharp_cea_720s_nrt`) because the definitive series lags real time
by weeks (measured 2026-07-02: 35 days). Both expose the same 17 keywords
(verified); NRT values are preliminary and may be revised in the definitive
record.

**Known limitations**
- Trained on solar-cycle-24 rising phase (2010–2012); skill degrades on other
  cycle phases (quantified in `skill_scorecard.json`).
- Only 17 of the 24 classic SHARP parameters are available live; the 7
  Lorentz-force parameters are absent from the JSOC keyword series.
- Labels derive from flare catalogs that attribute flares to NOAA active
  regions; mis-attributed or region-less flares are label noise (quantified in
  `gap_diagnosis.json`).
- Probabilities are isotonic-calibrated on training-era data; operational-year
  calibration is shown in the dashboard reliability diagram.

## 2. `flare_sharp_model.joblib` — the SWAN-SF benchmark model (NOT deployable)

ExtraTrees, TSS 0.87 on the cleaned SWAN-SF benchmark. **Cannot run on live
data**: it lives in the Cleaned-SWANSF "LSBZM" normalized feature space whose
normalization parameters are not distributed with the model. Kept as the
benchmark comparison point and to power the always-on web forecast's SHARP track
when features are supplied externally. This gap between "great benchmark score"
and "cannot actually be deployed" motivated the whole research question.

## 3. `storm_kp_model.joblib` — geomagnetic-storm model

| | |
|---|---|
| **Task** | P(Kp ≥ 5 storm within 24 h) from 24 h of L1 solar-wind history |
| **Architecture** | logistic regression (isotonic-calibrated), 8 derived channels × 7 summary stats |
| **Training data** | NASA OMNI hourly (CDAWeb HAPI), chronological split with a 5-day gap |
| **Evaluation** | held-out test TSS ≈ 0.47–0.51; base rate ~9% |
| **Generalisation check** | `storm_scorecard.json` — the same benchmark-vs-operational gap appears here too |

**Deployed operating point: `high_recall`** (max-TSS on validation, threshold
~0.588) — not `balanced`/F1. The storm model shows the same benchmark→operational
gap as the flare model (`storm_scorecard.json`), so it deploys on the
TSS-objective point the project's finding endorses, matching the validation-TSS
"frozen" threshold that scorecard already reports as the deployment number.
Trade-off: recall-heavy (more storm warnings, lower precision) — the deliberate
choice when misses cost more than false alarms.

**Separate feature space** from the flare models (L1 solar wind vs solar
magnetograms) — never mixed, by design.

---

### Provenance & reproducibility

- Every experiment: fixed seeds, leakage-free splits, in-memory training
  (`save=False`) so deployed artifacts are never silently replaced.
- `python -m solarflare.reproduce` regenerates every research artifact and
  RESULTS.md.
- Models ship via Git LFS; datasets are gitignored but rebuildable from public
  sources (JSOC, HEK, Harvard Dataverse, CDAWeb, SWPC warehouse).
