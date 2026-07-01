# Helios — AI Solar Flare Predictor

A live, self-training solar-flare forecasting system. It learns from the
**SWAN-SF** magnetic-field dataset, classifies what the Sun is doing **right
now** from live NOAA data, and forecasts flare probability at **12 / 24 / 48 h**
lead times — with an escalating warning for X-class events. Ships with a
polished web **forecast page**.

## The research question

> **Do standard benchmark scores overstate the real-time operational skill of
> ML flare forecasts — and does training on live satellite data close the gap?**

Helios answers it with a reproducible 2×2 experiment (benchmark-trained vs
live-trained models × benchmark vs unseen-operational-year test sets), bootstrap
confidence intervals, frozen-threshold deployment scores, a storm-model
generalisation check, a mechanism diagnosis (label noise / distribution shift /
model divergence), and a comparison against NOAA's own archived official
forecasts. **One command reproduces everything:**

```bash
python -m solarflare.reproduce     # -> RESULTS.md + all JSON artifacts
```

See **RESULTS.md** (auto-generated findings) and **MODEL_CARD.md** (honest model
documentation). The dashboard's "Model skill scorecard" panel shows the result
live with CIs, per-year chips, a reliability diagram, and the NOAA comparison.

> **Model status:** a model bake-off (HistGB, RandomForest, ExtraTrees and
> Logistic, plus LightGBM & XGBoost when those optional libraries are installed)
> picks the best on a held-out validation
> partition. On the real SWAN-SF M-class task **ExtraTrees wins**, scoring
> **TSS 0.870** (recall 0.93) on the clean test partition at the *balanced*
> operating point. Three selectable operating points trade recall for
> precision — see [Operating points](#operating-points-recall-vs-precision).

---

## The one thing you must understand first

**The training data and the live data are different physics**, and the system
is built to respect that instead of faking a connection:

| | Trains on | Predicts from (live) |
|---|---|---|
| **SHARP ML model** | SDO/HMI **magnetic** parameters (24 SHARP features) | needs live SHARP (JSOC) — optional |
| **Flux nowcast/forecast** | — (statistical) | NOAA **GOES X-ray flux** |
| **NOAA region forecast** | — (NOAA's own model) | NOAA region probabilities |

A model trained on magnetic SHARP parameters **cannot** literally consume X-ray
flux — they are different feature spaces. Pretending it could would be exactly
the "cheating" the project brief warns against. So the predictor runs **three
independent tracks** and ensembles whatever is available, always returning a
clean forecast even if a source is down.

```
                       ┌─────────────────────────────┐
  NOAA GOES X-ray ───► │ NOWCAST  (exact, no ML)      │ ─► current class + X-WARNING
   (always live)       └─────────────────────────────┘
                       ┌─────────────────────────────┐
                  ───► │ FLUX forecast (persistence) │ ─┐
                       └─────────────────────────────┘  │
  NOAA regions    ───► │ NOAA official region forecast│ ─┼─► ENSEMBLE ─► 12/24/48h
                       ┌─────────────────────────────┐  │
  SWAN-SF model   ───► │ SHARP ML  (when fed magnetic)│ ─┘
                       └─────────────────────────────┘
```

Because a flare's class is *defined* by its GOES 1–8 Å peak flux, identifying a
flare **happening now** needs no ML at all — it's a measurement. The machine
learning is reserved for **forecasting the future**, which is the honest place
for it.

---

## Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. prove the whole pipeline works (NO download, NO GPU needed)
python scripts/smoke_test.py
#    -> trains on a synthetic fixture in a TEMP dir (your real model in
#       models/ is untouched), evaluates skill scores, then hits live NOAA.

# 3. run the API + open the forecast page
python -m uvicorn api.server:app --port 8000
#    -> open http://127.0.0.1:8000/
```

### Train on the real SWAN-SF data

```bash
python -m solarflare.download      # pulls the .pkl partitions (Google Drive)
python -m solarflare.train         # trains on partitions 1-3, tests on 5
```

Training writes three artifacts to `models/`:

| File | Use |
|---|---|
| `flare_sharp_model.joblib` | **load this** (recommended) |
| `flare_sharp_model.pkl` | same payload, plain pickle (you asked for `.pkl`) |
| `flare_sharp_model.meta.json` | human-readable metrics + provenance |

---

## How it avoids "cheating"

* **Leakage-free split** — trains on the augmented partitions, tunes the
  decision threshold on a *separate* validation partition, and reports final
  numbers on a **clean test partition the model never saw**. This is the
  standard SWAN-SF protocol.
* **Honest metrics** — accuracy is useless when 95% of samples are "no-flare",
  so the headline is **TSS** (True Skill Statistic = recall − false-alarm rate;
  0 = no skill) and **HSS** (Heidke Skill Score). See `solarflare/evaluate.py`.
* **Class imbalance** handled with balanced sample weights, not by resampling
  the test set.
* **Calibrated probabilities** — isotonic calibration so a "21%" forecast means
  roughly 21%.

---

## Operating points (recall vs. precision)

Flares are ~1% of samples, so you can't have high recall *and* high precision —
you pick a trade-off. Training fits all three on validation and reports each on
the clean test partition (ExtraTrees winner):

| Point | Threshold | TSS | Recall | Precision | Use when |
|---|---|---|---|---|---|
| `high_recall` | 0.066 | 0.745 | **0.99** | 0.05 | never miss a flare |
| `balanced` *(default)* | 0.329 | **0.870** | 0.93 | 0.18 | best overall skill |
| `high_precision` | 0.754 | 0.678 | 0.70 | **0.34** | fewest false alarms |

**What's a "good" precision here?** Above the 1.3% base rate = skill. Research
optimises TSS (precision 5–15%); operational alerting wants ~30–50%. On this
task the realistic ceiling while staying useful is **~34%** (`high_precision`),
still catching ~70% of flares. Select the point in `config.yaml`
(`live.operating_point`). The UI also shows a probability **band**
(Low/Moderate/High/Severe) to sidestep the hard threshold entirely.

---

## Flare taxonomy (matches the brief)

Driven entirely by `config.yaml`:

| Category | GOES 1–8 Å peak flux | Treated as |
|---|---|---|
| A, B, **weak C** (< C5) | < 5×10⁻⁶ W/m² | **no-flare** |
| **C** (≥ C5) | 5×10⁻⁶ – 10⁻⁵ | tracked |
| **M** | 10⁻⁵ – 10⁻⁴ | tracked |
| **X** | ≥ 10⁻⁴ | tracked + **WARNING** |
| X ≥ X5 / X10 | ≥ 5×10⁻⁴ / 10⁻³ | **SEVERE / EXTREME** |

The default trainable target is **"any M-or-greater flare in the next 24 h"**
(the Cleaned-SWANSF labels are binary). To get a 4-way *no-flare/C/M/X* model,
supply the **original** SWAN-SF multi-class labels and set
`training.task: multiclass` in `config.yaml` — the same pipeline handles it.

---

## API

| Endpoint | Returns |
|---|---|
| `GET /` | the forecast web page |
| `GET /health` | liveness + whether the SHARP model is loaded |
| `GET /api/forecast` | full prediction: nowcast + 12/24/48 h + ensemble + tracks |
| `GET /api/flux` | recent GOES X-ray series (for the chart) |
| `GET /api/regions` | current active regions + NOAA per-region probabilities |
| `GET /api/hazard` | dayside HF-radio-blackout footprint (NOAA D-RAP) + subsolar point |
| `GET /api/geomag` | high-latitude auroral oval (NOAA OVATION) + Kp/G-scale + aurora visibility |
| `GET /api/satellites` | satellites at risk by altitude band (CelesTrak TLE; `?scope=default\|all`) |
| `GET /api/storm` | geomagnetic-storm forecast — P(Kp≥5, 24 h) from the L1 solar wind (OMNI-trained ML) |
| `GET /api/sharp_live` | live SHARP ML flare forecast — P(M+ in 24 h) per active region from JSOC magnetic data (our own JSOC-trained model; `?at=ISO` for a historical demo) |
| `GET /api/scorecard` | the research result: benchmark vs operational TSS with bootstrap CIs, frozen thresholds, per-year scores, reliability data (+ storm check & NOAA baseline when built) |
| `GET /api/diagnosis` | WHY the gap exists — label audit, distribution shift, permutation importance |
| `GET /api/impact` | plain-language R/S/G space-weather impact statements + historical cost anchors |
| `GET /api/alerts` | active threshold alerts (log/webhook) |
| `POST /api/alerts/demo` | fire one clearly-labelled demo alert through the real channels |
| `GET /api/notify/status` | email-notifier state (dry-run/live, thresholds, pending + recent predictions) |
| `GET /api/notify/history.csv` | prediction-history spreadsheet (forecast vs verified outcome) |
| `GET /api/experiment/leadtime[.png]` | lead-time-vs-skill experiment results + figure |

---

## The space-weather platform (Parts 1 & 2)

Beyond the flare forecast, Helios layers on a geographic hazard map, an orbital
view, a real geomagnetic-storm model, and a reproducible skill experiment — each in
its **own feature space** (the cardinal rule: never mix the physics).

| Layer | Driven by | Honest status |
|---|---|---|
| **Dayside HF blackout** (2D map + 3D globe) | solar **X-rays** (NOAA D-RAP) | visualization of a NOAA product |
| **Satellites at risk** (3D globe) | flare level × orbit altitude | a flare-RISK *indicator*, not a per-satellite prediction |
| **Auroral oval / geomagnetic** (high-lat) | **Earth's field** + solar wind (NOAA OVATION + Kp) | visualization of NOAA products |
| **Storm forecaster** | **L1 solar wind** (NASA OMNI) | real ML — P(Kp≥5 in 24 h), TSS ≈ 0.5 |
| **Lead-time experiment** | all of the above + DONKI CMEs | reproducible skill-vs-lead study |

**Why separate hazard layers?** Solar X-rays drive flares and the *dayside* HF
blackout; the Sun's *magnetic* field (SHARP) drives flare prediction; *Earth's*
field + the L1 solar wind drive *geomagnetic storms* at *high latitudes*. Different
feature spaces — never mixed. The map shows each as its own toggleable layer.

### Geomagnetic-storm model (the scientific core)

`solarflare/storm.py` (+ `stormdata.py`) trains on **NASA OMNI** hourly data
(CDAWeb HAPI): a 24 h window of L1 solar-wind drivers (Bz, speed, density, dynamic
pressure, IMF clock angle, Newell coupling) → **P(a G1+ storm, Kp ≥ 5, in the next
24 h)**. Same honest protocol as the flare model — a **leakage-free chronological
split with a multi-day gap** (never shuffled), isotonic-calibrated probabilities,
three operating points, TSS/HSS via `evaluate.py`. Live inference rebuilds the
features from the real-time L1 feeds, with NOAA's Kp forecast as corroboration and a
climatology fallback.

```bash
python -m solarflare.storm               # train on real OMNI (HAPI)
python -m solarflare.storm --synthetic   # offline fixture (no download)
```

### Lead-time vs skill experiment

`solarflare/experiments/leadtime_skill.py` forecasts the *same* storm target from
several vantage points and plots **skill (TSS) vs forecast lead time**: climatology
(floor), persistence, an L1 logistic, 27-day recurrence, and a CME track (NASA
DONKI). Leakage-free; deterministic with a fixed seed:

```bash
python -m solarflare.experiments.leadtime_skill
```

Outputs land in `solarflare/experiments/results/` (CSV, JSON, a publication PNG, and
`RESULTS.md`) and the figure shows in the dashboard. Headline finding: no single
vantage point forecasts storms well at every lead — L1 wins at short lead, CMEs
extend to 1–3 days, and only a weak 27-day recurrence reaches further out.

### Impacts, aurora & alerts

* **Impact statements** (`/api/impact`) map NOAA's R/S/G scales to plain language.
* **Aurora visibility** ("how far south") is derived from Kp on `/api/geomag`.
* **Threshold alerts** (`/api/alerts`, `solarflare/alerts.py`) always log, POST to a
  webhook if `alerts.webhook_url` or `$ALERT_WEBHOOK_URL` is set, and keep email off
  by default. **No secrets are committed.**

---

## Resilience / failsafes

Every live fetch (`solarflare/sources.py`):

1. tries multiple URLs in order (primary → secondary → shorter feed),
2. has a hard timeout,
3. **caches** each success to `.cache/` (the failsafe backup),
4. falls back to the most recent cache if all sources fail, clearly flagged
   "cached / N min old",
5. if *everything* is down, the predictor returns a **climatological fallback**
   forecast plus a notice — it never errors out.

Live sources used: NOAA SWPC GOES X-ray (primary + secondary + 6-hour),
NOAA Solar Region Summary, NOAA GOES flare events, and NASA DONKI (optional
key). Add your free key at <https://api.nasa.gov> in `config.yaml`.

---

## Make it better (it's a *first* model on purpose)

* Train on the real SWAN-SF partitions (`download` → `train`).
* Add multi-class labels for true upper-C / M / X separation.
* Wire **live SHARP** from JSOC/SDO so the ML track runs in production
  (`predictor.sharp_forecast` already accepts a feature vector).
* Swap `HistGradientBoosting` for LightGBM/XGBoost or an LSTM in
  `solarflare/train.py` — the data + eval harness stay the same.
* Tune the flux-persistence priors in `solarflare/fluxmodel.py` against the
  GOES flare catalogue.

---

## Honest limitations

* Solar-flare forecasting has a **hard skill ceiling**; no model gets near
  "certainty". Treat outputs as probabilistic guidance alongside official
  [NOAA SWPC](https://www.swpc.noaa.gov/) products.
* The always-on web forecast runs the **flux + NOAA** tracks; the **SHARP ML**
  track activates only when magnetic features are supplied (the synthetic smoke
  model proves the plumbing).
* `DEMO_KEY` for NASA DONKI is heavily rate-limited — add your own key.

---

## File map

```
config.yaml                 all thresholds, paths, sources, training, experiment & alert options
solarflare/
  config.py    labels.py    config loader · flare taxonomy + X-warnings
  data.py      train.py     SWAN-SF loader + features · flare trainer (.pkl/.joblib)
  evaluate.py  download.py  TSS/HSS skill scores · dataset downloader
  sources.py   nowcast.py   resilient live fetchers · current-state from flux
  fluxmodel.py predictor.py persistence forecaster · orchestrator + fallbacks
  hazard.py    geomag.py    dayside HF blackout (D-RAP) · auroral oval + Kp (OVATION)
  satellites.py             satellites-at-risk by altitude band (CelesTrak TLE)
  storm.py     stormdata.py geomagnetic-storm forecaster · OMNI loader + features
  impact.py    alerts.py    R/S/G plain-language impacts · threshold alerts (+ demo)
  sharpdata.py sharptrain.py JSOC+HEK dataset builder · live-model trainer
  swansf_data.py sharp_live.py SWAN-SF tar parser · live JSOC inference
  scorecard.py storm_scorecard.py  the 2x2 research experiment (+CIs) · storm check
  reproduce.py             one-command reproduction of every research artifact
  experiments/leadtime_skill.py    lead-time vs skill experiment (CSV/PNG/RESULTS.md)
  experiments/gap_diagnosis.py     why the gap exists (labels/shift/importance)
  experiments/noaa_baseline.py     deployed model vs NOAA's archived official forecast
api/server.py                FastAPI backend + serves the frontend
frontend/index.html          the forecast page (dark ops dashboard + map/globe)
scripts/smoke_test.py        end-to-end proof with no download
scripts/regression_test.py   read-only guard run after every change
```
