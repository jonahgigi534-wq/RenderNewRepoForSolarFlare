# Handoff: verified holes + fix plan (red-team audit, 2026-07-02)

## STATUS UPDATE (session of 2026-07-02 evening, Jonathan's machine)

- **#1 CLOSED for the research/ pipeline.** The gate is now FAIL-CLOSED
  (`scorecard.label_gate_status`: unmeasured years are excluded too — silence
  cannot pass); exps 1/4/5 pass every period through it and record exclusions
  + reasons in their JSONs. The 2023 measurement was independently reproduced
  (43/289 = 0.15; 2014 control 180/191 = 0.94) and the sweep extended:
  2017 = 0.95 (42/44), 2020 = 0.00 (0/2), 2021 = 0.55 (17/31),
  2022 = 0.11 (20/186); 2018/2019 have zero M+ flares. All measured rates are
  in config `scorecard.label_attribution_by_year`.
- **Replacement period built:** `eval_2017_declining` (Aug–Oct 2017, the X9.3
  era; 995 windows, 26 positives, 3 flaring regions). Exps 1/4/5 + figures
  regenerated: H₀ rejection now rests on 2014 + 2017; exp4's live-data
  increment is bounded at +0.02.
- **DATA MACHINE still owed:** `python -m solarflare.reproduce` —
  skill_scorecard.json / dose_response.json / RESULTS.md still carry 2023.
  The fail-closed gate needs measured rates for every scored year: 2013–2017
  are in config, so the regen keeps 2013/15/16/17 and drops 2023.
- **Deployment flipped:** `sharp_live.operating_point: operational` (rationale
  in MODEL_CARD). The dashboard badge will now show the self-corrected point.
- **History grading semantics corrected** (notify._row_record): alerts grade
  HIT / FALSE_ALARM; daily rows grade CORRECT / INCORRECT against the 50%
  line. Rows re-grade automatically on the next export from the DB.
- **ACTION NEEDED — notifier machine:** recipients were scrubbed from
  config.yaml. Set `$HELIOS_RECIPIENTS` (comma-separated) there or alert/daily
  e-mails silently drop to dry-run.
- requirements.txt now includes `drms` + `scipy`.
- PAPER/ABSTRACT/BOARD were updated **as reference drafts at Jonathan's
  direction** — the team writes the competition manuscript from scratch in
  their own words (AI-use disclosure still applies).

## STATUS (session of 2026-07-02 ~18:00 CT, laptop WITHOUT datasets)

All 12 items implemented and pushed (commits 8ea1136..d39dab3 + this one);
both test suites green after every change; offline CI added and green.

- **#2 DONE & VERIFIED LIVE**: /api/sharp_live at true now() returns
  available:true from hmi.sharp_cea_720s_nrt (10 regions, data 74 min old,
  full-disk P(M+)=0.38). Historical ?at= stays on the definitive series.
- **#1 CODE DONE, REGEN PENDING**: label-attribution gate shipped
  (config scorecard.min_label_attribution + scorecard.label_excluded_years);
  measured rates 2013=0.98, 2014=0.94, 2015=0.98, 2016=0.94, 2023=0.15
  (scripts/label_attribution.py; **2017-2024 sweep unfinished** — rerun
  `python scripts/label_attribution.py 2017 2024`). **This machine has no
  data/sharp_live datasets, so artifacts still carry 2023** — next session on
  the data machine: `python -m solarflare.reproduce`, then list which
  RESULTS.md numbers moved (expect: 2023 rows drop, operational means/CIs
  recompute over 2013/15/16/17, dose_response test set = 2015-17).
- #3/#4: autoDeploy:false; prospective-record reality documented in
  render.yaml. **LFS billing still unchecked** — only the repo OWNER account
  (solarflarepredictor-cmd) can see Settings -> Billing; a collaborator can't.
- #5/#6/#7: one-model LRU in sharp_live._MODELS; DEMO_ALERT_TOKEN header gate
  (+ email refuses to send with live SMTP and no token); caches LRU-capped 32,
  ?at= rounded to the hour, per-key JSOC fetch lock.
- #8/#9/#12: dashboard scorecard headlines FROZEN TSS (peak demoted to the
  ceiling line); both panels state their differing leakage guards in-caption;
  date-picker ceiling derives from the clock.
- #10/#11: README "Rebuilding the research datasets" section with expected
  counts; .github/workflows/ci.yml runs scripts/offline_test.py.
- **HUMANS**: the paper's 2023 claims rest on the same AR-attributed labels —
  re-check them (no AI edits to PAPER/ABSTRACT/BOARD, per the rules).

For the next Claude Code session. Every finding below was **verified with
evidence** (measured, not guessed) during a hostile audit on 2026-07-02.
Work through them in order. Numbers/commands assume this repo root; the venv
python is `.venv/Scripts/python.exe` (Windows) — on other machines use the
project venv equivalent.

## Ground rules (do not skip)

- **Never write prose for PAPER.md / ABSTRACT.md / BOARD.md** — competition
  rule: no AI on the manuscript. Product code, experiments, data, JSON
  artifacts, and RESULTS.md (auto-generated) are all fine.
- **No Claude co-author trailer in commits** (attribution is disabled in
  settings; keep it that way).
- A second Claude Code (Render deploy machine) also pushes to `main`.
  `git pull --rebase` BEFORE starting work. If you make a local **merge**
  commit, push it directly — do NOT `git pull --rebase` afterwards (it
  flattens the merge and conflicts; this bit us once already).
- Definition of done for any change: `scripts/regression_test.py` and
  `scripts/smoke_test.py` green, artifacts regenerated when the change affects
  them (`python -m solarflare.reproduce`), commit + push.
- Heavy work (JSOC pulls, retraining, scorecard reruns) in background shells.

## P0 — published-results correctness

### 1. dataset_2023.npz labels are ~85% false-negative — 2023 numbers are wrong
**Evidence (measured):** HEK/SWPC M+ flares with a NOAA AR number: 2014 =
180/191 (**94%**); 2023 = 43/289 (**15%**). Our labeler
(`sharpdata.fetch_flares` → `build_windows`) can only mark a positive when the
flare has an AR number, so ~85% of real 2023 flares are labeled "no flare".
The 0.5% base rate in dataset_2023.npz (vs 2.4% in 2014) is catalog decay,
not the Sun. Every 2023 cell currently in `skill_scorecard.json`,
`dose_response.json`, and the recalibrate transfer checks is biased DOWN by
label noise. (`noaa_baseline.json` 2023 is FINE — its ground truth counts
flare-days without requiring AR numbers.)

**Fix (staged):**
1. Immediately: exclude 2023 from label-dependent evaluations — add an
   attribution-quality gate (e.g. config `scorecard.min_label_attribution: 0.8`
   with a per-year attribution measurement, or a hardcoded exclude list with
   the measured rates in a comment). Regenerate artifacts + RESULTS.md.
   Surface the exclusion honestly in the JSON (`excluded_years` + reason).
2. Then: sweep attribution rates for 2015–2023 (28-day HEK chunks, count
   `ar_noaanum` present vs total for frm_name=SWPC, M+) to find WHERE the
   catalog decays — decides which years are trustworthy.
3. Optional (bigger): rebuild 2023 labels without AR dependence — match flare
   times to SHARP patches by heliographic position (needs LAT_FWT/LON_FWT
   keywords added to the pull) or use an alternative attributed catalog.
4. Tell the humans: the paper's 2023 claims rest on similarly-built labels —
   they need to re-check (do NOT edit the paper yourself).

### 2. Live inference queries the wrong JSOC series — "live" has never worked at now()
**Evidence (measured 2026-07-02):** definitive `hmi.sharp_cea_720s` latest
T_REC = **2026.05.28** (35 days stale); `hmi.sharp_cea_720s_nrt` latest T_REC
= **2026.07.02_20:48** (~1 h old). `sharp_live.fetch_recent_windows` pulls the
definitive series for the last ~17 h → always empty at true now(), on any
machine. The dashboard's "no current JSOC data in this environment" message is
this bug, not a sandbox quirk.

**Fix:** config `sharp_live.live_series: hmi.sharp_cea_720s_nrt` used by
live inference only (training/datasets stay on the definitive series —
that separation is scientifically correct and worth stating in MODEL_CARD).
`sharpdata.fetch_sharp` takes the series from config today via the SERIES
constant — thread a series argument through instead. Verify first that all 17
`sharp_live.keywords` exist in the NRT series (`drms.Client().keys(...)`).
Show data age in the dashboard card. Acceptance: `GET /api/sharp_live` (no
`?at=`) returns `available: true` with `as_of` within a few hours, today.
The Sun is active (X/M flares this week) — this makes the demo genuinely live.

## P1 — operations

### 3. Git-LFS bandwidth quota is likely already blown
177 MB of LFS models fetched per Render auto-deploy × ~11 pushes/day.
Free quota = 1 GB/**month**. Check GitHub → Settings → Billing → LFS.
**Fix:** set `autoDeploy: false` in render.yaml (deploy manually per
milestone); consider `GIT_LFS_SKIP_SMUDGE=1` build + fetching only the models
the server actually serves.

### 4. Prospective record dies on Render
render.yaml has no persistent disk and free tier sleeps: notify.db +
prediction_history.csv reset on every deploy, and the notifier loop isn't
running while asleep. **Fix:** document that the canonical prospective record
lives on the local always-on machine (current reality), and/or add a Render
disk (paid) with `notify.db_path`/`history_csv` pointed at it.

### 5. Variant dropdown can OOM the free dyno
Three loadable models (22+78+77 MB on disk, 2–3× in RAM). render.yaml's own
comment warned about two. **Fix:** LRU of one live variant at a time in
`sharp_live._MODELS` (evict on switch), or hide variants on Render via env.

### 6. `POST /api/alerts/demo` is an unauthenticated email trigger
With SMTP_PASSWORD set on a public deploy, anyone can spam the three
hardcoded recipients. **Fix:** require a token (env `DEMO_ALERT_TOKEN`,
checked in the endpoint; dashboard sends it from a prompt) or restrict the
public path to log+webhook only.

### 7. JSOC stampede + unbounded caches in the request path
`sharp_live._CACHE`/`_WINDOWS_CACHE` key on raw `?at=` (unlimited keys, no
eviction); concurrent identical requests all fetch JSOC (minutes) with no
lock. **Fix:** round `at` to the hour, LRU-cap both caches (~32), and guard
the fetch with a per-key lock.

## P2 — consistency / presentation / hygiene

### 8. Two panels, two protocols
The 2×2 table (v3) WITHHOLDS the multi-year model's benchmark cell as
unfixable leakage; the dose-response chart DRAWS benchmark points for
2011/2012-containing steps using region-exclusion instead. A sharp judge will
ask. Team decision needed: withhold those dose points too, or state the
difference in both panels' captions. (Context: dose-response benchmark cells
are methodologically the same as the 2×2's cell (c) — backwards-in-time but
region-disjoint.)

### 9. Peak TSS is still the table's headline column
The project's own finding is that peak is the generous number; frozen
(deployment) TSS sits in a caption. Consider swapping emphasis on the
dashboard scorecard table.

### 10. Fresh-clone reproducibility gap
Datasets are gitignored; `reproduce` needs a 1.2 GB manual Dataverse tar +
~2.5 h of JSOC year builds first. Add a README "data setup" section with the
exact build commands + expected dataset shapes/checksums, so the
"one command" claim is honest.

### 11. No CI
Add a GitHub Actions workflow running an offline subset: import checks, the
pure-math regression sections (evaluate/magnitude/history-merge), and JSON
artifact contract checks — skip network sections via an env flag. Gates the
auto-deploying main branch.

### 12. Small stale bits
Date-picker max hardcoded `2026-05-25` (frontend/index.html) — derive from
config or bump; with fix #2, true "now" works and the note text changes.

## After P0 fixes
Regenerate everything (`python -m solarflare.reproduce`), verify both test
suites, check the dashboard renders (headless: node-extract functions from
frontend/index.html and run against skill_scorecard.json — see repo history
for the pattern), push, and list for the team which RESULTS.md numbers moved
(the humans update the paper themselves).
