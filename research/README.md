# research/ — the experiments behind PAPER.md

Every headline number and figure in [PAPER.md](../PAPER.md) is produced by a
script in this folder. Results land in `research/results/` (committed) and
figures in `figures/` (committed). Raw fetched datasets cache to
`data/sharp_live/` (gitignored — rebuilt from JSOC/HEK on demand).

| Script | Paper section | What it does |
|---|---|---|
| `exp1_multiperiod_rescore.py` | §5, Fig 1 | Scores the SWAN-SF-trained model on live JSOC 2014 / 2015 / 2017 with cluster-bootstrap 95% CIs (2023 is declared but excluded by the fail-closed label gate) |
| `../solarflare/scorecard.py` | §8, Fig 2 | The 2×2: benchmark- vs live-trained × benchmark vs operational test (`python -m solarflare.scorecard`) |
| `exp2_distribution_shift.py` | §9, Fig 3 | KS effect sizes (D) for all 17 SHARP features, benchmark era (2011) vs operational (2015) + missing-data rates |
| `exp4_recalibration.py` | §8 (fix) | Three frozen thresholds (val-F1 default, benchmark-val TSS, live-recalibrated) tested on unseen 2015/2017 with paired gain CIs — shows the fix is the TSS objective, not live data |
| `exp5_second_model.py` | §5 | Trains LightGBM under the identical protocol; the benchmark-vs-live gap replicates on a second architecture |
| `physics_interpretation.py` | §10, Figs 5–6 | RandomForest feature importance per SHARP parameter + PCA of the feature space |
| `make_figures.py` | all figures | Regenerates figures/fig1–7 from `results/*.json` |

Run from the repo root (needs `requirements.txt` + the LFS models pulled):

```bash
python research/exp1_multiperiod_rescore.py     # ~15 min (JSOC+HEK fetches)
python research/exp2_distribution_shift.py      # ~5 min
python research/physics_interpretation.py       # ~5 min
python research/exp4_recalibration.py           # ~15 min (reuses cached fetches)
python -m solarflare.scorecard                  # needs data/sharp_live/*.npz training sets
python research/make_figures.py                 # instant (reads results/*.json)
```

Determinism: fixed seeds (42) for bootstrap/PCA; JSOC/HEK fetches are historical
queries and return identical records for identical date ranges.
