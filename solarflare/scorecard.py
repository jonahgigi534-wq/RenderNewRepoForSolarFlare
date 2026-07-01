"""The project's research experiment, made reproducible.

Research question: *Do standard benchmark scores overstate the real-time
operational skill of ML flare forecasts, and does training on live satellite data
close the gap?*

A 2x2: two training sources x two test sets, scored with peak TSS.

               |  BENCHMARK test (held-out SWAN-SF)  |  OPERATIONAL test (live JSOC, unseen year)
  ------------ | ----------------------------------- | ------------------------------------------
  Benchmark-   |          (a)                        |            (b)   <- overstatement = a - b
   trained     |                                     |
  Live-trained |          (c)                        |            (d)

  * Benchmarks overstate operational skill  <=>  a - b > 0
  * Live training closes the gap            <=>  (c - d) < (a - b)

Both models are trained in-memory (save=False) so the deployed model is untouched.
Writes skill_scorecard.json for the dashboard "Model Skill Scorecard" panel.

Run:  python -m solarflare.scorecard
"""
from __future__ import annotations

import json
import os

from . import data as dataio
from . import evaluate as ev
from . import sharpdata
from . import sharptrain
from .config import load_config


def _peak_tss(payload: dict, X, y) -> float:
    """Best-achievable TSS over all thresholds (Youden's J) — a threshold-free,
    fair skill measure for comparing models across different test sets."""
    p = payload["model"].predict_proba(X)[:, 1]
    _, tss = ev.best_threshold(y, p, "tss")
    return round(float(tss), 3)


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    root = cfg["_project_root"]
    dd = os.path.join(root, "data", "sharp_live")

    # Two models, trained in-memory (deployed model untouched):
    print("Training benchmark model (SWAN-SF 2010-2012) ...", flush=True)
    v2 = sharptrain.train(os.path.join(dd, "dataset_swansf_p1.npz"), cfg, save=False)
    print("Training live model (JSOC 2014) ...", flush=True)
    v1 = sharptrain.train(os.path.join(dd, "dataset_2014.npz"), cfg, save=False)

    # BENCHMARK test = held-out region-disjoint split of SWAN-SF p1 (neither model
    # trained on those exact windows). OPERATIONAL test = live JSOC 2015 (unseen year).
    dp = sharpdata.load_dataset(os.path.join(dd, "dataset_swansf_p1.npz"))
    _, _, ite = sharptrain.time_group_split(dp["groups"], dp["end_times"],
                                            cfg["sharp_live"]["split"])
    Xb, yb = dataio.build_matrix(dp["X3d"][ite], cfg), dp["y"][ite]
    do = sharpdata.load_dataset(os.path.join(dd, "dataset_2015.npz"))
    Xo, yo = dataio.build_matrix(do["X3d"], cfg), do["y"]

    rows = []
    for name, detail, m in [("Benchmark-trained", "SWAN-SF 2010-2012", v2),
                            ("Live-trained", "JSOC live data (2014)", v1)]:
        b, o = _peak_tss(m, Xb, yb), _peak_tss(m, Xo, yo)
        rows.append({"name": name, "detail": detail, "benchmark_tss": b,
                     "operational_tss": o, "gap": round(b - o, 3)})

    bench = next(r for r in rows if r["name"] == "Benchmark-trained")
    live = next(r for r in rows if r["name"] == "Live-trained")
    out = {
        "title": "Benchmark score vs. real operational skill",
        "metric": "peak TSS (True Skill Statistic; higher is better, 0 = no skill)",
        "test_sets": {
            "benchmark": f"held-out SWAN-SF magnetograms ({int(len(yb))} windows, {int(yb.sum())} flares)",
            "operational": f"live JSOC 2015, an unseen year ({int(len(yo))} windows, {int(yo.sum())} flares)",
        },
        "models": rows,
        "findings": {
            "benchmarks_overstate": bool(bench["gap"] > 0.02),
            "overstatement_gap": bench["gap"],
            "live_training_closes_gap": bool(live["gap"] < bench["gap"]),
        },
    }
    path = os.path.join(root, "skill_scorecard.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return out


def main():
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
