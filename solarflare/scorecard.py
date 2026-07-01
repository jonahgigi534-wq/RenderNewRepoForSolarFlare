"""The project's research experiment, made reproducible.

Research question: *Do standard benchmark scores overstate the real-time
operational skill of ML flare forecasts, and does training on live satellite data
close the gap?*

A 2x2: two training sources x two test sets.

               |  BENCHMARK test (held-out SWAN-SF)  |  OPERATIONAL test (live JSOC, unseen year)
  ------------ | ----------------------------------- | ------------------------------------------
  Benchmark-   |          (a)                        |            (b)   <- overstatement = a - b
   trained     |                                     |
  Live-trained |          (c)                        |            (d)

  * Benchmarks overstate operational skill  <=>  a - b > 0
  * Live training closes the gap            <=>  (c - d) < (a - b)

SKILL METRIC: TSS at a decision threshold chosen on a held-out VALIDATION split
(never on the test set). Earlier versions used peak TSS (threshold maximised on
the test set itself), which is optimistic; selecting the threshold on validation
and evaluating on a disjoint test set is the honest, deployment-faithful measure.

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


def _val_selected_tss(payload: dict, Xval, yval, Xte, yte) -> float:
    """Honest TSS: pick the threshold that maximises TSS on the VALIDATION split,
    then evaluate at that fixed threshold on the disjoint TEST set."""
    pv = payload["model"].predict_proba(Xval)[:, 1]
    thr, _ = ev.best_threshold(yval, pv, "tss")            # chosen on validation only
    pt = payload["model"].predict_proba(Xte)[:, 1]
    pred = (pt >= float(thr)).astype(int)
    return round(float(ev.tss(yte, pred)), 3)


def _val_test_split(d: dict, cfg: dict):
    """Region-disjoint chronological val/test split of a dataset -> (Xval,yval,Xte,yte)."""
    _, ival, ite = sharptrain.time_group_split(d["groups"], d["end_times"],
                                               cfg["sharp_live"]["split"])
    Xval, yval = dataio.build_matrix(d["X3d"][ival], cfg), d["y"][ival]
    Xte, yte = dataio.build_matrix(d["X3d"][ite], cfg), d["y"][ite]
    return Xval, yval, Xte, yte


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    root = cfg["_project_root"]
    dd = os.path.join(root, "data", "sharp_live")

    # Two models, trained in-memory (deployed model untouched):
    print("Training benchmark model (SWAN-SF 2010-2012) ...", flush=True)
    v2 = sharptrain.train(os.path.join(dd, "dataset_swansf_p1.npz"), cfg, save=False)
    print("Training live model (JSOC 2014) ...", flush=True)
    v1 = sharptrain.train(os.path.join(dd, "dataset_2014.npz"), cfg, save=False)

    # BENCHMARK test = held-out region-disjoint val/test split of SWAN-SF p1.
    # OPERATIONAL test = live JSOC 2015 (an unseen year), also split into val/test so
    # the decision threshold is chosen on operational VALIDATION, never on the test.
    dp = sharpdata.load_dataset(os.path.join(dd, "dataset_swansf_p1.npz"))
    Xbv, ybv, Xbt, ybt = _val_test_split(dp, cfg)
    do = sharpdata.load_dataset(os.path.join(dd, "dataset_2015.npz"))
    Xov, yov, Xot, yot = _val_test_split(do, cfg)

    rows = []
    for name, detail, m in [("Benchmark-trained", "SWAN-SF 2010-2012", v2),
                            ("Live-trained", "JSOC live data (2014)", v1)]:
        b = _val_selected_tss(m, Xbv, ybv, Xbt, ybt)
        o = _val_selected_tss(m, Xov, yov, Xot, yot)
        rows.append({"name": name, "detail": detail, "benchmark_tss": b,
                     "operational_tss": o, "gap": round(b - o, 3)})

    bench = next(r for r in rows if r["name"] == "Benchmark-trained")
    live = next(r for r in rows if r["name"] == "Live-trained")
    out = {
        "title": "Benchmark score vs. real operational skill",
        "metric": "TSS at a validation-selected threshold (chosen on a held-out split, "
                  "not the test set)",
        "test_sets": {
            "benchmark": f"held-out SWAN-SF magnetograms ({int(len(ybt))} windows, {int(ybt.sum())} flares)",
            "operational": f"live JSOC 2015, an unseen year ({int(len(yot))} windows, {int(yot.sum())} flares)",
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
