"""Self-correcting deployment: recalibrate a deployed model's operating
threshold on REAL operational data and save it into the artifact as a new
operating point — the research finding turned into a product feature.

Why: the scorecard shows a model's validation-tuned threshold transfers badly
to operational data (threshold drift is a large share of the benchmark-vs-
operational gap). The fix is not retraining — it is recalibrating the
threshold on one operational period and freezing it.

What this does (and does NOT do):
  * Adds/updates ONE operating point named "operational" in the artifact:
    the TSS-optimal threshold on a single calibration year of live JSOC data,
    with full provenance (year, date, skill report on that year).
  * Never touches the model, its probabilities, the default threshold, or the
    other operating points. Live behavior only changes if you opt in by
    setting `sharp_live.operating_point: operational` in config.yaml.
  * Leakage discipline: the calibration year defaults to the EARLIEST year
    strictly after the model's training span, so every later year remains an
    untouched test set. The transfer check below never scores the
    calibration year itself.

    python -m solarflare.recalibrate                      # default deployed model
    python -m solarflare.recalibrate --variant multiyear  # a config-defined variant
    python -m solarflare.recalibrate --year 2015          # override the calibration year
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np

from . import data as dataio
from . import evaluate as ev
from . import sharpdata
from .config import load_config
from .scorecard import detect_operational_years, frozen_tss
from .sharp_live import variant_spec


def _artifact_path(cfg: dict, variant: str | None) -> str:
    label, path = variant_spec(cfg, variant)
    if path is None:
        raise KeyError(f"unknown variant '{variant}' — see config sharp_live.variants")
    if not os.path.isabs(path):
        path = os.path.join(cfg["_project_root"], path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"model artifact not found: {path}")
    return path


def _training_cutoff_year(payload: dict) -> int:
    """Last calendar year the model saw in training (from saved data_span)."""
    span = payload.get("data_span") or []
    try:
        return datetime.fromisoformat(str(span[1]).replace("Z", "+00:00")).year
    except (IndexError, ValueError, TypeError):
        return 2014                                    # conservative fallback


def _year_matrix(cfg: dict, year: int):
    dd = os.path.join(cfg["_project_root"], "data", "sharp_live")
    d = sharpdata.load_dataset(os.path.join(dd, f"dataset_{year}.npz"))
    return dataio.build_matrix(d["X3d"], cfg), d["y"]


def recalibrate(variant: str | None = None, year: int | None = None,
                cfg: dict | None = None, save: bool = True) -> dict:
    cfg = cfg or load_config()
    path = _artifact_path(cfg, variant)
    payload = joblib.load(path)
    cutoff = _training_cutoff_year(payload)
    dd = os.path.join(cfg["_project_root"], "data", "sharp_live")
    valid = [y for y in detect_operational_years(dd) if y > cutoff]
    # detect_operational_years excludes the standard training-era years; also
    # respect THIS artifact's own span (e.g. the multiyear variant ends 2014).
    if year is None:
        if not valid:
            raise RuntimeError(f"no operational dataset after the training span "
                               f"(ends {cutoff}) — build one first")
        year = valid[0]                                # earliest -> later years stay untouched
    elif year <= cutoff:
        raise ValueError(f"calibration year {year} overlaps the training span "
                         f"(ends {cutoff}) — that would be leakage")

    label, _ = variant_spec(cfg, variant)
    print(f"Recalibrating '{label}' on live JSOC {year} (training ends {cutoff}) ...")
    X, ycal = _year_matrix(cfg, year)
    p = payload["model"].predict_proba(X)[:, 1]
    thr, tss_cal = ev.best_threshold(ycal, p, "tss")
    report = ev.full_report(ycal, (p >= thr).astype(int))
    old_thr = float(payload["operating_points"]["high_recall"]["threshold"])
    print(f"  threshold: {old_thr:.3f} (training-val) -> {thr:.3f} (recalibrated)  "
          f"[TSS {tss_cal:.3f} on {year}, n={len(ycal)}]")

    # Honest transfer check: frozen at each threshold on the LATER years only.
    transfer = {}
    for ty in [y for y in valid if y != year]:
        Xt, yt = _year_matrix(cfg, ty)
        pt = payload["model"].predict_proba(Xt)[:, 1]
        transfer[str(ty)] = {"frozen_tss_training_val_thr": round(frozen_tss(yt, pt, old_thr), 3),
                             "frozen_tss_recalibrated_thr": round(frozen_tss(yt, pt, float(thr)), 3),
                             "n": int(len(yt)), "positives": int(yt.sum())}
        print(f"  {ty}: frozen TSS {transfer[str(ty)]['frozen_tss_training_val_thr']:.3f} "
              f"-> {transfer[str(ty)]['frozen_tss_recalibrated_thr']:.3f}")

    payload.setdefault("operating_points", {})["operational"] = {
        "threshold": float(thr),
        "calibrated_on_year": int(year),
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "calibration_report": report,
        "transfer_check": transfer,
        "note": "threshold recalibrated on one operational year of live JSOC data; "
                "opt in via config sharp_live.operating_point: operational",
    }
    if save:
        joblib.dump(payload, path)
        meta_path = path.replace(".joblib", ".meta.json")
        meta = {k: v for k, v in payload.items() if k != "model"}
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)
        print(f"Saved 'operational' operating point -> {path} (+ .meta.json)")
    return {"variant": variant or "", "label": label, "calibration_year": int(year),
            "old_threshold": old_thr, "new_threshold": float(thr),
            "calibration_tss": round(float(tss_cal), 3), "transfer": transfer}


def main():
    ap = argparse.ArgumentParser(description="Recalibrate a deployed model's threshold "
                                             "on live operational data.")
    ap.add_argument("--variant", default="", help="variant key (blank = default deployed model)")
    ap.add_argument("--year", type=int, default=None, help="calibration year (default: earliest "
                                                           "operational year after training)")
    ap.add_argument("--dry-run", action="store_true", help="compute but do not save")
    args = ap.parse_args()
    out = recalibrate(variant=args.variant or None, year=args.year, save=not args.dry_run)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
