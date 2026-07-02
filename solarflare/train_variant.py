"""Train and save a DEPLOYABLE model variant defined in config.yaml
(`sharp_live.variants`) — SEPARATELY from the default deployed model, so the
research finding (does more live training data help operationally?) becomes a
live, comparable product feature instead of only a backtest number.

The default deployed model (sharp_live.model_path) is never touched: this
writes to the variant's own model_path.

    python -m solarflare.train_variant multiyear

Serve it live at  GET /api/sharp_live?variant=multiyear  (see solarflare.sharp_live).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from . import sharpdata, sharptrain
from .config import load_config


def train_variant(key: str, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    variants = cfg["sharp_live"].get("variants", {})
    if key not in variants:
        raise KeyError(f"no variant '{key}' in config sharp_live.variants "
                       f"(available: {sorted(variants)})")
    spec = variants[key]
    years = spec["train_years"]
    dd = os.path.join(cfg["_project_root"], "data", "sharp_live")
    missing = [y for y in years if not os.path.exists(os.path.join(dd, f"dataset_{y}.npz"))]
    if missing:
        raise FileNotFoundError(
            f"missing dataset(s) for years {missing} — build via "
            f"python -m solarflare.sharpdata --start YYYY-01-01 --end YYYY+1-01-01 "
            f"--out data/sharp_live/dataset_YYYY.npz")

    print(f"Training variant '{key}' ({spec.get('label', key)}) on years {years} ...")
    parts = [sharpdata.load_dataset(os.path.join(dd, f"dataset_{y}.npz")) for y in years]
    d = {"X3d": np.concatenate([p["X3d"] for p in parts]),
        "y": np.concatenate([p["y"] for p in parts]),
        "groups": np.concatenate([p["groups"] for p in parts]),
        "end_times": [t for p in parts for t in p["end_times"]]}

    out_path = spec["model_path"]
    if not os.path.isabs(out_path):
        out_path = os.path.join(cfg["_project_root"], out_path)
    payload = sharptrain.train(d, cfg, save=True, out_path=out_path)
    print(f"\nVariant '{key}' saved -> {out_path}")
    print(f"  winner={payload['winner']}  test TSS={payload['metrics']['tss']:.3f}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="Train a deployable live-SHARP model variant.")
    ap.add_argument("key", help="variant key from config.yaml sharp_live.variants (e.g. multiyear)")
    args = ap.parse_args()
    train_variant(args.key)


if __name__ == "__main__":
    main()
