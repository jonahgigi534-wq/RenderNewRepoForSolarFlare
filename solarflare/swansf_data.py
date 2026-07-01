"""Parse the ORIGINAL SWAN-SF benchmark (Harvard Dataverse doi:10.7910/DVN/EBCFKM)
into our (n, 60, 17) window format, so the live-JSOC model can train on the full
solar-cycle benchmark instead of just 2014.

SWAN-SF layout inside each partitionN_instances.tar.gz:
    partitionN/FL/<instance>.csv   -> an M-class+ flare followed (label 1)
    partitionN/NF/<instance>.csv   -> it did not (label 0)
Each CSV is a tab-separated 12 h MVTS instance (60 timesteps) of RAW physical
SHARP params. We keep only the 17 params JSOC's hmi.sharp_cea_720s serves live
(config sharp_live.keywords), so a model trained here still runs on live data.

Streams straight from the .tar.gz (no multi-GB disk extraction). Filename encodes
the active region (ar<N>) and window start (s<ISO>) for a leakage-free split.
"""
from __future__ import annotations

import re
import tarfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import sharpdata
from .config import load_config

_FN = re.compile(r"_ar(\d+)_s(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")


def _parse_name(name: str):
    m = _FN.search(name)
    if not m:
        return 0, None
    try:
        t = datetime.fromisoformat(m.group(2)).replace(tzinfo=timezone.utc)
    except ValueError:
        t = None
    return int(m.group(1)), t


def parse_tar(tar_path: str, cfg: dict, *, win: int = 60, min_finite: float = 0.5,
              verbose: bool = True):
    """Stream a SWAN-SF partition tar.gz -> (X3d, y, groups, end_times)."""
    keys = cfg["sharp_live"]["keywords"]                  # the 17 JSOC-serveable params
    X, y, groups, ends = [], [], [], []
    n_seen = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".csv"):
                continue
            parts = m.name.split("/")
            if len(parts) < 3 or parts[-2] not in ("FL", "NF"):
                continue
            label = 1 if parts[-2] == "FL" else 0
            fobj = tf.extractfile(m)
            if fobj is None:
                continue
            try:
                df = pd.read_csv(fobj, sep="\t", usecols=keys)
                mat = df[keys].to_numpy(dtype=float)      # (T, 17) in config order
            except Exception:                             # noqa: BLE001 (skip malformed)
                continue
            n_seen += 1
            if mat.ndim != 2 or mat.shape[0] == 0:
                continue
            if mat.shape[0] >= win:                       # enforce exactly `win` timesteps
                mat = mat[-win:]
            else:
                pad = np.full((win - mat.shape[0], len(keys)), np.nan)
                mat = np.vstack([pad, mat])
            if np.isfinite(mat).mean() < min_finite:
                continue
            ar, t = _parse_name(parts[-1])
            X.append(mat)
            y.append(label)
            groups.append(ar)
            ends.append(t or datetime(2011, 1, 1, tzinfo=timezone.utc))
            if verbose and len(X) % 5000 == 0:
                print(f"  parsed {len(X)} kept / {n_seen} seen ...", flush=True)
    if not X:
        return np.empty((0, win, len(keys))), np.array([]), np.array([]), []
    return np.asarray(X), np.asarray(y, dtype=int), np.asarray(groups), ends


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Parse SWAN-SF partition tar(s) -> npz.")
    ap.add_argument("--tars", nargs="+", required=True, help="partitionN_instances.tar.gz path(s)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config()
    Xs, ys, gs, es = [], [], [], []
    for tp in args.tars:
        print(f"parsing {tp} ...", flush=True)
        X, y, g, e = parse_tar(tp, cfg)
        print(f"  -> {len(y)} instances, {int(y.sum())} positive ({y.mean():.3%})", flush=True)
        Xs.append(X); ys.append(y); gs.append(g); es.extend(e)
    X = np.concatenate(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
    d = {"X3d": X, "y": y, "groups": g, "end_times": es}
    sharpdata.save_dataset(d, args.out)
    print(f"\nSWAN-SF dataset: {len(y)} samples, {int(y.sum())} positive "
          f"({y.mean():.3%}), {len(set(g.tolist()))} regions -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
