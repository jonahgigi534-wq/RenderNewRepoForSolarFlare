"""Build a SHARP flare-forecasting dataset straight from JSOC + HEK, in RAW
physical units — so we own the entire pipeline and never depend on an external
normalization we can't reproduce.

Sources
  * SHARP magnetic parameters: JSOC `hmi.sharp_cea_720s` keyword time series
    (via the `drms` client). Same 24 SHARP keywords the existing model uses.
  * Flare labels: NOAA SWPC's GOES flare list via the HEK API (M-class+ events
    with their NOAA active-region number and peak time).

A sample = a 12 h window (60 records at the 720 s / 12-min cadence) of the 24
SHARP params for ONE active region; the label is 1 if an M-class-or-greater
flare from that region's NOAA AR occurs within the next 24 h.

Raw features are summarised by `solarflare.data.build_matrix` (24 params x 7
stats = 168), exactly like the existing model — but here a StandardScaler is fit
and SAVED with the classifier, so live JSOC data goes through the identical
transform. This module is OFFLINE/training only (it is never imported by the API
request path).
"""
from __future__ import annotations

import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

from .config import load_config

CADENCE_MIN = 12                      # hmi.sharp_cea_720s -> one record / 12 min
SERIES = "hmi.sharp_cea_720s"
HEK_URL = "https://www.lmsal.com/hek/her"


# ----------------------------------------------------------------------
# Time helpers
# ----------------------------------------------------------------------
def _parse_trec(s: str) -> datetime | None:
    """'2014.01.01_00:00:00_TAI' -> aware UTC datetime (TAI~UTC at our cadence)."""
    try:
        s = str(s).replace("_TAI", "").strip()
        d, t = s.split("_")
        return datetime.strptime(f"{d} {t}", "%Y.%m.%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


def _jsoc_time(dt: datetime) -> str:
    return dt.strftime("%Y.%m.%d_%H:%M:%S_TAI")


# ----------------------------------------------------------------------
# Fetch: SHARP keyword time series (JSOC) and flares (HEK)
# ----------------------------------------------------------------------
def _query_range(client, keystr: str, t0: datetime, t1: datetime,
                 series: str = SERIES, depth: int = 0):
    """Query [t0, t1); JSOC caps result size, so on a size error split and recurse.
    Transient network failures (connection reset, timeout) retry with backoff — a
    multi-hour build must not die on one dropped socket."""
    import drms
    import pandas as pd
    days = max(1, round((t1 - t0).total_seconds() / 86400))
    q = f"{series}[][{_jsoc_time(t0)}/{days}d]"
    for attempt in range(4):
        try:
            return client.query(q, key=keystr)
        except drms.exceptions.DrmsQueryError:
            if days <= 1 or depth > 7:
                raise
            mid = t0 + (t1 - t0) / 2
            return pd.concat([_query_range(client, keystr, t0, mid, series, depth + 1),
                              _query_range(client, keystr, mid, t1, series, depth + 1)],
                             ignore_index=True)
        except Exception:                         # noqa: BLE001 (network hiccup — retry)
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))


def fetch_sharp(t_start: datetime, t_end: datetime, cfg: dict, *, chunk_days: int | None = None,
                verbose: bool = True, series: str | None = None):
    """Return per-record tuples (T_REC, HARPNUM, NOAA_AR, vec[17]) pulled from JSOC
    in chunks (auto-split if a chunk exceeds JSOC's result cap). Needs `drms`.

    `series` defaults to the definitive science series (training/datasets). Live
    inference passes config `sharp_live.live_series` (the _nrt series) instead —
    the definitive series lags real time by weeks, NRT by ~1 h."""
    import drms
    client = drms.Client()
    series = series or SERIES
    keys = cfg["sharp_live"]["keywords"]                  # the 17 available channels, in order
    keystr = "T_REC,HARPNUM,NOAA_AR," + ",".join(keys)
    chunk_days = chunk_days or int(cfg["sharp_live"].get("chunk_days", 5))
    rows = []
    cur = t_start
    while cur < t_end:
        nxt = min(cur + timedelta(days=chunk_days), t_end)
        t0 = time.time()
        df = _query_range(client, keystr, cur, nxt, series)
        if verbose:
            print(f"  JSOC {cur.date()} +{(nxt-cur).days}d -> {len(df):>6} rows "
                  f"({time.time()-t0:.1f}s)")
        if len(df):
            trec = df["T_REC"].to_numpy()
            harp = df["HARPNUM"].to_numpy()
            arr = np.nan_to_num(df["NOAA_AR"].to_numpy(dtype=float), nan=0.0).astype(int)
            mat = df[keys].to_numpy(dtype=float)          # (n, 17) raw physical values
            for i in range(len(df)):
                t = _parse_trec(trec[i])
                if t is not None:
                    rows.append((t, int(harp[i]), int(arr[i]), mat[i]))
        cur = nxt
    return rows


def fetch_flares(t_start: datetime, t_end: datetime, *, chunk_days: int = 28,
                 source: str = "SWPC", verbose: bool = True) -> dict:
    """M-class+ GOES flares with a NOAA AR number, from HEK (NOAA SWPC catalog).
    Returns {noaa_ar: sorted [datetime peak times]}."""
    by_ar: dict[int, list[datetime]] = {}
    cur = t_start
    n = 0
    while cur < t_end:
        nxt = min(cur + timedelta(days=chunk_days), t_end)
        params = {
            "cmd": "search", "type": "column", "event_type": "FL",
            "event_starttime": cur.strftime("%Y-%m-%dT%H:%M:%S"),
            "event_endtime": nxt.strftime("%Y-%m-%dT%H:%M:%S"),
            "event_coordsys": "helioprojective",
            "x1": "-1200", "x2": "1200", "y1": "-1200", "y2": "1200",
            "result_limit": "2000", "cosec": "2",
            "return": "fl_goescls,ar_noaanum,event_peaktime,frm_name",
        }
        try:
            j = requests.get(HEK_URL, params=params, timeout=120).json()
        except Exception as exc:                          # noqa: BLE001
            if verbose:
                print(f"  HEK {cur.date()} failed: {type(exc).__name__}")
            cur = nxt
            continue
        for e in j.get("result", []):
            if e.get("frm_name") != source:               # one authoritative catalog
                continue
            cls = e.get("fl_goescls") or ""
            if cls[:1] not in ("M", "X"):
                continue
            ar, pk = e.get("ar_noaanum"), e.get("event_peaktime")
            if not ar or not pk:
                continue
            try:
                t = datetime.fromisoformat(str(pk)).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            by_ar.setdefault(int(ar), []).append(t)
            n += 1
        cur = nxt
    for ar in by_ar:
        by_ar[ar].sort()
    if verbose:
        print(f"  HEK: {n} M/X flares with AR across {len(by_ar)} regions")
    return by_ar


# ----------------------------------------------------------------------
# Window + label construction
# ----------------------------------------------------------------------
def _flare_within(by_ar: dict, ar: int, t0: datetime, t1: datetime) -> bool:
    times = by_ar.get(ar)
    if not times:
        return False
    return any(t0 < t <= t1 for t in times)


def build_windows(rows: list, by_ar: dict, cfg: dict, *, obs_h: int = 12,
                  pred_h: int = 24, stride_h: int = 6, min_finite: float = 0.8):
    """Slice each region's record stream into 60-record (12 h) windows and label
    each by whether an M+ flare from its AR follows within `pred_h` hours.

    Returns X3d (n, 60, 24), y (n,), groups (n, HARPNUM), end_times (n,)."""
    win = obs_h * 60 // CADENCE_MIN                        # 60 records
    step = max(1, stride_h * 60 // CADENCE_MIN)            # records to advance
    max_span = timedelta(hours=obs_h * 1.4)               # reject windows straddling big gaps
    pred = timedelta(hours=pred_h)

    by_harp: dict[int, list] = {}
    for t, harp, ar, vec in rows:
        by_harp.setdefault(harp, []).append((t, ar, vec))

    X, y, groups, ends = [], [], [], []
    for harp, recs in by_harp.items():
        recs.sort(key=lambda r: r[0])
        n = len(recs)
        if n < win:
            continue
        times = [r[0] for r in recs]
        ars = np.fromiter((r[1] for r in recs), dtype=np.int64, count=n)
        mat = np.stack([r[2] for r in recs])              # (n, 17) once per region
        for i in range(0, n - win + 1, step):
            j = i + win - 1
            if times[j] - times[i] > max_span:            # gap inside the window
                continue
            block = mat[i:i + win]                        # (60, 17) view
            if np.isfinite(block).mean() < min_finite:    # too much missing data
                continue
            nz = ars[i:i + win]; nz = nz[nz > 0]
            ar = int(np.bincount(nz).argmax()) if nz.size else 0
            end_t = times[j]
            label = 1 if (ar and _flare_within(by_ar, ar, end_t, end_t + pred)) else 0
            X.append(block.copy()); y.append(label); groups.append(harp); ends.append(end_t)
    if not X:
        return np.empty((0, win, len(cfg["sharp_live"]["keywords"]))), np.array([]), np.array([]), []
    return np.asarray(X), np.asarray(y, dtype=int), np.asarray(groups), ends


# ----------------------------------------------------------------------
# Orchestration (offline)
# ----------------------------------------------------------------------
def build_dataset(t_start: datetime, t_end: datetime, cfg: dict | None = None,
                  *, obs_h: int = 12, pred_h: int = 24, stride_h: int = 6, verbose: bool = True):
    """Full pull -> windows for [t_start, t_end). Flares fetched out to t_end+pred_h
    so the last windows can be labeled."""
    cfg = cfg or load_config()
    if verbose:
        print(f"Fetching SHARP {t_start.date()} .. {t_end.date()} from JSOC ...")
    rows = fetch_sharp(t_start, t_end, cfg, verbose=verbose)
    if verbose:
        print(f"  total SHARP records: {len(rows)}")
        print("Fetching M+ flares (HEK / SWPC) ...")
    by_ar = fetch_flares(t_start, t_end + timedelta(hours=pred_h + 2), verbose=verbose)
    X3d, y, groups, ends = build_windows(rows, by_ar, cfg, obs_h=obs_h,
                                         pred_h=pred_h, stride_h=stride_h)
    return {"X3d": X3d, "y": y, "groups": groups, "end_times": ends,
            "n_records": len(rows), "n_flare_regions": len(by_ar)}


def save_dataset(d: dict, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path, X3d=d["X3d"], y=d["y"], groups=d["groups"],
        end_times=np.array([t.isoformat() for t in d["end_times"]]),
    )


def load_dataset(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    ends = [datetime.fromisoformat(s) for s in z["end_times"]]
    return {"X3d": z["X3d"], "y": z["y"], "groups": z["groups"], "end_times": ends}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build a SHARP flare dataset from JSOC+HEK.")
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default="2014-04-01")
    ap.add_argument("--stride-h", type=int, default=6)
    ap.add_argument("--out", default="", help="save the dataset to this .npz path")
    args = ap.parse_args()
    cfg = load_config()
    t0 = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    t1 = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    t = time.time()
    d = build_dataset(t0, t1, cfg, stride_h=args.stride_h)
    X, y = d["X3d"], d["y"]
    n, pos = len(y), int(y.sum()) if len(y) else 0
    print("\n=== DATASET ===")
    print(f"  window shape : {X.shape}")
    print(f"  samples      : {n}")
    print(f"  positives    : {pos}  (base rate {pos/n:.3%})" if n else "  no samples")
    print(f"  regions      : {len(set(d['groups'].tolist())) if n else 0}")
    print(f"  build time   : {time.time()-t:.0f}s")
    if args.out and n:
        save_dataset(d, args.out)
        print(f"  saved        : {args.out}")


if __name__ == "__main__":
    main()
