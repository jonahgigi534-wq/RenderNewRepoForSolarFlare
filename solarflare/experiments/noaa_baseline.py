"""Score the DEPLOYED model against NOAA SWPC's own official M-class forecast —
on identical days of an operational year.

"We beat the benchmark" means little; the standard everyone is actually measured
against is NOAA's human-in-the-loop daily forecast. SWPC's Report of Solar and
Geophysical Activity (RSGA, issued ~2200 UT daily) contains explicit next-day
M-class event probabilities ("III. Event probabilities ... Class M 55/50/45").
NOAA archives them per year in the SWPC warehouse.

Protocol (full-disk, daily, honest):
  * NOAA:  the Class-M first-day probability from day D's RSGA -> forecast for D+1.
  * Ours:  the deployed live-SHARP model's windows ENDING on day D, one per
    region (its last window that day), combined to a full-disk probability
    1 - prod(1 - p_region) -> forecast for the 24 h after each window ends.
  * Truth: did ANY M-class+ flare occur on day D+1 (HEK / SWPC event list).
  * Scores: Brier (proper), Brier skill vs climatology, and peak TSS. Same days,
    same truth, both forecasters.

Caveat stated up front: our 24 h window starts at the window end (staggered
through day D), NOAA's covers calendar day D+1 — close but not identical
horizons. Both are scored against the same daily truth, and the comparison is
noted in the output.

Writes noaa_baseline.json (merged into /api/scorecard by the server).

Run:  python -m solarflare.experiments.noaa_baseline
"""
from __future__ import annotations

import io
import json
import os
import re
import tarfile
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import numpy as np
import requests

from .. import data as dataio
from .. import sharp_live, sharpdata
from ..config import load_config
from ..scorecard import peak_tss

# SWPC's warehouse is an FTP server first and foremost; the HTTPS mirror does
# not resolve from every network, so FTP is tried first.
WAREHOUSE_URLS = ["ftp://ftp.swpc.noaa.gov/pub/warehouse",
                  "https://ftp.swpc.noaa.gov/pub/warehouse"]
_RSGA_NAME = re.compile(r"(\d{8})RSGA\.txt$", re.IGNORECASE)
_CLASS_M = re.compile(r"Class\s*M\s+(\d{1,3})\s*/", re.IGNORECASE)


# ----------------------------------------------------------------------
# NOAA archived forecasts (RSGA warehouse tarball, cached on disk)
# ----------------------------------------------------------------------
def fetch_rsga_year(year: int, cache_dir: str, *, verbose: bool = True) -> dict:
    """{date issued -> next-day Class-M probability (0..1)} for one year."""
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, f"noaa_rsga_{year}.tar.gz")
    if not os.path.exists(local):
        import urllib.request
        blob, last = None, None
        for base in WAREHOUSE_URLS:
            url = f"{base}/{year}/{year}_RSGA.tar.gz"
            if verbose:
                print(f"  downloading {url} ...", flush=True)
            try:
                with urllib.request.urlopen(url, timeout=120) as r:   # noqa: S310 (fixed NOAA hosts)
                    blob = r.read()
                break
            except Exception as exc:                      # noqa: BLE001 (try next mirror)
                last = exc
                if verbose:
                    print(f"    failed: {type(exc).__name__}", flush=True)
        if blob is None:
            raise RuntimeError(f"RSGA archive unreachable for {year}: {last}")
        with open(local, "wb") as fh:
            fh.write(blob)
    out = {}
    with tarfile.open(local, "r:gz") as tf:
        for m in tf:
            nm = _RSGA_NAME.search(m.name or "")
            if not m.isfile() or not nm:
                continue
            fobj = tf.extractfile(m)
            if fobj is None:
                continue
            try:
                text = io.TextIOWrapper(fobj, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            pm = _CLASS_M.search(text)
            if not pm:
                continue
            try:
                issued = datetime.strptime(nm.group(1), "%Y%m%d").date()
                out[issued] = min(100, int(pm.group(1))) / 100.0
            except ValueError:
                continue
    if verbose:
        print(f"  RSGA {year}: {len(out)} daily Class-M forecasts parsed", flush=True)
    return out


# ----------------------------------------------------------------------
# Daily truth: any M+ flare that calendar day (HEK, AR number NOT required)
# ----------------------------------------------------------------------
def fetch_daily_m_flares(t0: datetime, t1: datetime, *, verbose: bool = True) -> set:
    """Set of UTC dates on which at least one M-class+ flare peaked."""
    days: set = set()
    cur = t0
    while cur < t1:
        nxt = min(cur + timedelta(days=28), t1)
        params = {
            "cmd": "search", "type": "column", "event_type": "FL",
            "event_starttime": cur.strftime("%Y-%m-%dT%H:%M:%S"),
            "event_endtime": nxt.strftime("%Y-%m-%dT%H:%M:%S"),
            "event_coordsys": "helioprojective",
            "x1": "-1200", "x2": "1200", "y1": "-1200", "y2": "1200",
            "result_limit": "2000", "cosec": "2",
            "return": "fl_goescls,event_peaktime,frm_name",
        }
        try:
            j = requests.get(sharpdata.HEK_URL, params=params, timeout=120).json()
        except Exception as exc:                          # noqa: BLE001
            if verbose:
                print(f"  HEK {cur.date()} failed: {type(exc).__name__}", flush=True)
            cur = nxt
            continue
        for e in j.get("result", []):
            if e.get("frm_name") != "SWPC":
                continue
            cls = (e.get("fl_goescls") or "")[:1]
            pk = e.get("event_peaktime")
            if cls not in ("M", "X") or not pk:
                continue
            try:
                days.add(datetime.fromisoformat(str(pk)).date())
            except ValueError:
                continue
        cur = nxt
    if verbose:
        print(f"  truth: {len(days)} M+ flare days", flush=True)
    return days


# ----------------------------------------------------------------------
# Our full-disk daily forecast from the DEPLOYED model
# ----------------------------------------------------------------------
def daily_fulldisk_ours(year: int, dd: str, cfg: dict) -> dict:
    """{date D -> full-disk P(M+ in ~24 h) from windows ending on D}."""
    payload = sharp_live.load_model(cfg)
    if payload is None:
        raise RuntimeError("deployed live SHARP model not found")
    d = sharpdata.load_dataset(os.path.join(dd, f"dataset_{year}.npz"))
    Xf = dataio.build_matrix(d["X3d"], cfg)
    proba = payload["model"].predict_proba(Xf)[:, 1]
    last_per_region_day: dict = {}
    for i, (g, t) in enumerate(zip(d["groups"], d["end_times"])):
        key = (t.date(), int(g))
        prev = last_per_region_day.get(key)
        if prev is None or d["end_times"][prev] < t:
            last_per_region_day[key] = i
    by_day = defaultdict(list)
    for (day, _g), i in last_per_region_day.items():
        by_day[day].append(float(proba[i]))
    return {day: 1.0 - float(np.prod([1.0 - p for p in ps]))
            for day, ps in by_day.items()}


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def _brier(pairs):
    return float(np.mean([(p - y) ** 2 for p, y in pairs]))


def _score(pairs) -> dict:
    """pairs = [(p, y)] on identical days."""
    y = np.array([y for _, y in pairs], dtype=int)
    p = np.array([p for p, _ in pairs], dtype=float)
    base = float(y.mean())
    brier = _brier(pairs)
    brier_clim = float(np.mean((base - y) ** 2))
    return {
        "n_days": int(len(y)), "flare_days": int(y.sum()), "base_rate": round(base, 3),
        "brier": round(brier, 4),
        "brier_skill_vs_climatology": round(1.0 - brier / brier_clim, 3) if brier_clim else None,
        "peak_tss": round(peak_tss(y, p), 3),
    }


def run(cfg: dict | None = None, years: list[int] | None = None) -> dict:
    cfg = cfg or load_config()
    root = cfg["_project_root"]
    dd = os.path.join(root, "data", "sharp_live")
    from ..scorecard import detect_operational_years
    years = years or detect_operational_years(dd)
    if not years:
        raise RuntimeError("no operational dataset_YYYY.npz found")

    per_year = {}
    for yr in years:
        print(f"=== {yr} ===", flush=True)
        noaa = fetch_rsga_year(yr, dd)
        ours = daily_fulldisk_ours(yr, dd, cfg)
        t0 = datetime(yr, 1, 1, tzinfo=timezone.utc)
        t1 = datetime(yr + 1, 1, 2, tzinfo=timezone.utc)
        flare_days = fetch_daily_m_flares(t0, t1)
        # NOAA's day-D forecast applies to D+1; ours from windows ending on D
        # applies to the following 24 h. Score both against "M+ flare on D+1",
        # restricted to days where BOTH forecasts exist.
        noaa_pairs, our_pairs = [], []
        for d_issue, p_noaa in sorted(noaa.items()):
            target = d_issue + timedelta(days=1)
            if target > date(yr, 12, 31):
                continue
            p_ours = ours.get(d_issue)
            if p_ours is None:
                continue
            y = 1 if target in flare_days else 0
            noaa_pairs.append((p_noaa, y))
            our_pairs.append((p_ours, y))
        if not noaa_pairs:
            print("  no overlapping forecast days — skipped", flush=True)
            continue
        per_year[str(yr)] = {"noaa_official": _score(noaa_pairs),
                             "helios_deployed": _score(our_pairs)}
        print(f"  {len(noaa_pairs)} shared days  "
              f"NOAA Brier={per_year[str(yr)]['noaa_official']['brier']}  "
              f"ours Brier={per_year[str(yr)]['helios_deployed']['brier']}", flush=True)

    out = {
        "title": "Deployed model vs. NOAA SWPC official M-class forecast",
        "protocol": "full-disk daily P(M+); NOAA = archived RSGA next-day Class-M "
                    "probability; ours = deployed live-SHARP model, last window per "
                    "region per day combined as 1-prod(1-p); truth = any M+ flare on "
                    "the target day (HEK/SWPC). Same days, same truth.",
        "caveat": "horizons are close but not identical: NOAA's probability covers "
                  "calendar day D+1; our windows end during day D and cover the "
                  "following 24 h. Brier is the primary metric (proper); climatology "
                  "(constant base rate) has Brier-skill 0 and peak TSS 0.",
        "years": per_year,
    }
    path = os.path.join(root, "noaa_baseline.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {path}", flush=True)
    return out


def main():
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
