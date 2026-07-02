"""Measure the HEK/SWPC AR-attribution rate per year — the label-quality basis
for `scorecard.label_attribution_by_year` in config.yaml.

Our labeler (sharpdata.fetch_flares -> build_windows) can only mark a positive
window when HEK/SWPC gives the flare a NOAA AR number, so the fraction of M+
flares WITH an AR number bounds label completeness for that year. A year far
below 1.0 under-counts positives and its TSS measures catalog decay, not model
skill (measured 2026-07-02: 2014 = 0.94, 2023 = 0.15).

Run:  python scripts/label_attribution.py [first_year] [last_year]
then paste the printed rates into config.yaml scorecard.label_attribution_by_year.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import requests

HEK_URL = "https://www.lmsal.com/hek/her"


def year_attribution(year: int, chunk_days: int = 28) -> tuple[int, int]:
    """(with_ar, total) M+ SWPC flares for the year."""
    t = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    with_ar = total = 0
    while t < end:
        nxt = min(t + timedelta(days=chunk_days), end)
        params = {
            "cmd": "search", "type": "column", "event_type": "FL",
            "event_starttime": t.strftime("%Y-%m-%dT%H:%M:%S"),
            "event_endtime": nxt.strftime("%Y-%m-%dT%H:%M:%S"),
            "event_coordsys": "helioprojective",
            "x1": "-1200", "x2": "1200", "y1": "-1200", "y2": "1200",
            "result_limit": "2000", "cosec": "2",
            "return": "fl_goescls,ar_noaanum,frm_name",
        }
        try:
            j = requests.get(HEK_URL, params=params, timeout=120).json()
        except Exception as exc:                          # noqa: BLE001
            print(f"  {year} chunk {t.date()} failed ({type(exc).__name__}) — skipped")
            t = nxt
            continue
        for e in j.get("result", []):
            if e.get("frm_name") != "SWPC":
                continue
            if (e.get("fl_goescls") or "")[:1] not in ("M", "X"):
                continue
            total += 1
            if e.get("ar_noaanum"):
                with_ar += 1
        t = nxt
    return with_ar, total


def main():
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 2013
    last = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    print(f"HEK/SWPC M+ AR-attribution rate, {first}-{last}")
    print("year  with_AR  total   rate")
    for year in range(first, last + 1):
        w, t = year_attribution(year)
        rate = w / t if t else float("nan")
        print(f"{year}  {w:>7}  {t:>5}   {rate:.2f}", flush=True)


if __name__ == "__main__":
    main()
