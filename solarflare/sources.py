"""Resilient live-data fetchers for NOAA SWPC and NASA.

Every fetch:
  1. tries each configured URL in order (primary -> secondary -> shorter feed),
  2. has a hard timeout,
  3. on success, writes a timestamped cache file (the failsafe backup),
  4. on total failure, falls back to the most recent cache so the predictor
     can still answer (clearly flagged as stale).

Nothing here ever raises to the caller: it returns a FetchResult describing
exactly what happened (live / cached / unavailable), which the predictor turns
into an honest "data freshness" line in the forecast.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from .config import load_config


@dataclass
class FetchResult:
    ok: bool
    data: object = None
    source: str = ""           # which URL answered
    status: str = "unavailable"  # live | cached | unavailable
    age_minutes: float | None = None
    error: str = ""
    notes: list[str] = field(default_factory=list)


def _cache_path(cfg: dict, key: str) -> str:
    os.makedirs(cfg["paths"]["cache_dir"], exist_ok=True)
    return os.path.join(cfg["paths"]["cache_dir"], f"{key}.json")


def _write_cache(cfg: dict, key: str, data) -> None:
    try:
        with open(_cache_path(cfg, key), "w", encoding="utf-8") as fh:
            json.dump({"_cached_at": time.time(), "data": data}, fh)
    except OSError:
        pass


def _read_cache(cfg: dict, key: str) -> tuple[object, float] | None:
    path = _cache_path(cfg, key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        age_min = (time.time() - blob["_cached_at"]) / 60.0
        return blob["data"], age_min
    except (OSError, ValueError, KeyError):
        return None


def _fetch(urls: list[str], cfg: dict, cache_key: str, *, as_json: bool,
           params: dict | None = None, fresh_within_min: float | None = None) -> FetchResult:
    """Resilient fetch shared by JSON and plain-text sources: try each URL,
    cache successes, fall back to cache, never raise.

    If `fresh_within_min` is set and a cache entry is younger than that, the
    cache is returned WITHOUT hitting the network (polite caching for sources
    that ask callers not to over-poll, e.g. CelesTrak's ~2h cadence).
    """
    if fresh_within_min is not None:
        cached = _read_cache(cfg, cache_key)
        if cached is not None and cached[1] <= fresh_within_min:
            data, age = cached
            return FetchResult(ok=True, data=data, source="cache", status="cached",
                               age_minutes=age, notes=[f"fresh cache ({age:.0f} min)"])
    timeout = cfg["live"]["request_timeout_s"]
    notes = []
    for url in urls:
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent": "solar-flare-predictor/0.1"})
            r.raise_for_status()
            data = r.json() if as_json else r.text
            _write_cache(cfg, cache_key, data)
            return FetchResult(ok=True, data=data, source=url, status="live",
                               age_minutes=0.0, notes=notes)
        except Exception as exc:                      # noqa: BLE001 (resilient by design)
            notes.append(f"{url.split('/')[-1]} failed: {type(exc).__name__}")
            continue

    # Every source failed -> fall back to cache (the failsafe).
    cached = _read_cache(cfg, cache_key)
    if cached is not None:
        data, age = cached
        notes.append(f"using cached copy ({age:.0f} min old)")
        return FetchResult(ok=True, data=data, source="cache", status="cached",
                           age_minutes=age, notes=notes)
    return FetchResult(ok=False, status="unavailable",
                       error="all sources failed and no cache available",
                       notes=notes)


def _fetch_json(urls, cfg, cache_key, params=None):
    return _fetch(urls, cfg, cache_key, as_json=True, params=params)


def _fetch_text(urls, cfg, cache_key):
    return _fetch(urls, cfg, cache_key, as_json=False)


# ----------------------------------------------------------------------
# Public fetchers
# ----------------------------------------------------------------------
def _parse_iso_utc(ts) -> datetime | None:
    """Parse an ISO-8601 timestamp to an aware UTC datetime, or None if malformed.
    A naive timestamp is assumed to already be UTC (NOAA feeds are UTC) so it can
    be safely differenced against datetime.now(timezone.utc) without a TypeError."""
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def get_xray_flux(cfg: dict | None = None) -> FetchResult:
    """GOES long-channel (0.1-0.8 nm) X-ray flux time series, newest last."""
    cfg = cfg or load_config()
    res = _fetch_json(cfg["live"]["xray_sources"], cfg, "xray_flux")
    if not res.ok:
        return res
    # Keep only the long band that defines flare class; sort by time. Stay
    # defensive about the feed/cache shape so a schema change can never raise to
    # the caller — this is the always-on path feeding nowcast/hazard/forecast.
    data = res.data if isinstance(res.data, list) else []
    rows = [d for d in data if isinstance(d, dict)
            and d.get("energy") == "0.1-0.8nm" and d.get("time_tag")]
    rows.sort(key=lambda d: d["time_tag"])
    res.data = rows
    if rows:
        last_t = _parse_iso_utc(rows[-1]["time_tag"])
        if last_t is not None:
            res.age_minutes = (datetime.now(timezone.utc) - last_t).total_seconds() / 60.0
    return res


def get_solar_regions(cfg: dict | None = None) -> FetchResult:
    """NOAA Solar Region Summary, filtered to the MOST RECENT observed day.

    The raw feed carries ~30 days of history; we keep only the latest date so
    counts reflect the current Sun. Each region also carries NOAA's official
    c/m/x_flare_probability (percent), which we surface for corroboration.
    """
    cfg = cfg or load_config()
    res = _fetch_json(cfg["live"]["srs_sources"], cfg, "solar_regions")
    if not res.ok or not isinstance(res.data, list) or not res.data:
        return res
    dates = [r.get("observed_date") for r in res.data if r.get("observed_date")]
    if dates:
        latest = max(dates)
        res.data = [r for r in res.data if r.get("observed_date") == latest]
        res.notes.append(f"regions for {latest}")
    return res


def noaa_region_forecast(cfg: dict | None = None) -> dict:
    """Aggregate NOAA's per-region daily flare probabilities into a Sun-wide
    24h probability via P(any) = 1 - prod(1 - p_region). This is NOAA SWPC's
    *official* forecast, used here as an authoritative corroboration track."""
    cfg = cfg or load_config()
    res = get_solar_regions(cfg)
    if not res.ok or not isinstance(res.data, list) or not res.data:
        return {"available": False, "reason": res.error or "no region data"}
    import numpy as np
    regions = res.data

    def _agg(field):
        ps = [r.get(field) for r in regions if isinstance(r.get(field), (int, float))]
        if not ps:
            return None
        comp = 1.0 - np.prod([1.0 - min(max(p / 100.0, 0.0), 1.0) for p in ps])
        return round(float(comp), 4)

    return {
        "available": True,
        "model": "noaa-swpc-region-forecast",
        "observed_date": regions[0].get("observed_date"),
        "n_regions": len(regions),
        "p_C_or_greater_24h": _agg("c_flare_probability"),
        "p_M_or_greater_24h": _agg("m_flare_probability"),
        "p_X_class_24h": _agg("x_flare_probability"),
        "status": res.status,
    }


def get_recent_flares(cfg: dict | None = None) -> FetchResult:
    """GOES flare event list for the last 7 days (corroboration / display)."""
    cfg = cfg or load_config()
    return _fetch_json(cfg["live"]["flare_event_sources"], cfg, "recent_flares")


def get_donki_flares(cfg: dict | None = None) -> FetchResult:
    """NASA DONKI flare catalogue (extra source; optional API key)."""
    cfg = cfg or load_config()
    d = cfg["live"].get("donki", {})
    if not d.get("enabled"):
        return FetchResult(ok=False, status="unavailable", error="DONKI disabled")
    return _fetch_json([d["url"]], cfg, "donki",
                       params={"api_key": d.get("api_key", "DEMO_KEY")})


def get_drap(cfg: dict | None = None) -> FetchResult:
    """NOAA D-RAP global HF-absorption grid (plain text)."""
    cfg = cfg or load_config()
    return _fetch_text(cfg["hazard"]["drap_sources"], cfg, "drap")


def get_noaa_scales(cfg: dict | None = None) -> FetchResult:
    """NOAA R/S/G scale values (current + 3-day forecast)."""
    cfg = cfg or load_config()
    return _fetch_json(cfg["hazard"]["scales_sources"], cfg, "noaa_scales")


def get_celestrak_tle(group: str, cfg: dict | None = None) -> FetchResult:
    """CelesTrak GP orbital data (TLE) for a satellite group, cached >= 2h."""
    cfg = cfg or load_config()
    s = cfg["satellites"]
    url = s["gp_url"].format(group=group)
    ttl = s["cache_hours"] * 60
    return _fetch([url], cfg, f"tle_{group}", as_json=False, fresh_within_min=ttl)


def get_ovation(cfg: dict | None = None) -> FetchResult:
    """NOAA OVATION auroral-oval grid (probability of visible aurora, global)."""
    cfg = cfg or load_config()
    return _fetch_json(cfg["geomag"]["ovation_sources"], cfg, "ovation_aurora")


def get_kp(cfg: dict | None = None) -> FetchResult:
    """NOAA planetary Kp index time series (current global geomagnetic activity)."""
    cfg = cfg or load_config()
    return _fetch_json(cfg["geomag"]["kp_sources"], cfg, "planetary_kp")


def get_l1_mag(cfg: dict | None = None) -> FetchResult:
    """Real-time L1 interplanetary magnetic field (DSCOVR/ACE), 7-day 1-min."""
    cfg = cfg or load_config()
    return _fetch_json(cfg["storm"]["l1_mag_sources"], cfg, "l1_mag")


def get_l1_plasma(cfg: dict | None = None) -> FetchResult:
    """Real-time L1 solar-wind plasma (speed/density), 7-day 1-min."""
    cfg = cfg or load_config()
    return _fetch_json(cfg["storm"]["l1_plasma_sources"], cfg, "l1_plasma")


def get_kp_forecast(cfg: dict | None = None) -> FetchResult:
    """NOAA SWPC official planetary Kp forecast (storm-model corroboration track)."""
    cfg = cfg or load_config()
    return _fetch_json(cfg["storm"]["kp_forecast_sources"], cfg, "kp_forecast")


def get_donki_cme(cfg: dict | None = None, start_date: str = "", end_date: str = "") -> FetchResult:
    """NASA DONKI CME catalogue for a date range (STEP 7 upstream-CME track).

    Cached PER date-range (DONKI is fetched in chunks and times out on multi-year
    spans): each chunk gets its own cache key, so a failed/rate-limited chunk can't
    fall back to a different window's data, and successful chunks are reused on
    reruns instead of re-hitting the heavily rate-limited DEMO_KEY API.
    """
    cfg = cfg or load_config()
    key = cfg.get("live", {}).get("donki", {}).get("api_key", "DEMO_KEY")
    urls = cfg["experiment"]["donki_cme_sources"]
    ttl = cfg["experiment"].get("cme_cache_days", 7) * 1440
    return _fetch(list(urls), cfg, f"donki_cme_{start_date}", as_json=True,
                  params={"startDate": start_date, "endDate": end_date, "api_key": key},
                  fresh_within_min=ttl)


def get_omni(cfg: dict, time_min: str, time_max: str) -> FetchResult:
    """NASA OMNI hourly via CDAWeb HAPI (CSV) for storm-model training. Cached
    (the training range is fixed per day, so re-runs reuse the cache)."""
    s = cfg["storm"]
    url = (f"{s['hapi_base']}/data?id={s['hapi_dataset']}"
           f"&parameters={','.join(s['omni_params'])}"
           f"&time.min={time_min}&time.max={time_max}&format=csv")
    ttl = s.get("cache_hours", 6) * 60
    return _fetch([url], cfg, "omni_train", as_json=False, fresh_within_min=ttl)
