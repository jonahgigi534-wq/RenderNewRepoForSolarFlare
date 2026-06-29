"""Geomagnetic-storm dataset: NASA OMNI (L1 solar wind paired with Kp).

Part 2 of the project — a SEPARATE feature space from the flare model. The flare
model learns from the *Sun's* magnetic field (SHARP); this learns from the *solar
wind at L1* (Bz, speed, density, coupling) to forecast *Earth's* geomagnetic
response (Kp). Never mixed.

Pipeline mirrors the flare side (data.py): each sample summarises a window of
solar-wind history into a flat feature vector via the SAME summary stats, so we
reuse `data.build_matrix`. Label = "did a Kp >= threshold storm occur in the next
prediction window".

Training data is OMNI hourly from CDAWeb HAPI; a synthetic fixture lets the whole
pipeline run offline. Nothing here hits the network unless asked to.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from . import data as dataio
from . import sources
from .config import load_config

# Derived solar-wind channels, in a FIXED order shared by training + live inference.
CHANNELS = ["Bmag", "By", "Bz", "N", "V", "P", "clock", "newell"]

# OMNI fill sentinels by the omni_params order [F, By, Bz, N, V, P, Kp, Dst]
# (authoritative OMNI2_H0_MRG1HR values). A reading at/above these is "missing".
_FILL = {"F": 999.9, "B": 999.9, "N": 999.9, "V": 9999.0, "P": 99.99,
         "KP": 99.0, "DST": 99999.0}

# Dynamic-pressure constant, used IDENTICALLY in training, the synthetic fixture,
# and live inference so the P channel never skews between train and serve. (OMNI's
# own Pressure1800 carries an alpha-particle correction; recomputing from N,V with
# one constant everywhere keeps the feature distribution consistent.)
_P_CONST = 1.6726e-6


def feature_names(cfg: dict) -> list[str]:
    stats = cfg["storm"]["summary_stats"]
    return [f"{ch}__{st}" for ch in CHANNELS for st in stats]


# ----------------------------------------------------------------------
# Per-hour derived channels (clock angle + Newell coupling)
# ----------------------------------------------------------------------
def _derive(Bmag, By, Bz, N, V, P):
    """Stack the 8 model channels for aligned per-hour arrays (NaN-safe inputs).
    clock = IMF clock angle atan2(By,Bz); newell = Newell (2007) coupling
    dPhi/dt = V^(4/3) * Bt^(2/3) * sin(|clock|/2)^(8/3), Bt = sqrt(By^2+Bz^2)."""
    Bt = np.sqrt(np.square(By) + np.square(Bz))
    clock = np.arctan2(By, Bz)                                   # radians, -pi..pi
    newell = (np.power(np.abs(V), 4.0 / 3.0)
              * np.power(Bt, 2.0 / 3.0)
              * np.power(np.abs(np.sin(clock / 2.0)), 8.0 / 3.0))
    return np.column_stack([Bmag, By, Bz, N, V, P, clock, newell])  # (T, 8)


def _windows(channels: np.ndarray, kp: np.ndarray, times: np.ndarray, cfg: dict):
    """Slide an observation window over the hourly series; label by the peak Kp in
    the following prediction window. Steps by Kp's 3-hour cadence to cut redundancy.
    Returns (X3d (n,obs,8), y (n,), t (n,) window-end times)."""
    s = cfg["storm"]
    obs, pred = int(s["observation_window_h"]), int(s["prediction_window_h"])
    thr10 = float(s["kp_storm_threshold"]) * 10.0                # OMNI Kp is Kp*10
    stride = int(s.get("sample_stride_h", 3))
    X, y, t = [], [], []
    T = len(kp)
    for i in range(obs, T - pred, stride):
        win = channels[i - obs:i]                               # (obs, 8)
        future = kp[i:i + pred]
        fut = future[np.isfinite(future)]
        if fut.size == 0 or not np.isfinite(win).any():
            continue                                            # need a label + some inputs
        X.append(win)
        y.append(1 if np.nanmax(future) >= thr10 else 0)
        t.append(times[i])
    if not X:
        return np.empty((0, obs, len(CHANNELS))), np.empty((0,), int), np.empty((0,), object)
    return np.asarray(X, float), np.asarray(y, int), np.asarray(t, object)


# ----------------------------------------------------------------------
# OMNI parsing
# ----------------------------------------------------------------------
def parse_omni_csv(text: str):
    """OMNI HAPI CSV (Time,F,By,Bz,N,V,P,Kp,Dst) -> aligned hourly arrays with
    fill values masked to NaN. Returns (times[obj], channels(T,8), kp(T))."""
    times, F, By, Bz, N, V, P, KP = [], [], [], [], [], [], [], []
    for line in text.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 9:
            continue
        try:
            ts = parts[0]
            f, by, bz, n, v, _p_omni, kp, _dst = (float(x) for x in parts[1:9])
        except ValueError:
            continue
        nv = np.nan if n >= _FILL["N"] else n
        vv = np.nan if v >= _FILL["V"] else v
        F.append(np.nan if f >= _FILL["F"] else f)
        By.append(np.nan if abs(by) >= _FILL["B"] else by)
        Bz.append(np.nan if abs(bz) >= _FILL["B"] else bz)
        N.append(nv)
        V.append(vv)
        # Recompute P from N,V with the shared constant (NOT OMNI's Pressure1800)
        # so training matches the live/synthetic paths exactly. NaN propagates.
        P.append(_P_CONST * nv * vv * vv)
        KP.append(np.nan if kp >= _FILL["KP"] else kp)
        times.append(ts)
    if not times:
        return np.empty((0,), object), np.empty((0, 8)), np.empty((0,))
    arr = lambda L: np.asarray(L, float)
    channels = _derive(arr(F), arr(By), arr(Bz), arr(N), arr(V), arr(P))
    return np.asarray(times, object), channels, arr(KP)


# ----------------------------------------------------------------------
# Leakage-free chronological split (with a gap to kill autocorrelation)
# ----------------------------------------------------------------------
def _chrono_split(X, y, t, cfg):
    s = cfg["storm"]["split"]
    n = len(y)
    gap = int(cfg["storm"]["split"].get("gap_days", 5) * 24 / int(cfg["storm"].get("sample_stride_h", 3)))
    i_tr = int(n * s["train"])
    i_va = int(n * (s["train"] + s["val"]))
    tr = slice(0, max(0, i_tr - gap))
    va = slice(i_tr, max(i_tr, i_va - gap))
    te = slice(i_va, n)
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def load_split(cfg: dict | None = None):
    """Fetch OMNI, engineer features, split chronologically. Raises if OMNI is
    unreachable (callers may fall back to the synthetic split)."""
    cfg = cfg or load_config()
    years = int(cfg["storm"]["train_years"])
    stop = datetime.now(timezone.utc).date()
    start = stop - timedelta(days=int(years * 365.25))
    res = sources.get_omni(cfg, f"{start}T00:00:00Z", f"{stop}T00:00:00Z")
    if not res.ok or not res.data:
        raise RuntimeError(f"OMNI fetch failed: {res.error or 'no data'}")
    times, channels, kp = parse_omni_csv(res.data)
    if len(kp) < 1000:
        raise RuntimeError(f"OMNI returned too few rows ({len(kp)})")
    X3d, y, t = _windows(channels, kp, times, cfg)
    X = dataio.build_matrix(X3d, _stats_cfg(cfg))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _chrono_split(X, y, t, cfg)
    return (Xtr, ytr), (Xva, yva), (Xte, yte), {"n_total": len(y), "pos_rate": float(np.mean(y))}


def _stats_cfg(cfg):
    """Tiny shim so data.build_matrix uses the STORM summary stats + channels."""
    return {"features": {"sharp": CHANNELS, "summary_stats": cfg["storm"]["summary_stats"]}}


# ----------------------------------------------------------------------
# Synthetic fixture (offline) — storms genuinely follow strong southward Bz +
# high speed, so there is real, honest signal to learn (evaluated held-out).
# ----------------------------------------------------------------------
def make_synthetic(n_hours: int, *, seed: int, cfg: dict):
    rng = np.random.default_rng(seed)
    obs = int(cfg["storm"]["observation_window_h"])
    # Smooth-ish hourly solar wind with occasional southward-Bz "CME" intervals.
    Bz = rng.normal(0, 3, n_hours)
    By = rng.normal(0, 3, n_hours)
    V = np.abs(rng.normal(420, 60, n_hours))
    N = np.abs(rng.normal(6, 3, n_hours))
    n_events = max(1, n_hours // 240)
    for _ in range(n_events):
        st = rng.integers(0, n_hours - 30)
        dur = int(rng.integers(8, 28))
        Bz[st:st + dur] -= rng.uniform(6, 22)                  # strong southward IMF
        V[st:st + dur] += rng.uniform(80, 350)                 # fast stream
        N[st:st + dur] += rng.uniform(3, 15)
    Bmag = np.sqrt(By ** 2 + Bz ** 2) + np.abs(rng.normal(2, 1, n_hours))
    P = _P_CONST * N * V ** 2
    channels = _derive(Bmag, By, Bz, N, V, P)
    # Kp surrogate driven by the Newell coupling (channel 7), smoothed + noisy.
    newell = channels[:, 7]
    drive = np.convolve(newell, np.ones(6) / 6, mode="same")
    kp = np.clip((drive / (np.nanmedian(drive) + 1e-9)) * 18 + rng.normal(0, 6, n_hours), 0, 90)
    times = np.asarray([f"syn{i}" for i in range(n_hours)], object)
    return channels, kp, times


def load_synthetic_split(cfg: dict | None = None):
    cfg = cfg or load_config()
    ch, kp, t = make_synthetic(20000, seed=7, cfg=cfg)
    X3d, y, tt = _windows(ch, kp, t, cfg)
    X = dataio.build_matrix(X3d, _stats_cfg(cfg))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _chrono_split(X, y, tt, cfg)
    return (Xtr, ytr), (Xva, yva), (Xte, yte), {"n_total": len(y), "pos_rate": float(np.mean(y))}


# ----------------------------------------------------------------------
# Live inference window — reconstruct the model's channels from L1 feeds
# ----------------------------------------------------------------------
def _parse_l1(rows, idx_by_name):
    """SWPC product (header row + rows) -> {col: np.array} + parsed times."""
    if not isinstance(rows, list) or len(rows) < 2:
        return None
    header = [str(h).lower() for h in rows[0]]
    cols = {name: header.index(col) for name, col in idx_by_name.items() if col in header}
    if "time" not in cols:
        return None
    times, data = [], {k: [] for k in cols if k != "time"}
    for r in rows[1:]:
        try:
            ts = datetime.fromisoformat(str(r[cols["time"]]).replace("Z", "").strip())
        except (ValueError, IndexError):
            continue
        times.append(ts.replace(tzinfo=timezone.utc))
        for k in data:
            try:
                v = float(r[cols[k]])
            except (ValueError, IndexError, TypeError):
                v = np.nan
            data[k].append(v)
    return {"t": times, **{k: np.asarray(v, float) for k, v in data.items()}}


def _hourly(times, vals, hour_starts):
    """Average a 1-min series into the given hourly bins (NaN if a bin is empty)."""
    out = np.full(len(hour_starts), np.nan)
    if not times:
        return out
    tarr = np.asarray([t.timestamp() for t in times])
    for i, hs in enumerate(hour_starts):
        lo, hi = hs.timestamp(), (hs + timedelta(hours=1)).timestamp()
        m = (tarr >= lo) & (tarr < hi) & np.isfinite(vals)
        if m.any():
            out[i] = float(np.nanmean(vals[m]))
    return out


def live_feature_vector(cfg: dict | None = None):
    """Build the model's 56-feature vector from live L1 mag + plasma. Returns
    (vector(1,F) or None, summary dict, status, notes)."""
    cfg = cfg or load_config()
    obs = int(cfg["storm"]["observation_window_h"])
    mag = sources.get_l1_mag(cfg)
    pls = sources.get_l1_plasma(cfg)
    notes = list(mag.notes) + list(pls.notes)
    if not mag.ok or not pls.ok:
        return None, {}, "unavailable", notes
    m = _parse_l1(mag.data, {"time": "time_tag", "by": "by_gsm", "bz": "bz_gsm", "bt": "bt"})
    p = _parse_l1(pls.data, {"time": "time_tag", "density": "density", "speed": "speed"})
    if not m or not p or not m.get("t") or not p.get("t"):
        return None, {}, "unavailable", notes
    # Anchor to the last COMPLETED hour (the current hour is only partially filled,
    # which would make the model's heavily-weighted *_last features a noisy partial
    # average vs OMNI's full-hour training values).
    latest = min(max(m["t"]), max(p["t"]))
    anchor = latest.replace(minute=0, second=0, microsecond=0)   # start of the current hour
    hour_starts = [anchor - timedelta(hours=obs - i) for i in range(obs)]   # obs full hours
    col = lambda d, k: d[k] if k in d else np.full(len(d["t"]), np.nan)      # missing col -> NaN
    By = _hourly(m["t"], col(m, "by"), hour_starts)
    Bz = _hourly(m["t"], col(m, "bz"), hour_starts)
    Bmag = _hourly(m["t"], col(m, "bt"), hour_starts)
    N = _hourly(p["t"], col(p, "density"), hour_starts)
    V = _hourly(p["t"], col(p, "speed"), hour_starts)
    P = _P_CONST * N * V ** 2
    channels = _derive(Bmag, By, Bz, N, V, P)                   # (obs, 8)
    X = dataio.build_matrix(channels[None, :, :], _stats_cfg(cfg))
    status = "live" if (mag.status == "live" and pls.status == "live") else "cached"

    def _last(a):
        a = a[np.isfinite(a)]
        return round(float(a[-1]), 2) if a.size else None
    summary = {"bz_nT": _last(Bz), "bt_nT": _last(Bmag), "speed_kms": _last(V),
               "density_cc": _last(N), "newell": round(float(np.nanmean(channels[:, 7])), 1)
               if np.isfinite(channels[:, 7]).any() else None,
               "window_h": obs, "observed_end": anchor.isoformat()}
    return X, summary, status, notes
