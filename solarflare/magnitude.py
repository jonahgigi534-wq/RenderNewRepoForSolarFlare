"""Peak-magnitude forecast + HONEST skill grading.

Forecasts a SPECIFIC next-24h peak flare magnitude and grades it the right way:
not "did the class letter match" (the trap — a wide tolerance around the modal
class scores high with zero skill) but the error in **log10-flux** (orders of
magnitude, because flux is logarithmic), turned into a **skill score** against two
dumb baselines. "Accurate" therefore means *beat the baselines*, not "landed in
the right band".

  * model        : a transparent log-space probability-weighted expectation over
                   the flare model's OWN class probabilities P(C+)/P(M+)/P(X).
  * persistence  : tomorrow's peak ~= today's 24h peak flux (the baseline to beat).
  * climatology  : a fixed median daily peak (the zero-skill anchor).

  skill = 1 - MAE_model / MAE_baseline   (>0 => genuinely better than the baseline)

This is a transparent STATISTICAL estimate, NOT a trained magnitude regressor: the
SWAN-SF labels shipped here are binary M+, so there are no continuous peak-flux
labels to train one — pretending otherwise would be the "cheating" the project
forbids. Honest by construction. Nothing here raises to the caller.
"""
from __future__ import annotations

import math

from . import labels
from .config import load_config

# Quiet days have ~no flare (peak flux ~0); floor both sides at the A/B boundary
# so log10 is finite and a calm day maps to a sensible background, not -inf.
FLUX_FLOOR = 1e-7


def _mcfg(cfg: dict) -> dict:
    return cfg.get("magnitude", {}) or {}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _clip01(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def band_reps(cfg: dict) -> dict:
    """Representative peak flux (W/m^2) per operational band — geometric midpoints,
    so each sits centrally in LOG space (the right average for a log quantity)."""
    r = _mcfg(cfg).get("band_rep_flux", {}) or {}
    return {
        "no-flare": _f(r.get("no-flare")) or 1.5e-6,   # typical low-C/B quiet-day peak
        "C":        _f(r.get("C")) or 7.1e-6,          # geo-mid C5..M1
        "M":        _f(r.get("M")) or 3.16e-5,         # geo-mid M1..X1
        "X":        _f(r.get("X")) or 3.16e-4,         # geo-mid X1..X10
    }


def climatology_flux(cfg: dict) -> float:
    """The zero-skill constant: a long-run median daily peak (~C3 by default)."""
    return _f(_mcfg(cfg).get("climatology_peak_flux")) or 3.0e-6


def expected_peak_flux(p_c, p_m, p_x, cfg: dict | None = None) -> float | None:
    """Log-space probability-weighted expected peak flux from the forecast's
    CUMULATIVE class probabilities P(C+), P(M+), P(X). Monotone and transparent:
    higher big-flare odds -> higher expected peak. Returns None if all inputs are
    missing (so the caller can fall back to persistence)."""
    cfg = cfg or load_config()
    pc, pm, px = _f(p_c), _f(p_m), _f(p_x)
    if pc is None and pm is None and px is None:
        return None
    pc, pm, px = (pc or 0.0), (pm or 0.0), (px or 0.0)
    # Cumulative -> per-band probability mass, clamped to [0, 1].
    w = {"X": _clip01(px), "M": _clip01(pm - px),
         "C": _clip01(pc - pm), "no-flare": _clip01(1.0 - pc)}
    total = sum(w.values()) or 1.0
    reps = band_reps(cfg)
    log_e = sum(mass * math.log10(reps[band]) for band, mass in w.items()) / total
    return 10 ** log_e


def predict(state: dict, cfg: dict | None = None) -> dict:
    """Build the magnitude forecast + its two baselines from a forecast `state`
    (notify._forecast_state, which carries p_C/p_M/p_X and peak_24h_flux)."""
    cfg = cfg or load_config()
    try:
        model = expected_peak_flux(state.get("p_C"), state.get("p_M"),
                                   state.get("p_X"), cfg)
    except Exception:                                  # noqa: BLE001 (never raise)
        model = None
    persist = _f(state.get("peak_24h_flux"))           # persistence: ~ today's 24h peak
    if model is None:                                  # no forecast probs -> lean on persistence
        model = persist
    clim = climatology_flux(cfg)
    return {
        "model_flux": model,
        "model_class": labels.flux_to_letter(model) if model else None,
        "persistence_flux": persist,
        "persistence_class": labels.flux_to_letter(persist) if persist else None,
        "climatology_flux": clim,
        "climatology_class": labels.flux_to_letter(clim),
        "method": "log-probability-weighted expectation (transparent estimate, "
                  "not a trained regressor)",
    }


def error_dex(pred_flux, actual_flux) -> float | None:
    """Absolute error in orders of magnitude (|Δlog10 flux|). Floored so a quiet
    day (flux ~ 0) maps to a finite background instead of -inf. None if either
    side is missing (e.g. a row logged before this feature existed)."""
    p, a = _f(pred_flux), _f(actual_flux)
    if p is None or a is None:
        return None
    return abs(math.log10(max(p, FLUX_FLOOR)) - math.log10(max(a, FLUX_FLOOR)))


def _verdict(s_persist, s_clim) -> str:
    if s_persist is None and s_clim is None:
        return "Not enough verified forecasts to judge skill yet."
    sp = -1.0 if s_persist is None else s_persist
    sc = -1.0 if s_clim is None else s_clim
    if sp > 0.05 and sc > 0.05:
        return "Genuinely skilful — beats both persistence and climatology."
    if sc > 0.05:
        return "Beats climatology but not persistence — limited skill."
    if sp > 0.05:
        return "Beats persistence but not climatology — mixed."
    return "No skill yet — not beating the naive baselines (this is the honest result)."


def skill_summary(rows) -> dict:
    """Aggregate honest skill from graded rows (dicts with err_dex / persist_err_dex
    / clim_err_dex; None-valued entries are skipped). skill = 1 - MAE_model/MAE_base."""
    graded = [r for r in rows if r.get("err_dex") is not None]
    n = len(graded)
    if not n:
        return {"available": False, "n": 0,
                "note": "No verified magnitude forecasts yet — accuracy builds over "
                        "time (one verified forecast per day)."}

    def _mae(key):
        vals = [r[key] for r in graded if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    mae_m, mae_p, mae_c = _mae("err_dex"), _mae("persist_err_dex"), _mae("clim_err_dex")

    def _skill(base):
        if mae_m is None or not base:
            return None
        return round(1.0 - mae_m / base, 3)

    s_persist, s_clim = _skill(mae_p), _skill(mae_c)
    return {
        "available": True,
        "n": n,
        "mae_model_dex": round(mae_m, 3) if mae_m is not None else None,
        "mae_persistence_dex": round(mae_p, 3) if mae_p is not None else None,
        "mae_climatology_dex": round(mae_c, 3) if mae_c is not None else None,
        "skill_vs_persistence": s_persist,
        "skill_vs_climatology": s_clim,
        "verdict": _verdict(s_persist, s_clim),
        "scoring": "error = |Δlog10 peak-flux| (dex); skill = 1 - MAE_model/MAE_baseline",
    }
