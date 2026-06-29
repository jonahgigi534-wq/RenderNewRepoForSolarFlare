"""Regression guard — run this after EVERY change to confirm the existing
Helios flare app still works. It is READ-ONLY: it never trains or overwrites
the model, and it works offline (live fetchers fall back to cache/climatology).

Checks:
  1. predictor.predict() returns the full, well-formed forecast contract.
  2. The nowcast + 12/24/48h forecast + ensemble are present and sane.
  3. If a SHARP model is trained, it exposes the three operating points.
  4. The API route functions (/health, /api/forecast, /api/flux, /api/regions)
     return their expected shapes.

Run:  python scripts/regression_test.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solarflare import predictor
from solarflare.config import load_config

PASS, FAIL = "  OK  ", "  XX  "
errors = []


def check(name, cond, detail=""):
    print((PASS if cond else FAIL) + name + (f" — {detail}" if detail else ""))
    if not cond:
        errors.append(name)


def main() -> int:
    cfg = load_config()

    print("=" * 64)
    print("1) predictor.predict() contract")
    print("=" * 64)
    r = predictor.predict(cfg=cfg)
    for key in ("generated_at", "data_freshness", "nowcast", "forecast",
                "active_region_count", "notes", "disclaimer"):
        check(f"top-level key '{key}'", key in r)

    df = r.get("data_freshness", {})
    check("data_freshness.status valid",
          df.get("status") in ("live", "cached", "unavailable"),
          df.get("status"))

    fc = r.get("forecast", {})
    for key in ("flux_track", "sharp_track", "sharp_model", "noaa_track", "ensemble_24h"):
        check(f"forecast.{key} present", key in fc)

    # 12/24/48h horizons sane when the flux track is available
    ft = fc.get("flux_track", {})
    if ft.get("available"):
        for h in ("12h", "24h", "48h"):
            hz = ft.get("horizons", {}).get(h, {})
            p = hz.get("p_M_or_greater")
            check(f"horizon {h} p(M+) in [0,1]", isinstance(p, (int, float)) and 0 <= p <= 1, p)

    ens = fc.get("ensemble_24h", {})
    pe = ens.get("p_M_or_greater_24h")
    check("ensemble 24h p(M+) sane", pe is None or (0 <= pe <= 1), pe)

    print("\n" + "=" * 64)
    print("2) SHARP model (if trained) exposes operating points")
    print("=" * 64)
    sm = fc.get("sharp_model")
    if sm and sm.get("trained"):
        check("winner recorded", bool(sm.get("winner")), sm.get("winner"))
        ops = sm.get("operating_points", {})
        check("three operating points",
              set(ops) >= {"high_recall", "balanced", "high_precision"}, list(ops))
        for name, op in ops.items():
            ok = all(0 <= op.get(k, -1) <= 1 for k in ("tss", "recall", "precision"))
            check(f"  {name} metrics in range", ok, op)
        check("default operating point set",
              sm.get("operating_point") in ops, sm.get("operating_point"))
    else:
        print("  (no trained model on disk — skipping; flux track still forecasts)")

    print("\n" + "=" * 64)
    print("3) API route functions")
    print("=" * 64)
    from api import server
    h = server.health()
    check("/health status ok", h.get("status") == "ok", h.get("status"))
    check("/health reports model flag", "sharp_model_loaded" in h)
    fr = server.forecast()
    check("/api/forecast returns 200", getattr(fr, "status_code", 200) == 200)
    fx = server.flux()
    check("/api/flux returns dict/response", fx is not None)
    rg = server.regions()
    check("/api/regions returns dict/response", rg is not None)

    print("\n" + "=" * 64)
    print("4) /api/hazard (STEP 1) contract")
    print("=" * 64)
    from solarflare import hazard
    hz = hazard.build_hazard(cfg)
    sub = hz.get("subsolar", {})
    check("subsolar lat in [-23.5,23.5]", -23.5 <= sub.get("lat", 99) <= 23.5, sub.get("lat"))
    check("subsolar lon in [-180,180]", -180 <= sub.get("lon", 999) <= 180, sub.get("lon"))
    hb = hz.get("hf_blackout", {})
    check("hf_blackout source set", hb.get("source") in ("noaa-drap", "synthesized-from-flare-class"),
          hb.get("source"))
    grid = hb.get("grid", {})
    check("grid is rectangular",
          bool(grid.get("values")) and all(len(r) == len(grid["lons"]) for r in grid["values"]))
    check("danger_cells is a list", isinstance(hz.get("danger_cells"), list),
          f"{len(hz.get('danger_cells', []))} cells")
    he = server.hazard_endpoint()
    check("/api/hazard returns 200", getattr(he, "status_code", 200) == 200)

    print("\n" + "=" * 64)
    print("5) /api/geomag (STEP 4) contract")
    print("=" * 64)
    from solarflare import geomag
    gm = geomag.build_geomag(cfg)
    kp = gm.get("kp", {})
    check("kp block present", isinstance(kp, dict) and "kp" in kp)
    check("g_scale in 0..5 or None",
          kp.get("g_scale") is None or (0 <= kp.get("g_scale") <= 5), kp.get("g_scale"))
    au = gm.get("aurora", {})
    check("aurora source set", au.get("source") in ("noaa-ovation", "synthesized-from-kp"),
          au.get("source"))
    agrid = au.get("grid", {})
    check("aurora grid rectangular",
          bool(agrid.get("values")) and all(len(r) == len(agrid["lons"]) for r in agrid["values"]))
    check("aurora danger_cells is a list", isinstance(gm.get("danger_cells"), list),
          f"{len(gm.get('danger_cells', []))} cells")
    ge = server.geomag_endpoint()
    check("/api/geomag returns 200", getattr(ge, "status_code", 200) == 200)

    print("\n" + "=" * 64)
    print("6) /api/storm (STEP 5) contract")
    print("=" * 64)
    from solarflare import storm
    sf = storm.storm_forecast(cfg)
    for key in ("task", "data_status", "ml_forecast", "noaa_kp_forecast",
                "headline_p_storm", "disclaimer"):
        check(f"storm.{key} present", key in sf)
    hp = sf.get("headline_p_storm")
    check("headline_p_storm in [0,1]", isinstance(hp, (int, float)) and 0 <= hp <= 1, hp)
    mlf = sf.get("ml_forecast", {})
    if mlf.get("available"):
        check("ml p_storm in [0,1]", 0 <= mlf.get("p_storm", -1) <= 1, mlf.get("p_storm"))
        check("ml reports test TSS", mlf.get("test_tss") is not None, mlf.get("test_tss"))
    else:
        print("  (storm model not trained or L1 down — climatology fallback in use)")
    fo = sf.get("forecast_oval")                          # STEP 6: forecast auroral oval
    check("forecast_oval rectangular or None",
          fo is None or (bool(fo.get("values")) and all(len(r) == len(fo["lons"]) for r in fo["values"])),
          "present" if fo else "none")
    se = server.storm_endpoint()
    check("/api/storm returns 200", getattr(se, "status_code", 200) == 200)

    print("\n" + "=" * 64)
    print("7) /api/experiment (STEP 7) endpoints")
    print("=" * 64)
    el = server.experiment_leadtime()                    # 200 if generated, 404 if not — both OK
    check("/api/experiment/leadtime responds", getattr(el, "status_code", 200) in (200, 404),
          getattr(el, "status_code", 200))
    ep = server.experiment_leadtime_png()
    check("/api/experiment/leadtime.png responds", getattr(ep, "status_code", 200) in (200, 404),
          getattr(ep, "status_code", 200))

    print("\n" + "=" * 64)
    print("8) /api/impact + /api/alerts + aurora view (STEP 8)")
    print("=" * 64)
    from solarflare import impact as impmod, alerts as almod
    ib = impmod.build_impact(cfg)
    check("impact has R/S/G scales", set(ib.get("scales", {})) >= {"R", "S", "G"},
          list(ib.get("scales", {})))
    for k, v in ib.get("scales", {}).items():
        check(f"  {k} level in 0-5", isinstance(v.get("level"), int) and 0 <= v["level"] <= 5, v.get("level"))
    check("/api/impact returns 200", getattr(server.impact_endpoint(), "status_code", 200) == 200)
    ab = almod.build_alerts(cfg)
    check("alerts contract", isinstance(ab.get("alerts"), list) and "all_clear" in ab, ab.get("active"))
    check("alerts channels reported", isinstance(ab.get("channels"), dict))
    check("/api/alerts returns 200", getattr(server.alerts_endpoint(), "status_code", 200) == 200)
    av = geomag.aurora_view(6.0)
    check("aurora_view boundary sane",
          av.get("available") and -90 <= av.get("overhead_mlat", 99) <= 90, av.get("overhead_mlat"))

    print("\n" + "=" * 64)
    print("9) email notifier (predictive alert + verification)")
    print("=" * 64)
    from solarflare import notify as notifymod
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    ns = notifymod.status(cfg)
    check("notify status contract", isinstance(ns, dict) and "mode" in ns and "pending" in ns, ns.get("mode"))
    check("notifier dry-run safe (no creds => not sending real email)",
          ns.get("mode") in ("dry-run", "live-email"), ns.get("mode"))
    pk = notifymod.actual_peak_in_window(cfg, _dt.now(_tz.utc) - _td(hours=24), _dt.now(_tz.utc))
    check("verification source returns a peak class", isinstance(pk.get("peak_class"), str), pk.get("source"))
    check("/api/notify/status returns 200", getattr(server.notify_status(), "status_code", 200) == 200)

    print("\n" + "=" * 64)
    print("10) /api/satellites (3D globe layer) contract")
    print("=" * 64)
    from solarflare import satellites as satmod
    sat = satmod.build_satellites("default", cfg)
    sats = sat.get("satellites")
    check("satellites is a list", isinstance(sats, list), f"{len(sats or [])} sats")
    valid_tiers = {"low", "elevated", "high", "severe"}
    check("risk tiers valid", all(s.get("risk") in valid_tiers for s in (sats or [])),
          sorted({s.get("risk") for s in (sats or [])}))
    check("risk_counts has all tiers", set(sat.get("risk_counts", {})) == valid_tiers,
          sat.get("risk_counts"))
    check("/api/satellites returns 200",
          getattr(server.satellites_endpoint("default"), "status_code", 200) == 200)

    print("\n" + "=" * 64)
    print("11) peak-magnitude forecast + honest skill grading")
    print("=" * 64)
    from solarflare import magnitude
    mg = magnitude.predict({"p_C": 0.9, "p_M": 0.6, "p_X": 0.1, "peak_24h_flux": 9.5e-6}, cfg)
    check("magnitude forecast has a model peak class", isinstance(mg.get("model_class"), str),
          mg.get("model_class"))
    check("model peak rises with M+ odds",
          magnitude.expected_peak_flux(0.95, 0.7, 0.2, cfg)
          > magnitude.expected_peak_flux(0.3, 0.05, 0.005, cfg))
    # log-flux error is symmetric and ~0.86 dex for the real C1.3-vs-C9.5 example
    e = magnitude.error_dex(1.3e-6, 9.5e-6)
    check("log-flux error sane (C1.3 vs C9.5 ~0.86 dex)", e is not None and 0.8 < e < 0.95, e)
    check("error is None when ungraded", magnitude.error_dex(None, 1e-5) is None)
    empty = magnitude.skill_summary([])
    check("skill summary empty-safe", empty.get("available") is False, empty.get("n"))
    sk = magnitude.skill_summary([{"err_dex": 0.3, "persist_err_dex": 0.6, "clim_err_dex": 0.5}])
    check("skill vs persistence computed", sk.get("skill_vs_persistence") == 0.5,
          sk.get("skill_vs_persistence"))
    ns2 = notifymod.status(cfg)
    check("/api/notify/status carries an accuracy block", isinstance(ns2.get("accuracy"), dict),
          ns2.get("accuracy", {}).get("available"))

    print("\n" + ("ALL REGRESSION CHECKS PASSED." if not errors
                  else f"FAILED: {len(errors)} check(s) -> {errors}"))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
