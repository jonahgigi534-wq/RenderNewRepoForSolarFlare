"""FastAPI backend serving the solar-flare forecast.

Run:
    uvicorn api.server:app --reload --port 8000
    # or: python -m api.server

Thin routing/wiring layer: each route delegates to a module in the `solarflare`
package and wraps the result in a JSON / file / CSV response. See the API table
in README.md (or the @app.get decorators below) for the full endpoint list.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from fastapi import Query

from solarflare import (alerts, geomag, hazard, impact, notify, predictor,
                        satellites, sources, storm)
from solarflare.config import load_config

cfg = load_config()
log = logging.getLogger("helios.server")


@asynccontextmanager
async def _lifespan(app):
    # Start the live email-notifier loop while the server runs (dry-run unless SMTP
    # is configured). Only fires under uvicorn, not on plain import (tests are safe).
    try:
        notify.start_background(cfg)
    except Exception as exc:                                   # noqa: BLE001 (never block startup)
        log.warning("notifier background loop failed to start: %s", exc)
    yield


app = FastAPI(title="Solar Flare Predictor", version="0.1.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

FRONTEND = os.path.join(cfg["_project_root"], "frontend", "index.html")


@app.get("/health")
def health():
    model = predictor.load_sharp_model(cfg)
    return {
        "status": "ok",
        "sharp_model_loaded": model is not None,
        "sharp_test_tss": (model or {}).get("metrics", {}).get("tss") if model else None,
    }


@app.get("/api/forecast")
def forecast():
    # SHARP features are not supplied in the always-on web path (live SHARP
    # needs JSOC); the flux track + ensemble still produce a full forecast.
    return JSONResponse(predictor.predict(cfg=cfg))


@app.get("/api/flux")
def flux():
    res = sources.get_xray_flux(cfg)
    if not res.ok:
        return JSONResponse({"ok": False, "error": res.error}, status_code=503)
    # Downsample to keep the payload light for the chart (~every 5 min).
    rows = res.data[::5]
    return {
        "ok": True,
        "status": res.status,
        "series": [{"t": r["time_tag"], "flux": r["flux"]} for r in rows if r.get("flux")],
    }


@app.get("/api/hazard")
def hazard_endpoint():
    """Dayside HF-radio-blackout footprint (D-RAP) + subsolar point. STEP 1."""
    return JSONResponse(hazard.build_hazard(cfg))


@app.get("/api/geomag")
def geomag_endpoint():
    """High-latitude geomagnetic/auroral hazard (OVATION oval + Kp). STEP 4."""
    return JSONResponse(geomag.build_geomag(cfg))


@app.get("/api/storm")
def storm_endpoint():
    """Geomagnetic-storm forecast from the L1 solar wind (OMNI-trained ML). STEP 5."""
    return JSONResponse(storm.storm_forecast(cfg))


@app.get("/api/sharp_live")
def sharp_live_endpoint(at: str = Query("", description="optional ISO-8601 UTC time for a historical demo; blank = now")):
    """Live SHARP ML flare forecast: P(M-class+ in 24h) per active region from JSOC
    magnetic-field data, using our own JSOC-trained model (saved preprocessing)."""
    from datetime import datetime, timezone
    from solarflare import sharp_live as sl
    at_time = None
    if at:
        try:
            at_time = datetime.fromisoformat(at).replace(tzinfo=timezone.utc)
        except ValueError:
            at_time = None
    return JSONResponse(sl.predict_live(cfg, at_time=at_time))


@app.get("/api/scorecard")
def scorecard_endpoint():
    """Model Skill Scorecard — benchmark vs. real operational TSS (the research result;
    built by `python -m solarflare.scorecard`)."""
    p = os.path.join(cfg["_project_root"], "skill_scorecard.json")
    if os.path.exists(p):
        return FileResponse(p, media_type="application/json", headers={"Cache-Control": "no-store"})
    return JSONResponse({"available": False, "note": "run: python -m solarflare.scorecard"}, status_code=404)


@app.get("/api/impact")
def impact_endpoint():
    """Plain-language R/S/G space-weather impact statements (STEP 8)."""
    return JSONResponse(impact.build_impact(cfg))


@app.get("/api/alerts")
def alerts_endpoint():
    """Threshold alerts — log/webhook (email off, no secrets). STEP 8."""
    return JSONResponse(alerts.build_alerts(cfg))


@app.get("/api/notify/status")
def notify_status():
    """Email-notifier state — mode (dry-run/live), thresholds, pending + recent predictions."""
    return JSONResponse(notify.status(cfg))


@app.get("/api/notify/history.csv")
def notify_history_csv():
    """Full prediction history as a CSV spreadsheet (forecast vs actual outcome)."""
    return Response(content=notify.history_csv_string(cfg), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=prediction_history.csv"})


EXPERIMENT_DIR = os.path.join(cfg["_project_root"], "solarflare", "experiments", "results")


@app.get("/api/experiment/leadtime")
def experiment_leadtime():
    """Lead-time-vs-skill results (STEP 7). LIVE: recomputes in the background on a
    rolling window (ending today) whenever the cached result goes stale."""
    from solarflare.experiments import leadtime_skill as lt
    ttl = cfg["experiment"].get("live_refresh_hours", 6)
    refreshing = lt.ensure_fresh(cfg, ttl)
    data = lt.load_result()
    if data is None:
        return JSONResponse({"available": False, "refreshing": refreshing,
                             "note": "generating — refresh shortly"}, status_code=202)
    age = lt.result_age_hours()
    data["live_meta"] = {"age_hours": round(age, 2) if age is not None else None,
                         "refreshing": refreshing, "refresh_after_hours": ttl}
    return JSONResponse(data)


@app.get("/api/experiment/leadtime.png")
def experiment_leadtime_png():
    """The 'skill vs required lead time' figure (STEP 7)."""
    p = os.path.join(EXPERIMENT_DIR, "leadtime_skill.png")
    if os.path.exists(p):
        return FileResponse(p, media_type="image/png", headers={"Cache-Control": "no-store"})
    return JSONResponse({"error": "figure not generated yet"}, status_code=404)


@app.get("/api/satellites")
def satellites_endpoint(scope: str = Query("default", pattern="^(default|all)$")):
    """Satellites + altitude-band flare-risk tiers. scope=default|all."""
    return JSONResponse(satellites.build_satellites(scope, cfg))


@app.get("/api/regions")
def regions():
    res = sources.get_solar_regions(cfg)
    if not res.ok or not isinstance(res.data, list):
        return JSONResponse({"ok": False, "regions": []}, status_code=503)
    keep = ("region", "spot_class", "mag_class", "number_spots", "area",
            "c_flare_probability", "m_flare_probability", "x_flare_probability")
    rows = [{k: r.get(k) for k in keep} for r in res.data]
    return {
        "ok": True,
        "status": res.status,
        "observed_date": res.data[0].get("observed_date") if res.data else None,
        "regions": rows,
    }


@app.get("/")
def index():
    if os.path.exists(FRONTEND):
        # No-cache so a redeployed page is always picked up by the browser.
        return FileResponse(FRONTEND, headers={"Cache-Control": "no-store"})
    return JSONResponse({"error": "frontend/index.html not found"}, status_code=404)


def main():
    import uvicorn
    uvicorn.run("api.server:app", host=cfg["api"]["host"], port=cfg["api"]["port"])


if __name__ == "__main__":
    main()
