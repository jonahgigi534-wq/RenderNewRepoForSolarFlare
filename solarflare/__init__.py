"""Solar Flare Predictor — a two-track (SHARP-ML + GOES-flux) forecasting system.

Modules
-------
config      : load the central config.yaml
labels      : flare-class taxonomy (GOES flux <-> category, NOAA letter codes)
data        : load/engineer the Cleaned-SWANSF .pkl dataset (+ synthetic fixture)
train       : fit and persist the SHARP model (.joblib + .pkl)
evaluate    : flare-forecasting skill scores (TSS, HSS, recall, ...)
sources     : resilient live data fetchers (NOAA / NASA) with caching + backups
nowcast     : exact "what is flaring right now" + X-class warnings from flux
fluxmodel   : statistical near-term forecaster from recent GOES flux
predictor   : orchestrates nowcast + forecasts into one clean prediction
"""
from .config import load_config

__all__ = ["load_config"]
__version__ = "0.1.0"
