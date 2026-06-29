"""Tiny config loader. Reads config.yaml from the project root and exposes it
as a plain dict with convenient attribute-style access (cfg["a"]["b"])."""
from __future__ import annotations

import os
from functools import lru_cache

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict:
    """Load and cache config.yaml. Pass a path to override the default."""
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # Resolve all paths relative to the project root so the tool works from
    # any working directory.
    for key, val in cfg.get("paths", {}).items():
        if not os.path.isabs(val):
            cfg["paths"][key] = os.path.join(PROJECT_ROOT, val)
    cfg["_project_root"] = PROJECT_ROOT
    return cfg
