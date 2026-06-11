"""Configuration loading for AdRank-ML.

A single YAML file (``config/config.yaml``) is the source of truth for the whole
pipeline. ``load_config`` returns a light dot-accessible wrapper and resolves the
active ``scale`` profile (``demo`` vs ``prod``) into a flat ``cfg.scale`` block so
downstream code never has to branch on the profile name.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels up from this file (src/adrank/config.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


class Config(dict):
    """A dict whose keys are also accessible as attributes (recursively)."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _resolve_paths(cfg: Config) -> None:
    """Make all configured paths absolute, anchored at the repo root."""
    paths = cfg.get("paths", {})
    for key, rel in list(paths.items()):
        p = Path(rel)
        if not p.is_absolute():
            p = REPO_ROOT / p
        paths[key] = str(p)
        p.mkdir(parents=True, exist_ok=True)
    cfg["paths"] = paths


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load the YAML config and flatten the active scale profile.

    The chosen profile (``scale.profile``) is copied to ``cfg.scale`` so callers
    can read ``cfg.scale.n_campaigns`` etc. without knowing which profile is live.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)

    cfg = Config(raw)

    scale_block = cfg["scale"]
    profile = scale_block.get("profile", "demo")
    if profile not in scale_block:
        raise ValueError(
            f"scale.profile={profile!r} has no matching block under `scale`."
        )
    resolved = dict(scale_block[profile])
    resolved["profile"] = profile
    cfg["scale"] = Config(resolved)

    _resolve_paths(cfg)
    return cfg


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import json

    c = load_config()
    print(json.dumps(c, indent=2, default=str))
