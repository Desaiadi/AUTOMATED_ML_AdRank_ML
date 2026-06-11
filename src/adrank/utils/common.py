"""Shared helpers: logging, deterministic seeding, and a timing context manager.

The timing helper underpins the project's "reduced training+scoring time from
62 -> 44 minutes" claim: every pipeline stage is wrapped in ``timed(...)`` and the
durations are collected into ``data/reports/timing.json`` for the backtest report.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed: int) -> np.random.Generator:
    """Seed all relevant RNGs and return a NumPy Generator for explicit use."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


# --- timing instrumentation -------------------------------------------------

_TIMINGS: dict[str, float] = {}


@contextmanager
def timed(stage: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Time a stage, accumulate it in the module-level registry, and log it."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _TIMINGS[stage] = _TIMINGS.get(stage, 0.0) + elapsed
        msg = f"[timing] {stage}: {elapsed:.2f}s"
        (logger or get_logger("adrank.timing")).info(msg)


def get_timings() -> dict[str, float]:
    return dict(_TIMINGS)


def reset_timings() -> None:
    _TIMINGS.clear()


def dump_timings(path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_TIMINGS, fh, indent=2)


def save_json(obj: object, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
