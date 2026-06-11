#!/usr/bin/env python
"""Run the full AdRank-ML pipeline end to end.

Thin wrapper around the CLI so the project has an obvious entrypoint:

    python scripts/run_pipeline.py                 # demo profile (default)
    python scripts/run_pipeline.py --profile prod  # 100M-impression target
    python scripts/run_pipeline.py --impressions 1000000

Equivalent to ``python -m adrank.cli all``.
"""
import sys
from pathlib import Path

# make `src/` importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adrank.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["all", *sys.argv[1:]]))
