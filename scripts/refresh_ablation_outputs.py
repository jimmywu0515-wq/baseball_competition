"""Recompute real-data ablations as part of a complete, validated release.

Usage: python scripts/refresh_ablation_outputs.py

The ablation results depend on the frozen primary threshold, scored pitch rows,
collapse episodes, and historical baselines. Recomputing them in isolation can
mix runs or claim parity for copied metrics, so this entry point runs the full
pipeline and its release validator.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_full_pipeline import run_pipeline


def main() -> None:
    run_pipeline(use_real_data=True)


if __name__ == "__main__":
    main()
