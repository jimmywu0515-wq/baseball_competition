"""Recompute lead-time artifacts as part of a complete real-data release."""
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
