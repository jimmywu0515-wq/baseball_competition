"""Validate a complete published or staged evaluation directory offline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.presentation import validate_release_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory containing one complete run")
    args = parser.parse_args()
    manifest = validate_release_artifacts(args.output_dir)
    print(f"Validated run {manifest['run_id']} protocol {manifest['protocol_sha256']}")


if __name__ == "__main__":
    main()
