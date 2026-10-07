"""Phase 0: assess hypothesized positive-sign confounding at warning starts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.diagnostics.directional_scale_diagnostics import main

if __name__ == "__main__":
    main("directional")
