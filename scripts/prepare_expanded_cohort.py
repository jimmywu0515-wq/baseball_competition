"""Freeze the expanded cohort and cache its Statcast data before modeling."""
from pathlib import Path
import sys
import logging

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.configuration import load_project_config
from src.data_ingest.cohort import resolve_cohort
from src.data_ingest.statcast_loader import StatcastLoader


def prepare_expanded_cohort(base_dir: Path = ROOT):
    """Freeze the cohort and materialize all raw cache segments."""
    base_dir = Path(base_dir)
    config = load_project_config(base_dir / "config" / "config.yaml")
    ingestion = config["ingestion"]
    audit = resolve_cohort(ingestion, base_dir / "data/raw/cohort")
    output = base_dir / "outputs/cohort_expansion"
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "selection_pool.csv", index=False)
    cohort = audit[audit["selected"]]
    print(cohort[["pitcher", "pitcher_name", "pretest_mlb_starts", "selection_reason"]].to_string(index=False), flush=True)
    loader = StatcastLoader(cache_dir=str(base_dir / "data" / "raw"))
    data = loader.fetch_real_pitchers_statcast(
        pitcher_ids=cohort["pitcher"].tolist(), start_dt=ingestion["start_date"],
        end_dt=ingestion["end_date"], pitcher_names=cohort.set_index("pitcher")["pitcher_name"].to_dict(),
        verify_empty_seasons=True,
    )
    loader.last_segment_audit.to_csv(output / "ingestion_segments.csv", index=False)
    print(f"Cached {len(data):,} pitches for {len(cohort)} preselected pitchers", flush=True)
    return data, audit, loader.last_segment_audit.copy()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prepare_expanded_cohort(ROOT)


if __name__ == "__main__":
    main()
