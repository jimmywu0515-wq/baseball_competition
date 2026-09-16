"""Data integrity gates for Statcast ingestion and warehouse publication."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd


PITCH_KEY = ["game_pk", "pitcher", "at_bat_number", "pitch_number"]
REQUIRED_RAW_COLUMNS = PITCH_KEY + [
    "game_date", "pitch_type", "release_speed", "release_pos_x",
    "release_pos_z", "release_extension", "release_spin_rate", "spin_axis",
    "pfx_x", "pfx_z", "actual_data_source",
]


class DataIntegrityError(RuntimeError):
    """Raised before publication when an integrity invariant is violated."""


def prepare_raw_pitch_data(
    raw_df: pd.DataFrame,
    expected_source: str,
    start_dt: str,
    end_dt: str,
    required_years: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Normalize, deduplicate exact repeats, and reject conflicting/raw gaps."""
    if raw_df.empty:
        raise DataIntegrityError("The raw Statcast download is empty.")
    missing_columns = sorted(set(REQUIRED_RAW_COLUMNS) - set(raw_df.columns))
    if missing_columns:
        raise DataIntegrityError(f"Raw data is missing required columns: {missing_columns}")

    result = raw_df.copy()
    if expected_source == "mlb_statcast":
        if "game_type" not in result.columns:
            raise DataIntegrityError("MLB Statcast data is missing the game_type field.")
        result = result[result["game_type"].astype(str).eq("R")].copy()
        if result.empty:
            raise DataIntegrityError("No MLB regular-season pitches remain after filtering game_type='R'.")
    result["game_date"] = pd.to_datetime(result["game_date"], errors="coerce").dt.normalize()
    if result["game_date"].isna().any():
        raise DataIntegrityError("Raw data contains invalid or missing game_date values.")
    if result[PITCH_KEY].isna().any().any():
        null_counts = result[PITCH_KEY].isna().sum().to_dict()
        raise DataIntegrityError(f"Raw pitch keys contain nulls: {null_counts}")

    # Exact API repeats are harmless and deterministic to remove. Rows sharing a
    # pitch key but disagreeing elsewhere are ambiguous and must stop the load.
    result = result.drop_duplicates().copy()
    conflicting = result.duplicated(PITCH_KEY, keep=False)
    if conflicting.any():
        examples = result.loc[conflicting, PITCH_KEY].drop_duplicates().head(5).to_dict("records")
        raise DataIntegrityError(f"Conflicting duplicate pitch keys detected: {examples}")

    start = pd.Timestamp(start_dt)
    end = pd.Timestamp(end_dt)
    observed_min = result["game_date"].min()
    observed_max = result["game_date"].max()
    if observed_min < start or observed_max > end:
        raise DataIntegrityError(
            f"Observed dates {observed_min.date()}..{observed_max.date()} exceed requested range "
            f"{start.date()}..{end.date()}."
        )
    sources = set(result["actual_data_source"].dropna().astype(str).unique())
    if sources != {expected_source}:
        raise DataIntegrityError(f"Expected source {expected_source!r}, found {sorted(sources)}")

    years = set(result["game_date"].dt.year.unique())
    required = set(required_years or [])
    if required - years:
        raise DataIntegrityError(f"Missing requested seasons: {sorted(required - years)}")

    result["ingest_season"] = result["game_date"].dt.year.astype(int)
    return result.sort_values(
        ["game_date", "game_pk", "pitcher", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)


def _assert_unique(df: pd.DataFrame, key: Sequence[str], name: str) -> None:
    duplicate_count = int(df.duplicated(list(key)).sum())
    if duplicate_count:
        raise DataIntegrityError(f"{name} contains {duplicate_count} duplicate keys on {list(key)}")


def audit_pipeline_frames(
    frames: Dict[str, pd.DataFrame],
    expected_source: str,
    required_years: Iterable[int],
) -> pd.DataFrame:
    """Validate cross-table invariants before any production table is replaced."""
    required_tables = {
        "raw_statcast_pitches", "stg_qualified_pitches", "dim_pitchers", "dim_games",
        "feat_pitcher_pitchtype_baseline", "feat_pitch_level_features",
        "fact_pitch_anomaly_scores", "fact_collapse_labels", "fact_alert_events",
        "mart_model_evaluation",
    }
    missing_tables = sorted(required_tables - set(frames))
    if missing_tables:
        raise DataIntegrityError(f"Missing pipeline outputs: {missing_tables}")

    raw = frames["raw_statcast_pitches"]
    qualified = frames["stg_qualified_pitches"]
    scores = frames["fact_pitch_anomaly_scores"]
    labels = frames["fact_collapse_labels"]
    games = frames["dim_games"]

    _assert_unique(raw, PITCH_KEY, "raw_statcast_pitches")
    _assert_unique(qualified, PITCH_KEY, "stg_qualified_pitches")
    _assert_unique(scores, PITCH_KEY, "fact_pitch_anomaly_scores")
    _assert_unique(labels, PITCH_KEY, "fact_collapse_labels")
    _assert_unique(games, ["game_pk", "pitcher"], "dim_games")

    raw_keys = pd.MultiIndex.from_frame(raw[PITCH_KEY])
    qualified_keys = pd.MultiIndex.from_frame(qualified[PITCH_KEY])
    if not qualified_keys.isin(raw_keys).all():
        raise DataIntegrityError("Qualified pitches contain keys absent from the raw table.")
    score_keys = pd.MultiIndex.from_frame(scores[PITCH_KEY])
    label_keys = pd.MultiIndex.from_frame(labels[PITCH_KEY])
    if len(scores) != len(labels) or not score_keys.equals(label_keys):
        raise DataIntegrityError("Gold score and label tables are not row-key aligned.")

    for table_name, frame in frames.items():
        if frame.empty and table_name in required_tables and table_name != "fact_alert_events":
            raise DataIntegrityError(f"{table_name} is unexpectedly empty.")
        source_column = next(
            (column for column in ("actual_data_source", "Actual Data Source") if column in frame.columns),
            None,
        )
        if source_column is None:
            raise DataIntegrityError(f"{table_name} is missing an actual data source column.")
        if not frame.empty:
            sources = set(frame[source_column].dropna().astype(str).unique())
            if sources != {expected_source}:
                raise DataIntegrityError(f"{table_name} has unexpected sources: {sorted(sources)}")

    raw_dates = pd.to_datetime(raw["game_date"], errors="raise")
    observed_years = set(raw_dates.dt.year.unique())
    missing_years = set(required_years) - observed_years
    if missing_years:
        raise DataIntegrityError(f"Raw warehouse frame is missing seasons: {sorted(missing_years)}")
    if (raw_dates.dt.year.eq(2025)).any():
        split_2025 = set(raw.loc[raw_dates.dt.year.eq(2025), "dataset_split"].astype(str).unique())
        if split_2025 != {"test"}:
            raise DataIntegrityError(f"2025 rows must remain held-out test data, found {sorted(split_2025)}")
    expected_splits = {2023: "train", 2024: "validation", 2025: "test"}
    for year, expected_split in expected_splits.items():
        season_mask = raw_dates.dt.year.eq(year)
        if season_mask.any():
            observed_splits = set(raw.loc[season_mask, "dataset_split"].astype(str).unique())
            if observed_splits != {expected_split}:
                raise DataIntegrityError(
                    f"{year} rows must be {expected_split!r}, found {sorted(observed_splits)}"
                )
    if expected_source == "mlb_statcast":
        game_types = set(raw["game_type"].dropna().astype(str).unique())
        if game_types != {"R"}:
            raise DataIntegrityError(f"MLB warehouse must contain regular-season games only: {sorted(game_types)}")

    now = datetime.now(timezone.utc).isoformat()
    report_rows = []
    for year, group in raw.assign(_year=raw_dates.dt.year).groupby("_year"):
        report_rows.append({
            "check_name": f"raw_season_{int(year)}",
            "status": "PASS",
            "row_count": int(len(group)),
            "details": f"{group['game_date'].min()} through {group['game_date'].max()}",
            "actual_data_source": expected_source,
            "checked_at_utc": now,
        })
    report_rows.extend([
        {
            "check_name": "raw_pitch_key_uniqueness", "status": "PASS",
            "row_count": int(len(raw)), "details": "zero duplicate pitch keys",
            "actual_data_source": expected_source, "checked_at_utc": now,
        },
        {
            "check_name": "gold_score_label_alignment", "status": "PASS",
            "row_count": int(len(scores)), "details": "identical ordered pitch keys",
            "actual_data_source": expected_source, "checked_at_utc": now,
        },
        {
            "check_name": "qualified_raw_referential_integrity", "status": "PASS",
            "row_count": int(len(qualified)), "details": "all qualified keys exist in raw",
            "actual_data_source": expected_source, "checked_at_utc": now,
        },
        {
            "check_name": "game_scope", "status": "PASS",
            "row_count": int(len(raw)),
            "details": "MLB regular season only" if expected_source == "mlb_statcast" else "explicit simulation benchmark",
            "actual_data_source": expected_source, "checked_at_utc": now,
        },
    ])
    return pd.DataFrame(report_rows)


def audit_persisted_warehouse(storage, expected_source: str, required_years: Iterable[int]) -> pd.DataFrame:
    """Reload persisted DuckDB tables and re-run the cross-table audit."""
    table_names = [
        "raw_statcast_pitches", "stg_qualified_pitches", "dim_pitchers", "dim_games",
        "feat_pitcher_pitchtype_baseline", "feat_pitch_level_features",
        "fact_pitch_anomaly_scores", "fact_collapse_labels", "fact_alert_events",
        "mart_model_evaluation",
    ]
    frames = {name: storage.load_table(name) for name in table_names}
    report = audit_pipeline_frames(frames, expected_source, required_years)
    report.loc[len(report)] = {
        "check_name": "duckdb_reload", "status": "PASS",
        "row_count": int(len(frames["raw_statcast_pitches"])),
        "details": "all production tables reloaded and revalidated",
        "actual_data_source": expected_source,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return report
