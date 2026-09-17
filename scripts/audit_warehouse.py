"""Run independent SQL checks against the persisted local warehouse."""
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    connection = duckdb.connect(str(PROJECT_ROOT / "data" / "baseball_warehouse.duckdb"), read_only=True)
    coverage = connection.execute(
        """
        SELECT year(game_date) AS season,
               count(*) AS pitches,
               count(DISTINCT game_pk) AS games,
               min(game_date) AS min_date,
               max(game_date) AS max_date,
               count(DISTINCT actual_data_source) AS source_count,
               count(DISTINCT dataset_split) AS split_count
        FROM raw_statcast_pitches
        GROUP BY 1 ORDER BY 1
        """
    ).fetchdf()
    duplicate_keys = connection.execute(
        """
        SELECT count(*) FROM (
            SELECT game_pk, pitcher, at_bat_number, pitch_number, count(*) AS n
            FROM raw_statcast_pitches
            GROUP BY ALL HAVING n > 1
        )
        """
    ).fetchone()[0]
    qualified_orphans = connection.execute(
        """
        SELECT count(*)
        FROM stg_qualified_pitches q
        ANTI JOIN raw_statcast_pitches r
        USING (game_pk, pitcher, at_bat_number, pitch_number)
        """
    ).fetchone()[0]
    report = connection.execute(
        """
        SELECT check_name, status, row_count, details
        FROM warehouse_integrity_report
        ORDER BY check_name
        """
    ).fetchdf()
    expected_tables = {
        "mart_historical_2024_evaluation", "mart_threshold_tradeoffs",
        "mart_bootstrap_confidence_intervals", "mart_lead_time_sensitivity",
        "fact_matched_warning_episodes", "mart_lead_time_distribution",
        "mart_missing_data_summary", "audit_excluded_pitchers", "audit_excluded_outings",
        "audit_unavailable_scores",
        "audit_cohort_selection", "audit_ingestion_segments", "mart_pitcher_model_evaluation",
    }
    persisted_tables = set(connection.execute("SHOW TABLES").fetchdf()["name"])
    missing_tables = sorted(expected_tables - persisted_tables)
    provenance_tables = expected_tables | {
        "raw_statcast_pitches", "stg_qualified_pitches", "dim_pitchers", "dim_games",
        "feat_pitcher_pitchtype_baseline", "feat_pitch_level_features",
        "fact_pitch_anomaly_scores", "fact_collapse_labels", "fact_alert_events",
        "mart_model_evaluation", "warehouse_integrity_report",
    }
    missing_source_columns = []
    unexpected_sources = {}
    for table_name in sorted(provenance_tables & persisted_tables):
        columns = set(connection.execute(f'DESCRIBE SELECT * FROM "{table_name}"').fetchdf()["column_name"])
        source_column = next(
            (name for name in ("actual_data_source", "Actual Data Source") if name in columns), None
        )
        if source_column is None:
            missing_source_columns.append(table_name)
            continue
        sources = {
            row[0] for row in connection.execute(
                f'SELECT DISTINCT "{source_column}" FROM "{table_name}" '
                f'WHERE "{source_column}" IS NOT NULL'
            ).fetchall()
        }
        if sources and sources != {"mlb_statcast"}:
            unexpected_sources[table_name] = sorted(map(str, sources))
    split_audit = connection.execute(
        """
        SELECT year(game_date) AS season, dataset_split, count(*) AS row_count
        FROM raw_statcast_pitches
        GROUP BY ALL ORDER BY season
        """
    ).fetchdf()
    connection.close()

    print(coverage.to_string(index=False))
    print(f"duplicate_pitch_keys={duplicate_keys}")
    print(f"qualified_orphans={qualified_orphans}")
    print(split_audit.to_string(index=False))
    print(f"missing_evidence_tables={missing_tables}")
    print(f"missing_source_columns={missing_source_columns}")
    print(f"unexpected_sources={unexpected_sources}")
    print(report.to_string(index=False))
    expected_split_map = {2023: "train", 2024: "validation", 2025: "test"}
    split_ok = all(
        len(split_audit[split_audit["season"].eq(year)]) == 1
        and split_audit.loc[split_audit["season"].eq(year), "dataset_split"].iloc[0] == split_name
        for year, split_name in expected_split_map.items()
    )
    if (duplicate_keys or qualified_orphans or missing_tables or missing_source_columns
            or unexpected_sources or not split_ok
            or not report["status"].eq("PASS").all()):
        raise SystemExit("Warehouse integrity audit failed.")


if __name__ == "__main__":
    main()
