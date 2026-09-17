"""Reproducible cohort selection from pre-test MLB records, never test outcomes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def mlb_json(path: str, params: dict) -> dict:
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
        )))
        response = session.get(f"https://statsapi.mlb.com/api/v1/{path}", params=params, timeout=60)
        response.raise_for_status()
        return response.json()


def season_pitching_records(year: int, cache_dir: Path) -> pd.DataFrame:
    """Cache official regular-season totals, retaining the aggregate for traded players."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"mlb_pitching_{year}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = mlb_json("stats", dict(
            stats="season", group="pitching", season=year, sportIds=1,
            gameType="R", playerPool="ALL", limit=2000,
        ))
        if not payload.get("stats") or not payload["stats"][0].get("splits"):
            raise RuntimeError(f"MLB returned no season records for {year}")
        path.write_text(json.dumps(payload), encoding="utf-8")
    splits = payload["stats"][0]["splits"]
    if len(splits) >= 2000:
        raise RuntimeError("MLB season response may be truncated; pagination required")
    rows = [{
        "pitcher": int(s["player"]["id"]), "pitcher_name": s["player"]["fullName"],
        "season": year, "starts": int(s["stat"]["gamesStarted"]),
        "appearances": int(s["stat"]["gamesPitched"]),
        "pitches": int(s["stat"]["numberOfPitches"]),
        "aggregate": "team" not in s,
    } for s in splits]
    frame = pd.DataFrame(rows)
    # API normally returns one aggregate per player. Refuse ambiguous team-only duplicates.
    duplicated = frame[frame.duplicated("pitcher", keep=False)]
    for pid, group in duplicated.groupby("pitcher"):
        if int(group["aggregate"].sum()) != 1:
            raise RuntimeError(f"Ambiguous MLB season totals for pitcher {pid}, {year}")
    return frame.sort_values("aggregate", ascending=False).drop_duplicates("pitcher")


def select_cohort(records: pd.DataFrame, retained_ids: list[int], target_size: int,
                  minimum_starts: int, cutoff_year: int, seed: int) -> pd.DataFrame:
    historical = records[records["season"].le(cutoff_year)].copy()
    if historical.duplicated(["pitcher", "season"]).any():
        raise ValueError("Duplicate pitcher-season selection records")
    pool = historical.groupby("pitcher", as_index=False).agg(
        pitcher_name=("pitcher_name", "first"), pretest_mlb_starts=("starts", "sum"),
    )
    pool = pool[pool["pretest_mlb_starts"].ge(minimum_starts)].copy()
    retained = set(map(int, retained_ids))
    if not retained.issubset(set(pool["pitcher"])):
        raise ValueError("Retained cohort contains pitchers without enough pre-test starts")
    if not len(retained) <= target_size <= len(pool):
        raise ValueError("Target cohort size must fit the eligible pool and retained cohort")
    # Stable hash ranks avoid dependence on API row order or numpy implementation changes.
    pool["selection_rank_hash"] = pool["pitcher"].map(
        lambda pid: hashlib.sha256(f"{seed}:{int(pid)}".encode()).hexdigest()
    )
    added = pool[~pool["pitcher"].isin(retained)].sort_values("selection_rank_hash").head(
        target_size - len(retained)
    )["pitcher"]
    pool["selected"] = pool["pitcher"].isin(retained | set(added))
    pool["selection_reason"] = "NOT_SAMPLED"
    pool.loc[pool["pitcher"].isin(added), "selection_reason"] = "SEEDED_PRETEST_SAMPLE"
    pool.loc[pool["pitcher"].isin(retained), "selection_reason"] = "RETAINED_ORIGINAL_COHORT"
    pool["selection_cutoff"] = f"{cutoff_year}-12-31"
    pool["selection_seed"] = seed
    pool["selection_source"] = "MLB Stats API regular-season pitching totals"
    return pool.sort_values("pitcher").reset_index(drop=True)


def resolve_cohort(ingestion: dict, cache_dir: Path) -> pd.DataFrame:
    policy = ingestion["cohort_selection"]
    years = list(map(int, policy["seasons"]))
    if max(years) >= 2025:
        raise ValueError("Cohort selection must use only pre-2025 records")
    records = pd.concat([season_pitching_records(y, cache_dir) for y in years], ignore_index=True)
    return select_cohort(records, ingestion["pitcher_ids"], int(policy["target_size"]),
                         int(policy["minimum_mlb_starts"]), max(years), int(policy["seed"]))
