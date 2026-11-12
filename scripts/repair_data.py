"""Run with python -m scripts.repair_data from the repository root."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.repair import (
    attach_tracking, engineer_features, label_future_reports, match_reports,
    model_feature_columns, prepare_games, validate_coverage,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    inputs = {"injuries": args.injuries, "games": args.games, "players": args.players,
              "tracking": args.tracking, "aliases": args.aliases, "coverage": args.coverage}
    raw = pd.read_csv(args.injuries)
    raw_dates = pd.to_datetime(raw["Date"], errors="raise")
    original_games = pd.read_csv(args.games)
    games = prepare_games(original_games)
    players = pd.read_csv(args.players)
    aliases = pd.read_csv(args.aliases)
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    windows = validate_coverage(coverage)
    if windows and coverage.get("source_sha256") != digest(args.injuries):
        raise ValueError("Verified coverage must reference this exact injury file's source_sha256")
    # Include report dates that could fall in a game row's future horizon.
    selected = raw.loc[raw_dates.between(games.GAME_DATE.min(), games.GAME_DATE.max() + pd.Timedelta(days=args.horizon_days))].copy()
    reports = match_reports(selected, players, aliases)
    if args.tracking:
        games = attach_tracking(games, pd.read_csv(args.tracking))
    features = engineer_features(games)
    candidates = label_future_reports(features, reports, coverage, args.horizon_days)

    cohort_verified = False
    manifest_error = "No verified full-cohort collection manifest supplied"
    if args.cohort_manifest:
        inputs["cohort_manifest"] = args.cohort_manifest
        manifest = json.loads(args.cohort_manifest.read_text(encoding="utf-8"))
        cohort_verified = (
            manifest.get("cohort") == "all_regular_season_player_appearances"
            and manifest.get("complete") is True
            and manifest.get("games_sha256") == digest(args.games)
            and manifest.get("row_count") == len(original_games)
        )
        if not cohort_verified:
            raise ValueError("Full-cohort manifest is incomplete or does not match the games file")
        manifest_error = ""

    target = reports.loc[reports.is_target_report].copy()
    unresolved = target.loc[target.PLAYER_ID.isna()]
    ready = bool(cohort_verified and windows and candidates.target_report_within_horizon.notna().any())
    blockers = []
    if not cohort_verified:
        blockers.append(manifest_error)
    if not windows:
        blockers.append("No independently verified injury-report coverage windows; event date extrema are not proof of completeness")
    if not candidates.target_report_within_horizon.notna().any():
        blockers.append("No fully observed prediction horizons available")
    if not target.empty and len(unresolved):
        blockers.append("Unresolved target-report identities: affected negative horizons remain unknown")

    # Observed source bounds are diagnostic only and never authorize negatives.
    source_start, source_end = raw_dates.min(), raw_dates.max()
    season_rows = []
    for season, group in candidates.groupby("SEASON_ID"):
        season_rows.append({
            "SEASON_ID": int(season), "rows": len(group),
            "players": int(group.PLAYER_ID.nunique()),
            "first_game": str(group.GAME_DATE.min().date()),
            "last_game": str(group.GAME_DATE.max().date()),
            "rows_horizon_after_source_end": int((group.label_window_end > source_end).sum()),
            "unknown_labels": int(group.target_report_within_horizon.isna().sum()),
            "known_positive_labels": int(group.target_report_within_horizon.eq(1).sum()),
        })
    audit = {
        "schema_version": 2,
        "prediction_contract": "After completed game; lower-body IL placement report on a later calendar date within horizon",
        "horizon_days": args.horizon_days,
        "outcome_limitations": ["Report date is not injury onset", "Non-contact mechanism is not established", "Repeated placements are not independent injury episodes"],
        "source_observed_start": str(source_start.date()),
        "source_observed_end": str(source_end.date()),
        "source_rows": len(raw), "study_report_rows": len(reports),
        "target_report_rows": len(target),
        "target_report_player_days": int(target.dropna(subset=["PLAYER_ID"]).drop_duplicates(["PLAYER_ID", "report_date"]).shape[0]),
        "unresolved_target_reports": len(unresolved),
        "target_reports_outside_game_cohort": int((target.PLAYER_ID.notna() & ~target.PLAYER_ID.isin(games.PLAYER_ID)).sum()),
        "unresolved_target_names": sorted(unresolved.Relinquished.dropna().str.strip().unique().tolist()),
        "game_rows": len(candidates), "players": int(candidates.PLAYER_ID.nunique()),
        "exact_duplicate_game_rows_removed": len(original_games) - len(games),
        "label_status_counts": {str(k): int(v) for k, v in candidates.label_status.value_counts().items()},
        "verified_coverage_windows": [{"start": str(lo.date()), "end": str(hi.date())} for lo, hi in windows],
        "full_cohort_verified": cohort_verified,
        "training_ready": ready, "blockers_and_warnings": blockers,
        "inputs": {name: {"path": path.as_posix(), "sha256": digest(path)} for name, path in inputs.items() if path},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    reports.to_csv(args.output / "report_identity_audit.csv", index=False)
    target.to_csv(args.output / "target_reports.csv", index=False)
    candidates.to_csv(args.output / "candidate_player_games.csv", index=False)
    # Always replace this file, including with zero rows on a blocked run.
    training = candidates.loc[candidates.target_report_within_horizon.notna()] if ready else candidates.iloc[:0]
    training.to_csv(args.output / "training_player_games.csv", index=False)
    pd.DataFrame(season_rows).to_csv(args.output / "coverage_by_season.csv", index=False)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (args.output / "feature_schema.json").write_text(json.dumps({
        "feature_columns": model_feature_columns(),
        "target_column": "target_report_within_horizon",
        "split_metadata": ["PLAYER_ID", "GAME_ID", "GAME_DATE", "SEASON_ID", "label_window_end"],
        "target_only_columns": ["observed_future_report_days", "label_status", "target_report_within_horizon"],
        "missing_values": "Preserved. Fit imputation on training data only.",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ["source_observed_end", "game_rows", "players", "target_report_rows", "unresolved_target_reports", "training_ready", "blockers_and_warnings"]}, indent=2))
    return audit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--injuries", type=Path, default=Path("data/raw/NBA_Injury_Stats.csv"))
    result.add_argument("--games", type=Path, default=Path("data/raw/gamelogs.csv"))
    result.add_argument("--players", type=Path, default=Path("data/reference/nba_players.csv"))
    result.add_argument("--tracking", type=Path, default=None, help="Optional tracking CSV; missing tracking never removes game rows")
    result.add_argument("--aliases", type=Path, default=Path("data/reference/player_aliases.csv"))
    result.add_argument("--coverage", type=Path, default=Path("config/injury_coverage.json"))
    result.add_argument("--cohort-manifest", type=Path)
    result.add_argument("--horizon-days", type=int, default=7)
    result.add_argument("--output", type=Path, default=Path("data/processed/v2"))
    return result


if __name__ == "__main__":
    run(parser().parse_args())
