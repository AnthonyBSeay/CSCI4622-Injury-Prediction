"""Audit the historical status archive separately from IL transaction records."""
import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.repair_data import digest
from src.data.availability import audit_archive, next_game_outcomes


def load_verified_archive(path: Path) -> pd.DataFrame:
    """Verify the downloaded bytes; this does not certify reporting completeness."""
    metadata = json.loads(path.with_suffix('.metadata.json').read_text(encoding='utf-8'))
    if metadata.get('sha256') != digest(path):
        raise ValueError('Injury archive bytes do not match the source metadata hash')
    if not metadata.get('url', '').startswith('https://docs.google.com/spreadsheets/d/e/'):
        raise ValueError('Expected the published historical injury archive source URL')
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/injury_reports_2022_23.csv"))
    parser.add_argument("--games", type=Path, default=Path("data/raw/cohort_2022_23/gamelogs.csv"))
    parser.add_argument("--teams", type=Path, default=Path("data/raw/cohort_2022_23/team_games.csv"))
    parser.add_argument("--reference", type=Path, default=Path("data/reference/nba_players.csv"))
    parser.add_argument("--aliases", type=Path, default=Path("data/reference/player_aliases.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/archive_audit"))
    args = parser.parse_args()
    raw = load_verified_archive(args.archive)
    games, teams = (pd.read_csv(path) for path in [args.games, args.teams])
    archive, coverage = audit_archive(raw, games, teams, pd.read_csv(args.reference), pd.read_csv(args.aliases))
    outcomes = next_game_outcomes(games, teams, archive)
    audit = {
        "source": "Vaughn Hajra's historical NBA injury report archive",
        "source_page": "https://statsurge.substack.com/p/downloadable-nba-injury-datasets",
        "source_bytes_verified": True,
        "source_rows": len(raw), "official_game_rows": len(games),
        "official_players": int(games.PLAYER_ID.nunique()),
        "official_games": int(games.GAME_ID.nunique()), "official_team_games": len(coverage),
        "team_games_with_status_rows": int(coverage.archive_rows.gt(0).sum()),
        "team_games_without_status_rows": int(coverage.archive_rows.eq(0).sum()),
        "schedule_unmatched_rows": int(archive.schedule_status.ne("matched").sum()),
        "identity_unmatched_rows": int(archive.PLAYER_ID.isna().sum()),
        "out_status_appearance_conflicts": int(archive.out_but_appeared.sum()),
        "next_game_outcome_counts": {str(k): int(v) for k, v in outcomes.outcome_status.value_counts().items()},
        "positive_next_game_outcomes": int(outcomes.target_next_game_lower_body_absence.eq(1).sum()),
        "negative_next_game_outcomes": int(outcomes.target_next_game_lower_body_absence.eq(0).sum()),
        "unknown_next_game_outcomes": int(outcomes.target_next_game_lower_body_absence.isna().sum()),
        "seven_day_IL_target_training_ready": False,
        "limitations": [
            "DATE is the scheduled game date, not report publication time or injury onset",
            "A team-game with no archive rows is unknown, not a confirmed healthy team",
            "Out statuses contradicted by official appearances are quarantined",
            "Exploratory next-game absence labels are a different target from seven-day IL placements",
            "Outcome missingness may be informative; no full-league calibration or unbiased evaluation claim is supported",
        ],
        "inputs": {path.as_posix(): digest(path) for path in [args.archive, args.archive.with_suffix('.metadata.json'), args.games, args.teams, args.reference, args.aliases]},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    archive.to_csv(args.output / "report_player_audit.csv", index=False)
    coverage.to_csv(args.output / "team_game_coverage.csv", index=False)
    outcomes.to_csv(args.output / "next_game_outcomes_exploratory.csv", index=False)
    (args.output / "archive_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
