"""Audit game-level injury statuses and construct explicitly observed outcomes.

The archive's DATE is the scheduled game date, not the timestamp of the report
or injury onset. Its rows must not be fed to the IL-transaction label builder.
"""

from __future__ import annotations

import re

import pandas as pd

from src.data.repair import BODY_PATTERNS, match_reports, prepare_games


def schedule_keys(teams: pd.DataFrame) -> pd.DataFrame:
    data = teams.copy()
    data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"]).dt.normalize()
    for col in ["GAME_ID", "TEAM_ID"]:
        data[col] = pd.to_numeric(data[col], errors="raise").astype("int64")
    if data.duplicated(["GAME_ID", "TEAM_ID"]).any():
        raise ValueError("Duplicate team-game schedule keys")
    matchups = []
    for row in data.itertuples(index=False):
        parts = re.split(r"\s+(?:@|vs\.)\s+", row.MATCHUP)
        if len(parts) != 2:
            raise ValueError(f"Invalid schedule matchup: {row.MATCHUP}")
        away, home = parts if " @ " in row.MATCHUP else parts[::-1]
        matchups.append(f"{away}@{home}")
    data["archive_matchup"] = matchups
    if data.duplicated(["GAME_DATE", "archive_matchup", "TEAM_NAME"]).any():
        raise ValueError("Ambiguous schedule date/matchup/team")
    return data


def audit_archive(raw: pd.DataFrame, games: pd.DataFrame, teams: pd.DataFrame, reference: pd.DataFrame, aliases: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"PLAYER", "STATUS", "REASON", "TEAM", "GAME", "DATE"}
    if not required.issubset(raw) or raw[list(required)].isna().any().any():
        raise ValueError("Archive has missing fields or null records")
    statuses = {"Out", "Questionable", "Probable", "Available", "Doubtful"}
    if not set(raw.STATUS).issubset(statuses):
        raise ValueError("Unrecognized archive status")
    if raw.duplicated(["PLAYER", "TEAM", "GAME", "DATE"]).any():
        raise ValueError("Multiple archive records per player/team/game need timestamp-aware resolution")
    schedule = schedule_keys(teams)
    data = raw.reset_index(drop=True).copy()
    data["archive_row"] = data.index
    data["GAME_DATE"] = pd.to_datetime(data.DATE, format="%m/%d/%Y", errors="raise")
    data = data.merge(schedule[["GAME_DATE", "archive_matchup", "TEAM_NAME", "GAME_ID", "TEAM_ID"]],
                      left_on=["GAME_DATE", "GAME", "TEAM"], right_on=["GAME_DATE", "archive_matchup", "TEAM_NAME"],
                      how="left", validate="many_to_one")
    data["schedule_status"] = data.GAME_ID.notna().map({True: "matched", False: "unmatched"})
    # Prefer the official season's names. Add current static spellings for those
    # same IDs; future naming changes do not collapse suffixes or add retirees.
    season_catalog = games[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates()
    season_catalog = pd.concat([season_catalog, reference[reference.PLAYER_ID.isin(games.PLAYER_ID)]], ignore_index=True).drop_duplicates()
    def natural_name(name: str) -> str:
        parts = name.split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}" if len(parts) == 2 else name.strip()
    identity_input = pd.DataFrame({"Date": data.GAME_DATE, "Relinquished": data.PLAYER.map(natural_name), "Notes": data.REASON,
                                   "Team": data.TEAM.map(lambda name: name.split()[-1])})
    season_aliases = aliases[aliases.PLAYER_ID.isin(season_catalog.PLAYER_ID)] if aliases is not None else None
    matched = match_reports(identity_input, season_catalog, season_aliases)
    # Static-catalog fallback can identify season-long absentees, but ambiguous
    # names are still not guessed. The season-specific result takes precedence.
    fallback = match_reports(identity_input, reference, aliases)
    for col in ["PLAYER_ID", "match_status", "candidate_ids"]:
        data[col] = matched[col].where(matched.PLAYER_ID.notna(), fallback[col]).to_numpy()
    data["PLAYER_ID"] = data.PLAYER_ID.astype("Int64")
    data["is_lower_body"] = data.REASON.map(lambda reason: reason.startswith("Injury/Illness -") and any(re.search(pattern, reason, re.I) for pattern in BODY_PATTERNS.values()))
    appearances = prepare_games(games)[["PLAYER_ID", "GAME_ID"]].assign(appeared=True)
    data["GAME_ID"] = data.GAME_ID.astype("Int64")
    data = data.merge(appearances, on=["PLAYER_ID", "GAME_ID"], how="left", validate="many_to_one")
    data["appeared"] = data.appeared.eq(True)
    data["out_but_appeared"] = data.STATUS.eq("Out") & data.appeared
    counts = data[data.schedule_status.eq("matched")].groupby(["GAME_ID", "TEAM_ID"]).size().rename("archive_rows")
    coverage = schedule.merge(counts, on=["GAME_ID", "TEAM_ID"], how="left", validate="one_to_one")
    coverage["archive_rows"] = coverage.archive_rows.fillna(0).astype(int)
    coverage["coverage_status"] = coverage.archive_rows.gt(0).map({True: "has_status_rows_not_proof_of_completeness", False: "no_archive_rows_unknown"})
    return data, coverage


def next_game_outcomes(games: pd.DataFrame, teams: pd.DataFrame, archive: pd.DataFrame, max_gap_days: int = 7) -> pd.DataFrame:
    """An exploratory absence target, distinct from the seven-day IL target.

    1: Out for a lower-body condition and absent from the official appearance log.
    0: appeared, or an Out absence documented for a non-lower-body reason.
    Unknown: missing/uncertain reports, identity ambiguity, source conflicts,
    transfers before the target game, or no next game within the horizon.
    No claim is made that these observed cases form an unbiased training sample.
    """
    if max_gap_days < 1:
        raise ValueError("max_gap_days must be positive")
    data = prepare_games(games)
    schedule = schedule_keys(teams).sort_values(["TEAM_ID", "GAME_DATE"])
    schedule["next_GAME_ID"] = schedule.groupby("TEAM_ID").GAME_ID.shift(-1).astype("Int64")
    schedule["next_GAME_DATE"] = schedule.groupby("TEAM_ID").GAME_DATE.shift(-1)
    data = data.merge(schedule[["GAME_ID", "TEAM_ID", "next_GAME_ID", "next_GAME_DATE"]], on=["GAME_ID", "TEAM_ID"], validate="many_to_one")
    data["days_to_next_game"] = (data.next_GAME_DATE - data.GAME_DATE).dt.days
    reports = archive.dropna(subset=["PLAYER_ID", "GAME_ID"]).copy()
    # Different raw names resolving to the same key are uncertain, never resolved
    # by arbitrary row order.
    duplicate_keys = set(map(tuple, reports.loc[reports.duplicated(["PLAYER_ID", "GAME_ID"], keep=False), ["PLAYER_ID", "GAME_ID"]].to_numpy()))
    report_map = {(int(row.PLAYER_ID), int(row.GAME_ID)): row for row in reports.itertuples(index=False)}
    appearance_keys = set(map(tuple, data[["PLAYER_ID", "GAME_ID"]].to_numpy()))
    histories = {pid: group.sort_values("GAME_DATE") for pid, group in data.groupby("PLAYER_ID")}
    values, reasons, source_rows = [], [], []
    for row in data.itertuples(index=False):
        value, reason, source_row = pd.NA, "no_next_game_within_horizon", pd.NA
        if pd.notna(row.next_GAME_ID) and row.days_to_next_game <= max_gap_days:
            key = (int(row.PLAYER_ID), int(row.next_GAME_ID))
            report = report_map.get(key)
            played = key in appearance_keys
            history = histories[row.PLAYER_ID]
            transferred = ((history.GAME_DATE > row.GAME_DATE) & (history.GAME_DATE <= row.next_GAME_DATE) & history.TEAM_ID.ne(row.TEAM_ID)).any()
            if transferred:
                reason = "team_change_before_target"
            elif key in duplicate_keys:
                reason = "ambiguous_report_key"
            elif report is not None and report.out_but_appeared:
                reason = "out_status_conflicts_with_appearance"
            elif played:
                value, reason = 0, "observed_appearance"
            elif report is not None and int(report.TEAM_ID) == int(row.TEAM_ID) and report.STATUS == "Out":
                value = int(report.is_lower_body)
                reason = "documented_lower_body_absence" if value else "documented_other_absence"
                source_row = report.archive_row
            else:
                reason = "absence_reason_unknown"
        values.append(value)
        reasons.append(reason)
        source_rows.append(source_row)
    data["target_next_game_lower_body_absence"] = pd.array(values, dtype="Int64")
    data["outcome_status"] = reasons
    data["outcome_archive_row"] = pd.array(source_rows, dtype="Int64")
    return data
