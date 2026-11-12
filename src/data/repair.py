"""Repair identities, preserve uncertainty, and build prospective report labels.

The outcome is a lower-body injured-list placement report, not biological onset
or a verified non-contact injury. Features are available after a completed game.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right

import numpy as np
import pandas as pd


BODY_PATTERNS = {
    "knee": r"\bknee\b|\b[ampl]?cl\b|\bmeniscus\b|\bpatell\w*",
    "ankle": r"\bankle\b",
    "achilles": r"\bachilles\b",
    "hamstring": r"\bhamstring\b",
    "calf": r"\bcalf\b",
    "foot": r"\bfoot\b|\bfeet\b|\btoe\w*|\bplantar\b",
    "hip_groin": r"\bhip\b|\bgroin\b|\badductor\b",
    "thigh_quad": r"\bthigh\b|\bquad\w*",
    "leg": r"\bleg\b|\btibia\b|\bfibula\b|\bshin\b",
}


def model_feature_columns() -> list[str]:
    """Explicit allowlist: outcome/audit columns must never become features."""
    columns = ["minutes_float", "distance", "speed", "tracking_available", "distance_per_min",
               "days_since_prev_appearance", "back_to_back_appearance", "games_7d"]
    for metric in ["minutes_float", "distance", "speed"]:
        columns.extend([f"{metric}_lag1", f"{metric}_prev_3_games_avg"])
        for days in [3, 7, 14, 28]:
            columns.extend([f"{metric}_{days}d_avg", f"{metric}_{days}d_observed"])
            if metric != "speed":
                columns.append(f"{metric}_{days}d_sum")
    return columns


def normalize_name(value: object) -> str:
    """Normalize spelling while preserving generational suffixes."""
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r"\([^)]*\)", " ", value).lower()
    # Remove punctuation without splitting initials or apostrophized names.
    value = re.sub(r"[^a-z0-9\s]", "", value)
    return " ".join(value.split())


def match_reports(
    raw: pd.DataFrame,
    players: pd.DataFrame,
    aliases: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Resolve every relinquished-player report; never choose among collisions.

    Alias rows require alias, PLAYER_ID, and evidence. Explicit aliases override
    the exact catalog name, so evidence is mandatory and retained in the audit.
    """
    if players["PLAYER_ID"].isna().any():
        raise ValueError("Player catalog contains a missing PLAYER_ID")
    catalog: dict[str, set[int]] = {}
    for row in players.itertuples(index=False):
        catalog.setdefault(normalize_name(row.PLAYER_NAME), set()).add(int(row.PLAYER_ID))
    known_ids = set(players["PLAYER_ID"].astype(int))
    alias_map: dict[str, tuple[int, str, pd.Timestamp | None, pd.Timestamp | None, str]] = {}
    if aliases is not None:
        for row in aliases.itertuples(index=False):
            key = normalize_name(row.alias)
            evidence = "" if pd.isna(row.evidence) else str(row.evidence).strip()
            pid = int(row.PLAYER_ID)
            if not key or not evidence or pid not in known_ids:
                raise ValueError("Aliases need a name, a known ID, and evidence")
            if key in alias_map:
                raise ValueError(f"Duplicate normalized alias: {key}")
            start_value, end_value = getattr(row, "start", None), getattr(row, "end", None)
            start = pd.Timestamp(start_value) if pd.notna(start_value) else None
            end = pd.Timestamp(end_value) if pd.notna(end_value) else None
            team_value = getattr(row, "team", "")
            team = str(team_value).strip() if pd.notna(team_value) else ""
            if start is not None and end is not None and start > end:
                raise ValueError("Invalid alias date range")
            alias_map[key] = (pid, evidence, start, end, team)

    data = raw.copy()
    data["source_row"] = data.index
    data["report_date"] = pd.to_datetime(data["Date"], errors="raise").dt.normalize()
    if data["report_date"].isna().any():
        raise ValueError("Missing report dates")
    # Acquired-only records are returns, not placement events. Retain them in
    # the match audit, but do not infer onset from either column.
    output = []
    for row in data.to_dict("records"):
        original = row.get("Relinquished", "")
        variants = [normalize_name(s) for s in str(original).split("/")] if pd.notna(original) else []
        candidates: set[int] = set()
        evidence = []
        used_alias = False
        for name in variants:
            alias = alias_map.get(name)
            applies = alias is not None and (
                (alias[2] is None or row["report_date"] >= alias[2])
                and (alias[3] is None or row["report_date"] <= alias[3])
                and (not alias[4] or str(row.get("Team", "")).strip() == alias[4])
            )
            if applies:
                pid, reason, _, _, _ = alias
                candidates.add(pid)
                evidence.append(reason)
                used_alias = True
            else:
                candidates.update(catalog.get(name, set()))
        status = "not_applicable" if not variants else "unmatched"
        if len(candidates) == 1:
            status = "alias" if used_alias else "exact"
        elif len(candidates) > 1:
            status = "ambiguous"
        notes = str(row.get("Notes", ""))
        bodies = sorted(k for k, pattern in BODY_PATTERNS.items() if re.search(pattern, notes, re.I))
        is_placement = bool(re.search(r"\bplaced on (?:the )?(?:IL|injured list)\b", notes, re.I))
        row.update(
            PLAYER_ID=next(iter(candidates)) if len(candidates) == 1 else pd.NA,
            match_status=status,
            candidate_ids="|".join(map(str, sorted(candidates))),
            match_evidence=" | ".join(evidence),
            body_regions="|".join(bodies),
            is_target_report=bool(is_placement and bodies),
            recovery_mentioned=bool(re.search(r"recover|surgery|rehab", notes, re.I)),
        )
        output.append(row)
    result = pd.DataFrame(output, columns=list(data.columns) + [
        "PLAYER_ID", "match_status", "candidate_ids", "match_evidence", "body_regions",
        "is_target_report", "recovery_mentioned",
    ])
    result["PLAYER_ID"] = result["PLAYER_ID"].astype("Int64")
    result["is_target_report"] = result["is_target_report"].astype(bool)
    return result


def prepare_games(games: pd.DataFrame) -> pd.DataFrame:
    """Validate keys and reject conflicting duplicate player-game records."""
    required = {"PLAYER_ID", "GAME_ID", "GAME_DATE", "SEASON_ID", "MIN"}
    missing = required - set(games)
    if missing:
        raise ValueError(f"Missing game columns: {sorted(missing)}")
    data = games.copy()
    for col in ["PLAYER_ID", "GAME_ID", "SEASON_ID"]:
        values = pd.to_numeric(data[col], errors="raise")
        if values.isna().any() or (values % 1 != 0).any():
            raise ValueError(f"Invalid integer game key: {col}")
        data[col] = values.astype("int64")
    data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"], errors="raise").dt.normalize()
    if data["GAME_DATE"].isna().any():
        raise ValueError("Missing game dates")
    data = data.drop_duplicates()
    if data.duplicated(["PLAYER_ID", "GAME_ID"]).any():
        raise ValueError("Conflicting player-game duplicates; resolve before labeling")
    if data.duplicated(["PLAYER_ID", "GAME_DATE"]).any():
        raise ValueError("Multiple appearances on one date need explicit timestamp handling")
    return data.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def attach_tracking(games: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
    data = tracking.rename(columns={"personId": "PLAYER_ID", "gameId": "GAME_ID"}).copy()
    for col in ["PLAYER_ID", "GAME_ID"]:
        data[col] = pd.to_numeric(data[col], errors="raise").astype("int64")
    data = data.drop_duplicates()
    if data.duplicated(["PLAYER_ID", "GAME_ID"]).any():
        raise ValueError("Conflicting tracking rows")
    cols = ["PLAYER_ID", "GAME_ID", "minutes", "speed", "distance"]
    return games.merge(data[cols], on=["PLAYER_ID", "GAME_ID"], how="left", validate="one_to_one")


def _minutes(value: object) -> float:
    if pd.isna(value):
        return np.nan
    try:
        text = str(value)
        if ":" in text:
            mins, secs = map(float, text.split(":"))
            if not (0 <= secs < 60 and mins >= 0):
                return np.nan
            return mins + secs / 60
        return float(text)
    except (ValueError, TypeError):
        return np.nan


def engineer_features(games: pd.DataFrame) -> pd.DataFrame:
    """Post-game, player-season histories; no imputation or fitted transforms.

    Calendar windows include the current completed game and exclude the left
    boundary. Missing tracking is preserved, with per-window observed counts.
    """
    data = prepare_games(games)
    fallback = data["MIN"].map(_minutes)
    data["minutes_float"] = data["minutes"].map(_minutes).combine_first(fallback) if "minutes" in data else fallback
    data.loc[~data["minutes_float"].between(0, 100), "minutes_float"] = np.nan
    for col in ["distance", "speed"]:
        if col not in data:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data.loc[~np.isfinite(data[col]) | (data[col] < 0), col] = np.nan
    data["tracking_available"] = data[["distance", "speed"]].notna().all(axis=1)
    data["distance_per_min"] = data["distance"] / data["minutes_float"].replace(0, np.nan)
    pieces = []
    for _, group in data.groupby(["PLAYER_ID", "SEASON_ID"], sort=False):
        group = group.copy().set_index("GAME_DATE", drop=False)
        group["days_since_prev_appearance"] = group["GAME_DATE"].diff().dt.days
        group["back_to_back_appearance"] = group["days_since_prev_appearance"].eq(1)
        for metric in ["minutes_float", "distance", "speed"]:
            values = group[metric]
            group[f"{metric}_lag1"] = values.shift(1)
            group[f"{metric}_prev_3_games_avg"] = values.shift(1).rolling(3, min_periods=1).mean()
            for days in [3, 7, 14, 28]:
                window = values.rolling(f"{days}D", min_periods=1, closed="right")
                group[f"{metric}_{days}d_avg"] = window.mean()
                group[f"{metric}_{days}d_observed"] = window.count()
                if metric != "speed":
                    group[f"{metric}_{days}d_sum"] = window.sum()
        group["games_7d"] = pd.Series(1., index=group.index).rolling("7D", closed="right").sum()
        pieces.append(group.reset_index(drop=True))
    if not pieces:
        raise ValueError("No games available")
    return pd.concat(pieces, ignore_index=True)


def validate_coverage(coverage: dict) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Require explicit, evidenced coverage; event extrema are not evidence."""
    windows = []
    for item in coverage.get("windows", []):
        if item.get("verified") is not True:
            continue
        if not str(item.get("evidence", "")).strip():
            raise ValueError("Verified coverage requires evidence")
        start, end = pd.Timestamp(item["start"]).normalize(), pd.Timestamp(item["end"]).normalize()
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError("Invalid coverage interval")
        windows.append((start, end))
    # Join adjacent windows so a horizon crossing their boundary is covered.
    merged = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1] + pd.Timedelta(days=1):
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def label_future_reports(games: pd.DataFrame, reports: pd.DataFrame, coverage: dict, horizon_days: int = 7) -> pd.DataFrame:
    """Label reports on dates (GAME_DATE, GAME_DATE + horizon], never backward.

    Unknown/ambiguous target identities conservatively block negative labels on
    overlapping horizons. Known positives remain identified within verified
    coverage. No claim of new injury episodes is made.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    data = games.copy()
    windows = validate_coverage(coverage)
    target = reports.loc[reports["is_target_report"]].copy()
    resolved = target[target["PLAYER_ID"].notna()].drop_duplicates(["PLAYER_ID", "report_date"])
    dates_by_player = {int(pid): sorted(group["report_date"].tolist()) for pid, group in resolved.groupby("PLAYER_ID")}
    unresolved = sorted(target.loc[target["PLAYER_ID"].isna(), "report_date"].unique())
    observed, values, statuses, ends = [], [], [], []
    for row in data.itertuples(index=False):
        start = pd.Timestamp(row.GAME_DATE)
        end = start + pd.Timedelta(days=horizon_days)
        dates = dates_by_player.get(int(row.PLAYER_ID), [])
        count = bisect_right(dates, end) - bisect_right(dates, start)
        covered = any(lo <= start + pd.Timedelta(days=1) and end <= hi for lo, hi in windows)
        uncertain = bisect_right(unresolved, end) > bisect_right(unresolved, start)
        observed.append(count)
        ends.append(end)
        if not covered:
            values.append(pd.NA)
            statuses.append("unknown_coverage")
        elif count:
            values.append(1)
            statuses.append("observed_positive")
        elif uncertain:
            values.append(pd.NA)
            statuses.append("unresolved_report_identity")
        else:
            values.append(0)
            statuses.append("covered_negative")
    data["prediction_date"] = data["GAME_DATE"]
    data["label_window_end"] = ends
    data["observed_future_report_days"] = observed
    data["target_report_within_horizon"] = pd.array(values, dtype="Int64")
    data["label_status"] = statuses
    return data
