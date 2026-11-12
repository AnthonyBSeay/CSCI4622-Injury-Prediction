"""Fetch complete historical regular-season player logs, independent of injury.

Uses NBA Stats' leaguegamelog endpoint with PlayerOrTeam=P and T. Intended for
completed 82-game seasons. Raw responses and provenance are retained. A failed
or incomplete collection never emits a complete cohort manifest.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from src.data.repair import prepare_games


ENDPOINT = "https://stats.nba.com/stats/leaguegamelog"


def fetch(season: str, mode: str, cache: Path, timeout: int = 30) -> pd.DataFrame:
    parameters = {"Counter": 0, "Direction": "DESC", "LeagueID": "00", "PlayerOrTeam": mode,
                  "Season": season, "SeasonType": "Regular Season", "Sorter": "DATE", "DateFrom": "", "DateTo": ""}
    url = ENDPOINT + "?" + urllib.parse.urlencode(parameters)
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com", "Accept": "application/json", "Accept-Encoding": "gzip",
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    payload = gzip.decompress(payload)
            data = json.loads(payload)
            sets = data.get("resultSets", [])
            result = next(s for s in sets if s.get("name") == "LeagueGameLog")
            frame = pd.DataFrame(result["rowSet"], columns=result["headers"])
            if frame.empty:
                raise ValueError(f"Empty NBA response: {season}/{mode}")
            cache.mkdir(parents=True, exist_ok=True)
            (cache / f"{season}_{mode}.json").write_bytes(payload)
            (cache / f"{season}_{mode}.metadata.json").write_text(json.dumps({
                "url": url, "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "sha256": hashlib.sha256(payload).hexdigest(), "rows": len(frame),
            }, indent=2) + "\n", encoding="utf-8")
            return frame
        except (OSError, ValueError, KeyError, StopIteration) as error:
            if attempt == 2:
                raise RuntimeError(f"NBA collection failed for {season}/{mode}: {error}") from error
            time.sleep(2 ** attempt)
    raise AssertionError("Unreachable")


def validate_season(players: pd.DataFrame, teams: pd.DataFrame, season: str, expected_games: int) -> dict:
    required = {"PLAYER_ID", "TEAM_ID", "GAME_ID", "GAME_DATE", "SEASON_ID", "MIN"}
    if not required.issubset(players):
        raise ValueError(f"Player response lacks columns: {required - set(players)}")
    if not {"GAME_ID", "TEAM_ID"}.issubset(teams):
        raise ValueError("Team response lacks GAME_ID or TEAM_ID")
    if players.duplicated(["PLAYER_ID", "GAME_ID"]).any() or teams.duplicated(["TEAM_ID", "GAME_ID"]).any():
        raise ValueError("Duplicate NBA response keys")
    if teams.GAME_ID.nunique() != expected_games or not teams.groupby("GAME_ID").TEAM_ID.nunique().eq(2).all():
        raise ValueError(f"Incomplete team schedule for {season}; expected {expected_games} games")
    player_keys = set(map(tuple, players[["GAME_ID", "TEAM_ID"]].to_numpy()))
    team_keys = set(map(tuple, teams[["GAME_ID", "TEAM_ID"]].to_numpy()))
    if player_keys != team_keys:
        raise ValueError(f"Missing or extra player team-games for {season}")
    if not players.groupby(["GAME_ID", "TEAM_ID"]).PLAYER_ID.nunique().ge(5).all():
        raise ValueError(f"Fewer than five player appearances for a team-game in {season}")
    expected_season_id = int("2" + season[:4])
    if not pd.to_numeric(players.SEASON_ID, errors="raise").eq(expected_season_id).all():
        raise ValueError("Unexpected season IDs in player response")
    prepare_games(players)
    return {"season": season, "games": int(teams.GAME_ID.nunique()), "rows": len(players), "players": int(players.PLAYER_ID.nunique())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", default=["2022-23"])
    parser.add_argument("--expected-games-per-season", type=int, default=1230)
    parser.add_argument("--output", type=Path, default=Path("data/raw/full_cohort"))
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    if len(args.seasons) != len(set(args.seasons)):
        parser.error("Duplicate seasons")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "cohort_manifest.json"
    # Invalidate any prior manifest before making network requests.
    manifest_path.write_text(json.dumps({"complete": False, "requested_seasons": args.seasons}) + "\n", encoding="utf-8")
    frames, seasons = [], []
    try:
        for season in args.seasons:
            players = fetch(season, "P", args.output / "responses", args.timeout)
            time.sleep(1)
            teams = fetch(season, "T", args.output / "responses", args.timeout)
            seasons.append(validate_season(players, teams, season, args.expected_games_per_season))
            frames.append(players)
        games = prepare_games(pd.concat(frames, ignore_index=True))
        csv_path = args.output / "gamelogs.csv"
        games.to_csv(csv_path, index=False)
        manifest = {
            "cohort": "all_regular_season_player_appearances", "complete": True,
            "source": ENDPOINT, "seasons": seasons, "row_count": len(games),
            "games_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "validation": "Full player endpoint responses; team schedule reconciliation; minimum five players per team-game",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2))
    except Exception as error:
        manifest_path.write_text(json.dumps({"complete": False, "requested_seasons": args.seasons, "error": str(error)}, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
