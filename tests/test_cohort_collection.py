import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from scripts.collect_full_cohort import read_response, validate_season


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.players = pd.DataFrame({
            "PLAYER_ID": list(range(10)), "TEAM_ID": [1] * 5 + [2] * 5,
            "GAME_ID": [10] * 10, "GAME_DATE": ["2023-01-01"] * 10,
            "SEASON_ID": [22022] * 10, "MIN": [48] * 10,
        })
        self.teams = pd.DataFrame({"GAME_ID": [10, 10], "TEAM_ID": [1, 2], "GAME_DATE": ["2023-01-01"] * 2, "MIN": [240, 240]})

    def test_complete_response_is_accepted(self):
        self.assertEqual(validate_season(self.players, self.teams, "2022-23", 1)["rows"], 10)

    def test_missing_team_players_rejected(self):
        with self.assertRaises(ValueError):
            validate_season(self.players.iloc[:5], self.teams, "2022-23", 1)

    def test_partial_schedule_rejected(self):
        with self.assertRaises(ValueError):
            validate_season(self.players, self.teams, "2022-23", 1230)

    def test_duplicate_player_rows_rejected(self):
        with self.assertRaises(ValueError):
            validate_season(pd.concat([self.players, self.players.iloc[:1]]), self.teams, "2022-23", 1)

    def test_incomplete_minutes_rejected_even_with_five_players(self):
        incomplete = self.players.copy()
        incomplete.loc[0, 'MIN'] = 0
        with self.assertRaisesRegex(ValueError, 'minutes do not reconcile'):
            validate_season(incomplete, self.teams, "2022-23", 1)

    def test_inconsistent_game_dates_rejected(self):
        changed = self.players.copy()
        changed.loc[0, 'GAME_DATE'] = '2023-01-02'
        with self.assertRaisesRegex(ValueError, 'dates disagree'):
            validate_season(changed, self.teams, "2022-23", 1)

    def test_cached_response_checks_hash_and_season(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2022-23_P.json"
            payload = json.dumps({"resultSets": [{"name": "LeagueGameLog", "headers": ["PLAYER_ID"], "rowSet": [[1]]}]}).encode()
            path.write_bytes(payload)
            metadata = {"url": "https://stats.nba.com/stats/leaguegamelog?Season=2022-23&PlayerOrTeam=P&SeasonType=Regular%20Season", "sha256": hashlib.sha256(payload).hexdigest()}
            path.with_suffix('.metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
            self.assertEqual(len(read_response(path, "2022-23", "P")), 1)
            with self.assertRaisesRegex(ValueError, "requested NBA source"):
                read_response(path, "2023-24", "P")
            path.write_bytes(payload + b' ')
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                read_response(path, "2022-23", "P")


if __name__ == "__main__":
    unittest.main()
