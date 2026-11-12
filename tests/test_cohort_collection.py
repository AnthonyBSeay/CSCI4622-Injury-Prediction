import unittest

import pandas as pd

from scripts.collect_full_cohort import validate_season


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.players = pd.DataFrame({
            "PLAYER_ID": list(range(10)), "TEAM_ID": [1] * 5 + [2] * 5,
            "GAME_ID": [10] * 10, "GAME_DATE": ["2023-01-01"] * 10,
            "SEASON_ID": [22022] * 10, "MIN": [48] * 10,
        })
        self.teams = pd.DataFrame({"GAME_ID": [10, 10], "TEAM_ID": [1, 2]})

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


if __name__ == "__main__":
    unittest.main()
