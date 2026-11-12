import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from src.data.availability import audit_archive, next_game_outcomes
from scripts.audit_injury_archive import load_verified_archive


def fixtures():
    teams = pd.DataFrame({
        "SEASON_ID": [22022] * 4, "GAME_ID": [10, 10, 11, 11], "TEAM_ID": [1, 2, 1, 2],
        "TEAM_ABBREVIATION": ["AAA", "BBB", "AAA", "BBB"], "TEAM_NAME": ["Team A", "Team B"] * 2,
        "GAME_DATE": ["2023-01-01"] * 2 + ["2023-01-03"] * 2,
        "MATCHUP": ["AAA vs. BBB", "BBB @ AAA", "AAA @ BBB", "BBB vs. AAA"], "MIN": [48] * 4,
    })
    games = teams.iloc[[0, 3]].copy().assign(PLAYER_ID=[1, 2], PLAYER_NAME=["Alex Smith", "Ben Jones"])
    reference = pd.DataFrame({"PLAYER_ID": [1, 2], "PLAYER_NAME": ["Alex Smith", "Ben Jones"]})
    raw = pd.DataFrame({"PLAYER": ["Smith, Alex"], "STATUS": ["Out"], "REASON": ["Injury/Illness - Left Knee; Soreness"],
                        "TEAM": ["Team A"], "GAME": ["AAA@BBB"], "DATE": ["01/03/2023"]})
    return games, teams, reference, raw


class AvailabilityTests(unittest.TestCase):
    def build(self, update=None, played=False):
        games, teams, reference, raw = fixtures()
        if update:
            for key, value in update.items():
                raw[key] = value
        if played:
            extra = teams.iloc[[2]].copy().assign(PLAYER_ID=1, PLAYER_NAME="Alex Smith")
            games = pd.concat([games, extra], ignore_index=True)
        archive, coverage = audit_archive(raw, games, teams, reference)
        result = next_game_outcomes(games, teams, archive)
        return result[result.PLAYER_ID.eq(1)].iloc[0], archive, coverage

    def test_lower_body_out_and_no_appearance_is_positive(self):
        row, archive, coverage = self.build()
        self.assertEqual(row.target_next_game_lower_body_absence, 1)
        self.assertEqual(archive.PLAYER_ID.iloc[0], 1)
        self.assertEqual(coverage.archive_rows.eq(0).sum(), 3)

    def test_out_and_appearance_is_quarantined(self):
        row, _, _ = self.build(played=True)
        self.assertTrue(pd.isna(row.target_next_game_lower_body_absence))
        self.assertEqual(row.outcome_status, "out_status_conflicts_with_appearance")

    def test_playing_is_observed_negative(self):
        row, _, _ = self.build({"STATUS": "Questionable"}, played=True)
        self.assertEqual(row.target_next_game_lower_body_absence, 0)

    def test_uncertain_status_and_absence_stay_unknown(self):
        row, _, _ = self.build({"STATUS": "Doubtful"})
        self.assertTrue(pd.isna(row.target_next_game_lower_body_absence))

    def test_report_for_other_game_does_not_authorize_negative(self):
        row, _, _ = self.build({"DATE": "01/01/2023", "GAME": "BBB@AAA"})
        self.assertTrue(pd.isna(row.target_next_game_lower_body_absence))

    def test_documented_upper_body_absence_is_other_cause(self):
        row, _, _ = self.build({"REASON": "Injury/Illness - Right Shoulder; Sprain"})
        self.assertEqual(row.target_next_game_lower_body_absence, 0)
        self.assertEqual(row.outcome_status, "documented_other_absence")

    def test_midfoot_and_tibial_injuries_are_lower_body(self):
        for reason in ['Injury/Illness - Left Midfoot; Sprain', 'Injury/Illness - Left Tibial Plateau; Fracture']:
            with self.subTest(reason=reason):
                row, _, _ = self.build({'REASON': reason})
                self.assertEqual(row.target_next_game_lower_body_absence, 1)

    def test_bad_schedule_join_remains_unknown(self):
        row, archive, _ = self.build({"GAME": "AAA@ZZZ"})
        self.assertEqual(archive.schedule_status.iloc[0], "unmatched")
        self.assertTrue(pd.isna(row.target_next_game_lower_body_absence))

    def test_missing_next_game_is_not_negative(self):
        games, teams, reference, raw = fixtures()
        archive, _ = audit_archive(raw, games, teams, reference)
        actual = next_game_outcomes(games, teams, archive)
        self.assertTrue(pd.isna(actual.loc[actual.GAME_ID.eq(11), "target_next_game_lower_body_absence"]).all())

    def test_transfer_before_target_is_unknown(self):
        games, teams, reference, raw = fixtures()
        changed_team = teams.iloc[[3]].copy().assign(PLAYER_ID=1, PLAYER_NAME="Alex Smith")
        games = pd.concat([games, changed_team], ignore_index=True)
        archive, _ = audit_archive(raw, games, teams, reference)
        row = next_game_outcomes(games, teams, archive).iloc[0]
        self.assertEqual(row.outcome_status, "team_change_before_target")
        self.assertTrue(pd.isna(row.target_next_game_lower_body_absence))

    def test_duplicate_status_rows_are_rejected(self):
        games, teams, reference, raw = fixtures()
        with self.assertRaises(ValueError):
            audit_archive(pd.concat([raw, raw]), games, teams, reference)

    def test_changed_archive_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'archive.csv'
            path.write_bytes(b'PLAYER,STATUS\nAlex,Out\n')
            metadata = {'url': 'https://docs.google.com/spreadsheets/d/e/test/pub?output=csv', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            path.with_suffix('.metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
            self.assertEqual(len(load_verified_archive(path)), 1)
            path.write_bytes(path.read_bytes() + b'Ben,Out\n')
            with self.assertRaisesRegex(ValueError, 'metadata hash'):
                load_verified_archive(path)


if __name__ == "__main__":
    unittest.main()
