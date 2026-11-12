import unittest

import pandas as pd

from src.data.repair import (
    attach_tracking, engineer_features, label_future_reports, match_reports,
    model_feature_columns, normalize_name, prepare_games, validate_coverage,
)


def games():
    return pd.DataFrame({
        "PLAYER_ID": [1, 1, 2, 2], "GAME_ID": [10, 11, 10, 11],
        "SEASON_ID": [22022] * 4, "GAME_DATE": ["2023-01-01", "2023-01-10"] * 2,
        "MIN": [10, 20, 100, 50],
    })


def reports(rows):
    return pd.DataFrame(rows, columns=["PLAYER_ID", "report_date", "is_target_report"]).assign(
        report_date=lambda d: pd.to_datetime(d.report_date))


def coverage():
    return {"windows": [{"start": "2023-01-01", "end": "2023-01-20", "verified": True, "evidence": "test fixture"}]}


class IdentityTests(unittest.TestCase):
    def test_suffixes_are_not_collapsed(self):
        players = pd.DataFrame({"PLAYER_ID": [56, 1627780], "PLAYER_NAME": ["Gary Payton", "Gary Payton II"]})
        raw = pd.DataFrame({"Date": ["2023-01-02"], "Relinquished": [" Gary Payton II "], "Notes": ["placed on IL with calf injury"]})
        actual = match_reports(raw, players)
        self.assertEqual(actual.PLAYER_ID.iloc[0], 1627780)
        self.assertNotEqual(normalize_name("Gary Payton"), normalize_name("Gary Payton II"))

    def test_ambiguous_names_are_preserved(self):
        players = pd.DataFrame({"PLAYER_ID": [1, 2], "PLAYER_NAME": ["Alex Smith", "Alex Smith"]})
        raw = pd.DataFrame({"Date": ["2023-01-02"], "Relinquished": ["Alex Smith"], "Notes": ["placed on IL with knee injury"]})
        actual = match_reports(raw, players)
        self.assertEqual(actual.match_status.iloc[0], "ambiguous")
        self.assertTrue(pd.isna(actual.PLAYER_ID.iloc[0]))

    def test_returns_are_not_placements(self):
        players = pd.DataFrame({"PLAYER_ID": [1], "PLAYER_NAME": ["Alex Smith"]})
        raw = pd.DataFrame({"Date": ["2023-01-02"], "Acquired": ["Alex Smith"], "Relinquished": [None], "Notes": ["activated from IL following knee injury"]})
        self.assertFalse(match_reports(raw, players).is_target_report.iloc[0])

    def test_alias_requires_evidence(self):
        players = pd.DataFrame({"PLAYER_ID": [1], "PLAYER_NAME": ["Alex Smith"]})
        raw = pd.DataFrame({"Date": ["2023-01-02"], "Relinquished": ["Al Smith"], "Notes": ["placed on IL with knee injury"]})
        aliases = pd.DataFrame({"alias": ["Al Smith"], "PLAYER_ID": [1], "evidence": [""]})
        with self.assertRaises(ValueError):
            match_reports(raw, players, aliases)

    def test_alias_context_prevents_historical_collision(self):
        players = pd.DataFrame({"PLAYER_ID": [1, 2], "PLAYER_NAME": ["Johnny Davis", "Johnny Davis"]})
        raw = pd.DataFrame({"Date": ["1980-01-02", "2023-01-02"], "Team": ["Blazers", "Wizards"], "Relinquished": ["Johnny Davis"] * 2, "Notes": ["placed on IL with knee injury"] * 2})
        aliases = pd.DataFrame({"alias": ["Johnny Davis"], "PLAYER_ID": [2], "evidence": ["fixture"], "start": ["2022-07-01"], "end": ["2023-06-30"], "team": ["Wizards"]})
        actual = match_reports(raw, players, aliases)
        self.assertTrue(pd.isna(actual.PLAYER_ID.iloc[0]))
        self.assertEqual(actual.PLAYER_ID.iloc[1], 2)

    def test_no_study_reports_is_supported(self):
        raw = pd.DataFrame(columns=["Date", "Relinquished", "Notes"])
        players = pd.DataFrame({"PLAYER_ID": [1], "PLAYER_NAME": ["Alex Smith"]})
        actual = label_future_reports(prepare_games(games()), match_reports(raw, players), {"windows": []})
        self.assertTrue(actual.target_report_within_horizon.isna().all())

    def test_missing_target_player_is_not_silently_ignored(self):
        raw = pd.DataFrame({"Date": ["2023-01-02"], "Relinquished": [None], "Notes": ["placed on IL with calf injury"]})
        players = pd.DataFrame({"PLAYER_ID": [1], "PLAYER_NAME": ["Alex Smith"]})
        actual = match_reports(raw, players)
        self.assertTrue(actual.is_target_report.iloc[0])
        self.assertTrue(pd.isna(actual.PLAYER_ID.iloc[0]))


class FeatureTests(unittest.TestCase):
    def test_calendar_windows_and_player_isolation(self):
        actual = engineer_features(games())
        a = actual[actual.PLAYER_ID.eq(1)]
        b = actual[actual.PLAYER_ID.eq(2)]
        self.assertEqual(a.minutes_float_7d_sum.iloc[-1], 20)
        self.assertTrue(pd.isna(b.minutes_float_prev_3_games_avg.iloc[0]))
        self.assertEqual(b.minutes_float_prev_3_games_avg.iloc[-1], 100)

    def test_future_changes_do_not_change_past_features(self):
        original = engineer_features(games())
        changed = games()
        changed.loc[changed.GAME_DATE.eq("2023-01-10"), "MIN"] = 99
        revised = engineer_features(changed)
        pd.testing.assert_frame_equal(original[original.GAME_DATE.eq("2023-01-01")], revised[revised.GAME_DATE.eq("2023-01-01")])

    def test_history_resets_between_seasons(self):
        data = games().iloc[:2].copy()
        data.loc[1, "SEASON_ID"] = 22023
        self.assertTrue(pd.isna(engineer_features(data).minutes_float_lag1.iloc[-1]))

    def test_conflicting_duplicates_raise(self):
        data = games()
        conflict = data.iloc[[0]].assign(MIN=99)
        with self.assertRaises(ValueError):
            prepare_games(pd.concat([data, conflict]))

    def test_missing_tracking_keeps_appearances(self):
        tracking = pd.DataFrame({"personId": [1], "gameId": [10], "minutes": ["10:00"], "speed": [4], "distance": [1]})
        actual = engineer_features(attach_tracking(prepare_games(games()), tracking))
        self.assertEqual(len(actual), 4)
        self.assertEqual(actual.tracking_available.sum(), 1)
        self.assertTrue(actual.minutes_float.notna().all())
        self.assertTrue(pd.isna(actual.loc[actual.GAME_ID.eq(11), "distance_7d_sum"]).all())

    def test_feature_allowlist_excludes_labels(self):
        columns = model_feature_columns()
        self.assertTrue(set(columns).issubset(engineer_features(games()).columns))
        self.assertFalse(any("report" in col or "target" in col or "label" in col for col in columns))


class LabelTests(unittest.TestCase):
    def test_labels_are_forward_and_exclude_same_day(self):
        data = prepare_games(games()).iloc[[0]]
        actual = label_future_reports(data, reports([(1, "2023-01-01", True), (1, "2023-01-08", True), (1, "2023-01-09", True)]), coverage())
        self.assertEqual(actual.observed_future_report_days.iloc[0], 1)
        self.assertEqual(actual.target_report_within_horizon.iloc[0], 1)

    def test_unverified_coverage_never_creates_negatives(self):
        actual = label_future_reports(prepare_games(games()), reports([]), {"windows": []})
        self.assertTrue(actual.target_report_within_horizon.isna().all())

    def test_horizon_must_be_fully_covered(self):
        cov = {"windows": [{"start": "2023-01-01", "end": "2023-01-12", "verified": True, "evidence": "fixture"}]}
        actual = label_future_reports(prepare_games(games()), reports([]), cov)
        self.assertTrue(actual.loc[actual.GAME_DATE.eq("2023-01-10"), "target_report_within_horizon"].isna().all())
        self.assertTrue(actual.loc[actual.GAME_DATE.eq("2023-01-01"), "target_report_within_horizon"].eq(0).all())

    def test_unresolved_target_blocks_negative_labels(self):
        actual = label_future_reports(prepare_games(games()), reports([(pd.NA, "2023-01-02", True)]), coverage())
        self.assertTrue(actual.loc[actual.GAME_DATE.eq("2023-01-01"), "target_report_within_horizon"].isna().all())

    def test_duplicate_reports_do_not_inflate_counts(self):
        actual = label_future_reports(prepare_games(games()), reports([(1, "2023-01-02", True)] * 2), coverage())
        self.assertEqual(actual.observed_future_report_days.iloc[0], 1)

    def test_verified_coverage_needs_evidence(self):
        with self.assertRaises(ValueError):
            validate_coverage({"windows": [{"start": "2023-01-01", "end": "2023-01-20", "verified": True}]})


if __name__ == "__main__":
    unittest.main()
