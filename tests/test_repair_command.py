import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.repair_data import digest, parser, run


class RepairCommandTests(unittest.TestCase):
    def test_export_requires_coverage_and_full_cohort_and_clears_stale_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games_path, injury_path = root / "games.csv", root / "injuries.csv"
            pd.DataFrame({"PLAYER_ID": [1], "GAME_ID": [10], "GAME_DATE": ["2023-01-01"], "SEASON_ID": [22022], "MIN": [30]}).to_csv(games_path, index=False)
            pd.DataFrame({"Date": ["2023-01-02"], "Relinquished": ["Alex Smith"], "Notes": ["placed on IL with ankle injury"]}).to_csv(injury_path, index=False)
            pd.DataFrame({"PLAYER_ID": [1], "PLAYER_NAME": ["Alex Smith"]}).to_csv(root / "players.csv", index=False)
            pd.DataFrame(columns=["alias", "PLAYER_ID", "evidence"]).to_csv(root / "aliases.csv", index=False)
            manifest = {"cohort": "all_regular_season_player_appearances", "complete": True, "row_count": 1, "games_sha256": digest(games_path)}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            coverage = {"source_sha256": digest(injury_path), "windows": [{"start": "2023-01-01", "end": "2023-01-20", "verified": True, "evidence": "synthetic complete-source fixture"}]}
            (root / "coverage.json").write_text(json.dumps(coverage), encoding="utf-8")
            args = parser().parse_args([
                "--games", str(games_path), "--injuries", str(injury_path),
                "--players", str(root / "players.csv"), "--aliases", str(root / "aliases.csv"),
                "--coverage", str(root / "coverage.json"), "--output", str(root / "out"),
            ])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(run(args)["training_ready"])
                args.cohort_manifest = root / "manifest.json"
                self.assertTrue(run(args)["training_ready"])
                self.assertEqual(len(pd.read_csv(root / "out/training_player_games.csv")), 1)
                (root / "coverage.json").write_text('{"windows": []}', encoding="utf-8")
                self.assertFalse(run(args)["training_ready"])
                self.assertTrue(pd.read_csv(root / "out/training_player_games.csv").empty)

    def test_changed_source_invalidates_verified_coverage(self):
        # The command checks the source hash before preparing any report labels.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"Date": ["2023-01-02"], "Relinquished": ["Alex Smith"], "Notes": ["placed on IL with ankle injury"]}).to_csv(root / "injuries.csv", index=False)
            pd.DataFrame({"PLAYER_ID": [1], "GAME_ID": [10], "GAME_DATE": ["2023-01-01"], "SEASON_ID": [22022], "MIN": [30]}).to_csv(root / "games.csv", index=False)
            pd.DataFrame({"PLAYER_ID": [1], "PLAYER_NAME": ["Alex Smith"]}).to_csv(root / "players.csv", index=False)
            pd.DataFrame(columns=["alias", "PLAYER_ID", "evidence"]).to_csv(root / "aliases.csv", index=False)
            (root / "coverage.json").write_text(json.dumps({"source_sha256": "stale", "windows": [{"start": "2023-01-01", "end": "2023-01-20", "verified": True, "evidence": "fixture"}]}), encoding="utf-8")
            args = parser().parse_args(["--injuries", str(root / "injuries.csv"), "--games", str(root / "games.csv"), "--players", str(root / "players.csv"), "--aliases", str(root / "aliases.csv"), "--coverage", str(root / "coverage.json")])
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                run(args)


if __name__ == "__main__":
    unittest.main()
