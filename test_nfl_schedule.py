import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl
from streamlit.testing.v1 import AppTest

from nfl_model import get_upcoming_games, prediction_week_mismatch


class UpcomingGameTests(unittest.TestCase):
    def test_excludes_kickoffs_that_have_passed_even_if_scores_are_missing(self):
        games = pd.DataFrame(
            [
                ["past", 10, "2026-11-11", "10:00", "REG", None, None],
                ["at-now", 10, "2026-11-11", "11:00", "REG", None, None],
                ["later-today", 10, "2026-11-11", "15:00", "REG", None, None],
                ["later-week", 10, "2026-11-12", "20:15", "REG", None, None],
                ["next-week", 11, "2026-11-19", "20:15", "REG", None, None],
            ],
            columns=[
                "game_id",
                "week",
                "gameday",
                "gametime",
                "game_type",
                "home_score",
                "away_score",
            ],
        )
        now = datetime(2026, 11, 11, 16, 0, tzinfo=timezone.utc)

        upcoming = get_upcoming_games(games, now=now)

        self.assertEqual(upcoming["game_id"].tolist(), ["later-today", "later-week"])
        self.assertEqual(upcoming["week"].unique().tolist(), [10])
        self.assertTrue((upcoming["kickoff"] > now).all())

    def test_uses_next_future_week_when_all_games_in_current_week_have_started(self):
        games = pd.DataFrame(
            [
                ["already-started", 10, "2026-11-11", "10:00", "REG", None, None],
                ["week-11", 11, "2026-11-19", "20:15", "REG", None, None],
                ["week-12", 12, "2026-11-26", "20:15", "REG", None, None],
            ],
            columns=[
                "game_id",
                "week",
                "gameday",
                "gametime",
                "game_type",
                "home_score",
                "away_score",
            ],
        )
        now = datetime(2026, 11, 11, 16, 0, tzinfo=timezone.utc)

        upcoming = get_upcoming_games(games, now=now)

        self.assertEqual(upcoming["game_id"].tolist(), ["week-11"])
        self.assertEqual(upcoming["week"].unique().tolist(), [11])


class PredictionWeekMismatchTests(unittest.TestCase):
    def test_reports_stale_prediction_csv_week(self):
        predictions = pd.DataFrame({"week": [10, 10, 10]})
        upcoming = pd.DataFrame({"week": [11, 11]})

        self.assertTrue(prediction_week_mismatch(predictions, upcoming))

    def test_accepts_matching_prediction_csv_week(self):
        predictions = pd.DataFrame({"week": [11, 11]})
        upcoming = pd.DataFrame({"week": [11, 11, 11]})

        self.assertFalse(prediction_week_mismatch(predictions, upcoming))

    def test_does_not_claim_mismatch_without_a_comparable_slate(self):
        predictions = pd.DataFrame({"week": [10]})
        no_upcoming = pd.DataFrame({"week": pd.Series(dtype=int)})

        self.assertFalse(prediction_week_mismatch(predictions, no_upcoming))


class DashboardScheduleFailureTests(unittest.TestCase):
    def test_schedule_download_failure_warns_and_keeps_dashboard_running(self):
        app_path = Path(__file__).with_name("nfl_edge_app.py")
        with (
            patch(
                "nflreadpy.load_schedules",
                side_effect=OSError("network unavailable"),
            ),
            patch(
                "nflreadpy.load_injuries",
                return_value=pl.DataFrame(),
            ),
        ):
            app = AppTest.from_file(str(app_path)).run()

        self.assertEqual(app.exception, [])
        self.assertTrue(
            any(
                "Prediction freshness cannot be verified" in warning.value
                for warning in app.warning
            )
        )
        self.assertGreater(len(app.dataframe), 0)


if __name__ == "__main__":
    unittest.main()
