import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from nfl_injury_adjustment_comparison import (
    compare_historical_games,
    summarize_comparison,
)


class InjuryAdjustmentComparisonTests(unittest.TestCase):
    def setUp(self):
        self.kickoff = datetime(2024, 9, 8, 17, tzinfo=timezone.utc)
        self.schedule_row = {
            "game_id": "2024_01_AWAY_HOME",
            "season": 2024,
            "week": 1,
            "game_type": "REG",
            "gameday": "2024-09-08",
            "gametime": "13:00",
            "home_team": "HOME",
            "away_team": "AWAY",
            "home_score": 24,
            "away_score": 17,
        }

    def test_comparison_computes_team_adjustments_and_home_margin_delta(self):
        legacy_reports = {
            (2024, 1, "HOME"): {"H1": ("LB", "OUT")},
            (2024, 1, "AWAY"): {"A1": ("LB", "QUESTIONABLE")},
        }
        report_history = {
            (2024, 1, "HOME"): {
                "H1": [
                    ("LB", "OUT", self.kickoff - timedelta(hours=2)),
                    (
                        "LB",
                        "QUESTIONABLE",
                        self.kickoff - timedelta(hours=1),
                    ),
                ]
            },
            (2024, 1, "AWAY"): {
                "A1": [
                    ("LB", "QUESTIONABLE", self.kickoff - timedelta(hours=1))
                ]
            },
        }

        result = compare_historical_games(
            [self.schedule_row],
            2024,
            legacy_reports,
            report_history,
        )

        self.assertEqual(len(result), 1)
        game = result.iloc[0]
        self.assertEqual(game["legacy_home_injury_adjustment"], -0.5)
        self.assertEqual(game["point_in_time_home_injury_adjustment"], -0.25)
        self.assertEqual(game["legacy_away_injury_adjustment"], -0.25)
        self.assertEqual(game["point_in_time_away_injury_adjustment"], -0.25)
        self.assertEqual(game["legacy_injury_margin_effect"], -0.25)
        self.assertEqual(game["point_in_time_injury_margin_effect"], 0.0)
        self.assertEqual(
            game["home_legacy_coverage"],
            "weighted_statuses_present",
        )
        self.assertEqual(
            game["predicted_margin_difference_point_in_time_minus_legacy"],
            0.25,
        )
        self.assertEqual(game["home_point_in_time_coverage"], "verified")
        self.assertEqual(game["away_point_in_time_coverage"], "verified")

    def test_comparison_records_missing_and_post_kickoff_coverage_warnings(self):
        legacy_reports = {
            (2024, 1, "HOME"): {"H1": ("LB", "OUT")},
            (2024, 1, "AWAY"): {"A1": ("LB", "OUT")},
        }
        report_history = {
            (2024, 1, "HOME"): {"H1": [("LB", "OUT", None)]},
            (2024, 1, "AWAY"): {
                "A1": [
                    ("LB", "OUT", self.kickoff + timedelta(minutes=1))
                ]
            },
            (2024, 2, "OTHER"): {
                "O1": [
                    ("LB", None, self.kickoff - timedelta(hours=1))
                ]
            },
        }
        no_report_game = {
            **self.schedule_row,
            "game_id": "2024_02_OTHER_MISSING",
            "week": 2,
            "home_team": "OTHER",
            "away_team": "MISSING",
        }
        no_kickoff_game = {
            **self.schedule_row,
            "game_id": "2024_03_NO_KICKOFF",
            "week": 3,
            "gametime": None,
        }

        result = compare_historical_games(
            [self.schedule_row, no_report_game, no_kickoff_game],
            2024,
            legacy_reports,
            report_history,
        )

        self.assertEqual(len(result), 2)
        game = result.loc[result["game_id"] == self.schedule_row["game_id"]].iloc[0]
        self.assertEqual(game["home_point_in_time_coverage"], "unknown")
        self.assertEqual(
            game["home_legacy_coverage"],
            "weighted_statuses_present",
        )
        self.assertIn(
            "timestamp is unavailable",
            game["home_point_in_time_warnings"],
        )
        self.assertEqual(game["away_point_in_time_coverage"], "unknown")
        self.assertIn(
            "excluded post-kickoff",
            game["away_point_in_time_warnings"],
        )
        no_report = result.loc[
            result["game_id"] == "2024_02_OTHER_MISSING"
        ].iloc[0]
        self.assertIn(
            "latest pre-kickoff status is not a supported",
            no_report["home_point_in_time_warnings"],
        )
        self.assertIn(
            "No injury-report records",
            no_report["away_point_in_time_warnings"],
        )

        self.assertEqual(
            no_report["home_legacy_coverage"],
            "feed_rows_without_legacy_statuses",
        )
        self.assertIn(
            "zero is not confirmation of health",
            no_report["home_legacy_warnings"],
        )

        summary = summarize_comparison(result)
        self.assertEqual(summary["games_compared"], 2)
        self.assertEqual(summary["team_side_coverage"]["unknown"], 4)
        self.assertEqual(
            summary["diagnostic_warning_counts"][
                "missing_or_unverifiable_timestamps"
            ],
            1,
        )
        self.assertEqual(
            summary["diagnostic_warning_counts"][
                "post_kickoff_updates_excluded"
            ],
            1,
        )
        self.assertGreater(
            summary["diagnostic_warning_counts"]["incomplete_injury_data"],
            0,
        )

    def test_summary_orders_largest_absolute_margin_differences(self):
        comparison = compare_historical_games(
            [self.schedule_row],
            2024,
            {(2024, 1, "HOME"): {"H1": ("LB", "OUT")}},
            {
                (2024, 1, "HOME"): {
                    "H1": [
                        (
                            "LB",
                            "QUESTIONABLE",
                            self.kickoff - timedelta(hours=1),
                        )
                    ]
                }
            },
        )
        larger_difference = comparison.copy()
        larger_difference.loc[
            0, "game_id"
        ] = "2024_02_LARGER_DIFFERENCE"
        larger_difference.loc[
            0, "predicted_margin_difference_point_in_time_minus_legacy"
        ] = -1.5
        comparison = pd.concat(
            [comparison, larger_difference],
            ignore_index=True,
        )

        summary = summarize_comparison(comparison, largest_count=1)

        self.assertEqual(
            summary["largest_margin_differences"][0]["game_id"],
            "2024_02_LARGER_DIFFERENCE",
        )
        self.assertEqual(
            summary["largest_margin_differences"][0][
                "absolute_margin_difference"
            ],
            1.5,
        )


if __name__ == "__main__":
    unittest.main()
