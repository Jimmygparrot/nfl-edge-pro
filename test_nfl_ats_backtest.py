import unittest

import pandas as pd

import nfl_ats_backtest as ats


def _predictions(rows):
    return pd.DataFrame(
        rows,
        columns=(
            "season",
            "game_id",
            "gameday",
            "home_team",
            "away_team",
            "predicted_home_spread",
            "actual_margin",
        ),
    )


def _schedules(rows):
    return pd.DataFrame(
        rows,
        columns=(
            "season",
            "game_type",
            "gameday",
            "home_team",
            "away_team",
            "spread_line",
        ),
    )


class AtsBacktestTests(unittest.TestCase):
    def test_negative_home_line_and_model_side_are_graded_correctly(self):
        predictions = _predictions(
            [(2024, "g1", "2024-09-08", "KC", "BAL", -7.0, 10.0)]
        )
        # nflverse spread_line +3 means the home team is favored by 3.
        schedules = _schedules(
            [(2024, "REG", "2024-09-08", "KC", "BAL", 3.0)]
        )

        result, unmatched = ats.match_and_grade(predictions, schedules)

        self.assertEqual(unmatched, 0)
        row = result.iloc[0]
        self.assertEqual(row["closing_home_spread"], -3.0)
        self.assertEqual(row["actual_cover_margin"], 7.0)
        self.assertEqual(row["model_vs_market_spread"], -4.0)
        self.assertEqual(row["model_ats_pick"], "HOME")
        self.assertEqual(row["model_ats_result"], "WIN")
        self.assertEqual(row["market_ats_outcome"], "HOME_COVER")

    def test_aliases_and_unique_one_day_schedule_difference_match(self):
        predictions = _predictions(
            [(2020, "legacy", "2020-09-14", "LAR", "WSH", 1.0, -3.0)]
        )
        schedules = _schedules(
            [(2020, "REG", "2020-09-13", "LA", "WAS", -2.5)]
        )

        result, unmatched = ats.match_and_grade(predictions, schedules)

        self.assertEqual(len(result), 1)
        self.assertEqual(unmatched, 0)
        self.assertEqual(result.iloc[0]["closing_home_spread"], 2.5)
        self.assertEqual(result.iloc[0]["schedule_gameday"], "2020-09-13")

    def test_pushes_and_equal_spread_edge_are_not_wins(self):
        predictions = _predictions(
            [
                (2024, "push", "2024-09-08", "KC", "BAL", -4.0, 3.0),
                (2024, "pass", "2024-09-15", "BUF", "MIA", -2.0, 7.0),
            ]
        )
        schedules = _schedules(
            [
                (2024, "REG", "2024-09-08", "KC", "BAL", 3.0),
                (2024, "REG", "2024-09-15", "BUF", "MIA", 2.0),
            ]
        )

        results, unmatched = ats.match_and_grade(predictions, schedules)

        self.assertEqual(unmatched, 0)
        push = results.loc[results["game_id"] == "push"].iloc[0]
        self.assertEqual(push["market_ats_outcome"], "PUSH")
        self.assertEqual(push["model_ats_result"], "PUSH")
        no_bet = results.loc[results["game_id"] == "pass"].iloc[0]
        self.assertEqual(no_bet["model_ats_pick"], "PASS")
        self.assertEqual(no_bet["model_ats_result"], "NO_BET")

    def test_missing_spread_ambiguous_match_and_reused_schedule_are_excluded(self):
        predictions = _predictions(
            [
                (2024, "missing", "2024-09-01", "KC", "BUF", -2.0, 7.0),
                (2024, "ambiguous", "2024-09-08", "KC", "BAL", -2.0, 7.0),
                (2024, "first", "2024-09-15", "KC", "CIN", -2.0, 7.0),
                (2024, "second", "2024-09-15", "KC", "CIN", -2.0, 7.0),
            ]
        )
        schedules = _schedules(
            [
                (2024, "REG", "2024-09-01", "KC", "BUF", None),
                (2024, "REG", "2024-09-07", "KC", "BAL", 2.0),
                (2024, "REG", "2024-09-09", "KC", "BAL", 2.5),
                (2024, "REG", "2024-09-15", "KC", "CIN", 3.0),
            ]
        )

        results, unmatched = ats.match_and_grade(predictions, schedules)

        self.assertEqual(results["game_id"].tolist(), ["first"])
        self.assertEqual(unmatched, 3)

    def test_summary_reports_coverage_pushes_and_decision_accuracy(self):
        predictions = _predictions(
            [
                (2024, "one", "2024-09-08", "KC", "BAL", -7.0, 10.0),
                (2024, "two", "2024-09-15", "BUF", "MIA", 0.0, 5.0),
                (2024, "missing", "2024-09-22", "NE", "NYJ", -1.0, 4.0),
            ]
        )
        schedules = _schedules(
            [
                (2024, "REG", "2024-09-08", "KC", "BAL", 3.0),
                (2024, "REG", "2024-09-15", "BUF", "MIA", 2.0),
            ]
        )
        results, unmatched = ats.match_and_grade(predictions, schedules)
        summary = ats.summarize_results(
            ats._prepare_predictions(predictions),
            results,
            {2024: unmatched},
        )
        season = summary.loc[summary["season"] == 2024].iloc[0]

        self.assertEqual(season["predictions"], 3)
        self.assertEqual(season["matched_games"], 2)
        self.assertEqual(season["unmatched_games"], 1)
        self.assertAlmostEqual(season["spread_coverage"], 2 / 3)
        self.assertEqual(season["ats_wins"], 1)
        self.assertEqual(season["ats_losses"], 1)
        self.assertEqual(season["ats_pushes"], 0)
        self.assertEqual(season["ats_accuracy"], 0.5)
        self.assertFalse(season["profitability_assessed"])


if __name__ == "__main__":
    unittest.main()
