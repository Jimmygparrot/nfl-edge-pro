import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

import nfl_backtest


class BacktestTests(unittest.TestCase):
    def test_prior_game_selector_excludes_same_day_future_and_incomplete_games(self):
        games = pd.DataFrame(
            {
                "game_id": ["prior", "same-day", "future", "incomplete"],
                "_game_date": [
                    date(2024, 9, 1),
                    date(2024, 9, 8),
                    date(2024, 9, 9),
                    date(2024, 9, 1),
                ],
                "home_score": [20, 20, 20, None],
                "away_score": [17, 17, 17, None],
            }
        )
        completed = games.loc[
            games["home_score"].notna() & games["away_score"].notna()
        ]

        prior = nfl_backtest._prior_completed_games(
            completed, date(2024, 9, 8)
        )

        self.assertEqual(prior["game_id"].tolist(), ["prior"])

    def test_backtest_builds_ratings_only_from_prior_date_games(self):
        schedules = pd.DataFrame(
            [
                ["g1", 2024, "REG", 1, "2024-09-01", "A", "B", 20, 10],
                ["g2", 2024, "REG", 2, "2024-09-08", "C", "A", 14, 21],
                ["g3", 2024, "REG", 3, "2024-09-15", "B", "C", 17, 24],
                ["g4", 2024, "REG", 4, "2024-09-22", "A", "C", 28, 14],
            ],
            columns=[
                "game_id",
                "season",
                "game_type",
                "week",
                "gameday",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
            ],
        )
        pbp = pd.DataFrame(
            [
                {
                    "game_id": game_id,
                    "season_type": "REG",
                    "play_type": "run",
                    "epa": 0.1,
                    "posteam": home,
                    "defteam": away,
                    "week": week,
                }
                for game_id, home, away, week in [
                    ("g1", "A", "B", 1),
                    ("g2", "C", "A", 2),
                    ("g3", "B", "C", 3),
                    ("g4", "A", "C", 4),
                ]
            ]
        )
        seen_game_ids = []

        def capture_prior_plays(plays):
            seen_game_ids.append(set(plays["game_id"]))
            return pd.DataFrame(
                {
                    "power": {"A": 1.0, "B": 0.0, "C": -1.0},
                    "rank": {"A": 1, "B": 2, "C": 3},
                    "off_epa": {"A": 0.1, "B": 0.0, "C": -0.1},
                    "def_epa": {"A": 0.0, "B": 0.0, "C": 0.0},
                }
            )

        with patch.object(
            nfl_backtest, "calculate_rankings", side_effect=capture_prior_plays
        ):
            results = nfl_backtest.backtest_season(
                2024, schedules, pbp, min_prior_team_games=1
            )

        self.assertEqual(results["game_id"].tolist(), ["g3", "g4"])
        self.assertEqual(seen_game_ids, [{"g1", "g2"}, {"g1", "g2", "g3"}])
        for index, result in enumerate(results.itertuples()):
            self.assertNotIn(result.game_id, seen_game_ids[index])

    def test_summary_excludes_ties_from_winner_accuracy_and_brier(self):
        results = pd.DataFrame(
            [
                {
                    "season": 2024,
                    "actual_winner": "H",
                    "predicted_winner": "H",
                    "actual_margin": 7,
                    "margin_error": 3.0,
                    "home_win_probability": 0.7,
                },
                {
                    "season": 2024,
                    "actual_winner": None,
                    "predicted_winner": "A",
                    "actual_margin": 0,
                    "margin_error": -2.0,
                    "home_win_probability": 0.4,
                },
            ]
        )

        summary = nfl_backtest.summarize_results(results)
        season = summary.loc[summary["season"] == 2024].iloc[0]

        self.assertEqual(season["games_predicted"], 2)
        self.assertEqual(season["ties_excluded"], 1)
        self.assertEqual(season["decisive_games"], 1)
        self.assertEqual(season["straight_up_accuracy"], 1.0)
        self.assertAlmostEqual(season["brier_score"], 0.09)
        self.assertAlmostEqual(season["margin_mae"], 2.5)
        self.assertAlmostEqual(season["margin_rmse"], (6.5) ** 0.5)


if __name__ == "__main__":
    unittest.main()
