import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import nfl_preseason_prior as experiment


class PreseasonPriorTests(unittest.TestCase):
    def test_franchise_aliases_keep_relocated_teams_continuous(self):
        self.assertEqual(experiment._franchise("STL"), experiment._franchise("LA"))
        self.assertEqual(experiment._franchise("OAK"), experiment._franchise("LV"))
        self.assertEqual(experiment._franchise("SD"), experiment._franchise("LAC"))

    def test_prior_weight_decays_as_current_games_accumulate(self):
        params = experiment.BlendParameters(0.5, 4.0)
        self.assertAlmostEqual(experiment._prior_share(0, params), 0.5)
        self.assertAlmostEqual(experiment._prior_share(4, params), 0.25)
        self.assertAlmostEqual(experiment._prior_share(8, params), 0.125)

    def test_absent_current_rating_uses_prior_rating_for_early_weeks(self):
        params = experiment.BlendParameters(0.5, 4.0)
        power, share = experiment._blended_power(3.0, np.nan, 0, params)
        self.assertEqual(power, 3.0)
        self.assertEqual(share, 1.0)

    def test_feature_extraction_uses_only_prior_date_games(self):
        schedules = pd.DataFrame(
            [
                ["g1", 2024, "REG", 1, "2024-09-01", "A", "B", 20, 10, 0.0],
                ["g2", 2024, "REG", 2, "2024-09-08", "C", "A", 14, 21, 0.0],
                ["g3", 2024, "REG", 3, "2024-09-15", "B", "C", 17, 24, 0.0],
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
                "spread_line",
            ],
        )
        plays = pd.DataFrame(
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
                ]
            ]
        )
        rated_ids = []

        def capture(prior_plays):
            rated_ids.append(set(prior_plays["game_id"]))
            return pd.DataFrame(
                {
                    "power": {"A": 1.0, "B": 0.0, "C": -1.0},
                    "off_epa": {"A": 0.1, "B": 0.0, "C": -0.1},
                }
            )

        prior = {"A": 1.0, "B": 0.0, "C": -1.0}
        with patch.object(experiment, "calculate_rankings", side_effect=capture):
            features = experiment.extract_season_features(
                2024, schedules, plays, prior, min_prior_team_games=1
            )

        self.assertEqual(features["game_id"].tolist(), ["g1", "g2", "g3"])
        self.assertEqual(rated_ids, [{"g1"}, {"g1", "g2"}])
        self.assertTrue(np.isnan(features.loc[0, "home_current_power"]))
        self.assertFalse(features.loc[0, "baseline_eligible"])
        for game_id, history in zip(features["game_id"].iloc[1:], rated_ids):
            self.assertNotIn(game_id, history)

    def test_blended_methods_compared_only_on_shared_eligible_games(self):
        base = pd.DataFrame(
            {
                "game_id": ["a", "b"],
                "absolute_margin_error": [7.0, 10.0],
                "actual_winner": ["H", "A"],
                "predicted_winner": ["H", "H"],
                "home_win_probability": [0.6, 0.7],
                "actual_margin": [3.0, -4.0],
                "model_ats_result": ["WIN", "LOSS"],
            }
        )
        blend = pd.DataFrame(
            {
                "game_id": ["a", "b"],
                "absolute_margin_error": [6.0, 12.0],
                "actual_winner": ["H", "A"],
                "predicted_winner": ["H", "A"],
                "home_win_probability": [0.65, 0.3],
                "actual_margin": [3.0, -4.0],
                "model_ats_result": ["WIN", "WIN"],
            }
        )
        comparison = experiment._paired_comparison(2024, base, blend, "same_games")
        self.assertEqual(comparison["paired_games"], 2)
        self.assertAlmostEqual(comparison["blend_minus_baseline_margin_mae"], 0.5)
        self.assertEqual(comparison["baseline_straight_up_accuracy"], 0.5)
        self.assertEqual(comparison["blend_straight_up_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
