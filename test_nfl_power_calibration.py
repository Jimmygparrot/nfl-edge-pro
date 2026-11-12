import unittest
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

import nfl_power_calibration as calibration


class PowerCalibrationTests(unittest.TestCase):
    def test_candidate_grid_contains_baseline_and_has_expected_size(self):
        self.assertIn(calibration.BASELINE, calibration.CANDIDATES)
        self.assertEqual(len(calibration.CANDIDATES), 240)

    def test_feature_extraction_only_rates_from_prior_dates(self):
        schedules = pd.DataFrame(
            [
                ["g1", 2024, "REG", 1, "2024-09-01", "A", "B", 20, 10, 0.0],
                ["g2", 2024, "REG", 2, "2024-09-08", "C", "A", 14, 21, 0.0],
                ["g3", 2024, "REG", 3, "2024-09-15", "B", "C", 17, 24, 0.0],
                ["g4", 2024, "REG", 4, "2024-09-22", "A", "C", 28, 14, 0.0],
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
                    ("g4", "A", "C", 4),
                ]
            ]
        )
        rated_ids = []

        def capture_prior(eligible_plays):
            rated_ids.append(set(eligible_plays["game_id"]))
            return pd.DataFrame(
                {
                    "net_epa": {"A": 0.2, "B": 0.0, "C": -0.2},
                    "sos": {"A": 0.1, "B": 0.0, "C": -0.1},
                    "recent_off_epa": {"A": 0.2, "B": 0.0, "C": -0.2},
                    "off_epa": {"A": 0.1, "B": 0.0, "C": -0.1},
                }
            )

        with patch.object(
            calibration, "calculate_rankings", side_effect=capture_prior
        ):
            result = calibration._extract_game_features(
                2024, schedules, plays, min_prior_team_games=1
            )

        self.assertEqual(result["game_id"].tolist(), ["g3", "g4"])
        self.assertEqual(rated_ids, [{"g1", "g2"}, {"g1", "g2", "g3"}])
        for target_game, history in zip(result["game_id"], rated_ids):
            self.assertNotIn(target_game, history)

    def test_baseline_weights_recreate_current_margin_formula(self):
        features = pd.DataFrame(
            {
                "home_net_epa": [0.22, -0.03],
                "away_net_epa": [-0.08, 0.10],
                "home_sos": [0.05, -0.01],
                "away_sos": [-0.02, 0.03],
                "home_recent_delta": [0.04, -0.02],
                "away_recent_delta": [-0.01, 0.01],
            }
        )
        current_formula = (
            (features["home_net_epa"] - features["away_net_epa"]) * 20
            + (features["home_sos"] - features["away_sos"]) * 10
            + (features["home_recent_delta"] - features["away_recent_delta"]) * 5
            + 1.5
        ).to_numpy()
        np.testing.assert_allclose(
            calibration._predict_margins(features, calibration.BASELINE),
            current_formula,
        )

    def test_market_spread_is_used_for_ats_grading_not_prediction_margin(self):
        # A home-favorite spread_line of +3 means conventional signed home line
        # -3. With a home margin prediction of +7, the model's side is home.
        pick, result = calibration._grade_ats(
            np.array([7.0]),
            np.array([10.0]),
            np.array([3.0]),
        )
        self.assertEqual(pick.tolist(), ["HOME"])
        self.assertEqual(result.tolist(), ["WIN"])

    def test_selection_only_scores_seasons_passed_as_training(self):
        candidate = calibration.CANDIDATES[0]
        features_2020 = pd.DataFrame(
            {
                "home_net_epa": [0.0],
                "away_net_epa": [0.0],
                "home_sos": [0.0],
                "away_sos": [0.0],
                "home_recent_delta": [0.0],
                "away_recent_delta": [0.0],
                "actual_margin": [0.0],
            }
        )
        features_2021 = features_2020.assign(actual_margin=1000.0)
        seen = []

        def score(seasons):
            seen.extend(seasons)
            return {candidate.key: 1.0}, sum(len(frame) for frame in seasons.values())

        with patch.object(calibration, "_candidate_errors", side_effect=score):
            calibration._select_weights(
                {2020: features_2020, 2021: features_2021},
                (2020,),
            )
        self.assertEqual(seen, [2020])


if __name__ == "__main__":
    unittest.main()
