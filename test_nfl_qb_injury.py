import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import polars as pl

from nfl_injuries import (
    QBContext,
    _qb_replacement_adjustment,
    _smoothed_qb_rates,
    get_injury_adjustment,
    get_point_in_time_injury_adjustment,
    load_injury_report_data,
    load_injury_report_history,
)


class QuarterbackReplacementTests(unittest.TestCase):
    def setUp(self):
        self.kickoff = datetime(2026, 11, 8, 18, tzinfo=timezone.utc)
        prior_game = self.kickoff - timedelta(days=7)
        self.context = QBContext(
            depth_charts={
                "SEA": [
                    (prior_game, "QB1", 1),
                    (prior_game, "QB2", 2),
                    (prior_game, "QB3", 3),
                    (self.kickoff + timedelta(minutes=1), "LATER", 1),
                ]
            },
            roster_players={
                (2026, 9, "SEA"): {"QB1", "QB2", "QB3", "LATER"}
            },
            passing_stats=[
                (prior_game, "game-1", "SEA", "QB1", 40.0, 40),
                (prior_game, "game-1", "SEA", "QB2", 20.0, 40),
                (prior_game, "game-1", "SEA", "QB3", -40.0, 40),
                (
                    self.kickoff + timedelta(minutes=1),
                    "future-game",
                    "SEA",
                    "QB1",
                    10000.0,
                    1,
                ),
            ],
        )
        self.snap_history = {("SEA", "PFR1"): [(8, 1.0, 0.0)]}
        self.gsis_to_pfr = {"QB1": "PFR1"}

    def test_missing_data_retains_fixed_qb_penalty(self):
        reports = {(2026, 9, "SEA"): {"QB1": ("QB", "OUT")}}
        adjustment = get_injury_adjustment(
            "SEA",
            9,
            reports,
            snap_history=self.snap_history,
            gsis_to_pfr=self.gsis_to_pfr,
            qb_adjustment_scale=1.0,
        )
        self.assertEqual(adjustment, -5.0)

    def test_post_kickoff_depth_and_stats_are_ignored(self):
        reports = {(2026, 9, "SEA"): {"QB1": ("QB", "OUT")}}
        before = _qb_replacement_adjustment(
            "SEA",
            9,
            2026,
            self.kickoff,
            reports,
            self.context,
            self.snap_history,
            self.gsis_to_pfr,
            0.1,
        )
        self.context.passing_stats = [
            row for row in self.context.passing_stats
            if row[0] < self.kickoff
        ]
        self.context.depth_charts["SEA"] = [
            row for row in self.context.depth_charts["SEA"]
            if row[0] <= self.kickoff
        ]
        after = _qb_replacement_adjustment(
            "SEA",
            9,
            2026,
            self.kickoff,
            reports,
            self.context,
            self.snap_history,
            self.gsis_to_pfr,
            0.1,
        )
        self.assertEqual(before, after)

    def test_out_qb2_is_skipped_for_next_available_replacement(self):
        reports = {
            (2026, 9, "SEA"): {
                "QB1": ("QB", "OUT"),
                "QB2": ("QB", "OUT"),
            }
        }
        result = _qb_replacement_adjustment(
            "SEA",
            9,
            2026,
            self.kickoff,
            reports,
            self.context,
            self.snap_history,
            self.gsis_to_pfr,
            1.0,
        )
        self.assertIsNotNone(result)
        _, handled_players = result
        self.assertEqual(handled_players, {"QB1", "QB2", "QB3"})

    def test_replacement_adjustment_replaces_instead_of_stacking_flat_qb_penalties(self):
        reports = {
            (2026, 9, "SEA"): {
                "QB1": ("QB", "OUT"),
                "QB2": ("QB", "OUT"),
            }
        }
        self.snap_history[("SEA", "PFR2")] = [(8, 1.0, 0.0)]
        self.gsis_to_pfr["QB2"] = "PFR2"
        adjustment = get_injury_adjustment(
            "SEA",
            9,
            reports,
            snap_history=self.snap_history,
            gsis_to_pfr=self.gsis_to_pfr,
            qb_context=self.context,
            kickoff=self.kickoff,
            qb_adjustment_scale=1.0,
        )
        self.assertEqual(adjustment, -5.0)

    def test_questionable_status_and_snap_share_are_applied_once(self):
        reports = {(2026, 9, "SEA"): {"QB1": ("QB", "QUESTIONABLE")}}
        self.snap_history[("SEA", "PFR1")] = [(8, 0.5, 0.0)]
        result = _qb_replacement_adjustment(
            "SEA",
            9,
            2026,
            self.kickoff,
            reports,
            self.context,
            self.snap_history,
            self.gsis_to_pfr,
            0.01,
        )
        self.assertIsNotNone(result)
        adjustment, _ = result
        self.assertLess(adjustment, 0)
        self.assertGreaterEqual(adjustment, -1.25)

    def test_small_samples_are_shrunk_toward_league_average(self):
        rates = _smoothed_qb_rates(
            self.context,
            "SEA",
            self.kickoff,
        )
        self.assertIsNotNone(rates)
        player_rates, _ = rates
        league_rate = 20.0 / 120.0
        self.assertGreater(player_rates["QB1"], league_rate)
        self.assertLess(player_rates["QB1"], 1.0)
        self.assertGreater(player_rates["QB3"], -1.0)
        self.assertLess(player_rates["QB3"], league_rate)

    def test_non_qb_snap_penalty_is_unchanged(self):
        reports = {(2026, 9, "SEA"): {"LB1": ("LB", "OUT")}}
        snap_history = {("SEA", "PFR-LB"): [(8, 0.0, 0.5)]}
        adjustment = get_injury_adjustment(
            "SEA",
            9,
            reports,
            snap_history=snap_history,
            gsis_to_pfr={"LB1": "PFR-LB"},
            qb_adjustment_scale=1.0,
        )
        self.assertEqual(adjustment, -0.25)

    def test_uncalibrated_scale_uses_existing_penalty(self):
        reports = {(2026, 9, "SEA"): {"QB1": ("QB", "OUT")}}
        adjustment = get_injury_adjustment(
            "SEA",
            9,
            reports,
            snap_history=self.snap_history,
            gsis_to_pfr=self.gsis_to_pfr,
            qb_context=self.context,
            kickoff=self.kickoff,
        )
        self.assertEqual(adjustment, -5.0)


class PointInTimeInjuryAdjustmentTests(unittest.TestCase):
    def setUp(self):
        self.kickoff = datetime(2024, 9, 6, 0, 20, tzinfo=timezone.utc)

    def adjustment(self, rows, kickoff=None):
        return get_point_in_time_injury_adjustment(
            "BAL",
            1,
            2024,
            kickoff or self.kickoff,
            {(2024, 1, "BAL"): {"PLAYER1": rows}},
        )

    def test_latest_pre_kickoff_status_wins_regardless_of_severity(self):
        result = self.adjustment(
            [
                ("LB", "OUT", self.kickoff - timedelta(hours=2)),
                ("LB", "QUESTIONABLE", self.kickoff - timedelta(hours=1)),
            ]
        )

        self.assertEqual(result.adjustment, -0.25)
        self.assertEqual(result.coverage, "verified")

    def test_post_kickoff_update_is_excluded_but_prior_status_is_used(self):
        result = self.adjustment(
            [
                ("LB", "OUT", self.kickoff - timedelta(hours=1)),
                ("LB", "QUESTIONABLE", self.kickoff + timedelta(minutes=1)),
            ]
        )

        self.assertEqual(result.adjustment, -0.5)
        self.assertEqual(result.coverage, "verified")
        self.assertTrue(
            any("excluded post-kickoff" in warning for warning in result.warnings)
        )

    def test_post_kickoff_only_record_is_unknown_not_healthy(self):
        result = self.adjustment(
            [("LB", "OUT", self.kickoff + timedelta(minutes=1))]
        )

        self.assertEqual(result.adjustment, 0.0)
        self.assertEqual(result.coverage, "unknown")
        self.assertTrue(
            any("coverage is unknown" in warning for warning in result.warnings)
        )

    def test_unknown_player_makes_coverage_partial(self):
        result = get_point_in_time_injury_adjustment(
            "BAL",
            1,
            2024,
            self.kickoff,
            {
                (2024, 1, "BAL"): {
                    "KNOWN": [
                        ("LB", "OUT", self.kickoff - timedelta(hours=1))
                    ],
                    "UNKNOWN": [("LB", "OUT", None)],
                }
            },
        )

        self.assertEqual(result.adjustment, -0.5)
        self.assertEqual(result.coverage, "partial")

    def test_missing_team_history_is_unknown(self):
        result = get_point_in_time_injury_adjustment(
            "BAL",
            1,
            2024,
            self.kickoff,
            {},
        )

        self.assertEqual(result.adjustment, 0.0)
        self.assertEqual(result.coverage, "unknown")
        self.assertTrue(
            any("coverage is unknown" in warning for warning in result.warnings)
        )

    def test_missing_or_naive_timestamp_is_unknown(self):
        for rows in (
            [("LB", "OUT", None)],
            [("LB", "OUT", datetime(2024, 9, 5, 23, 0))],
            [
                ("LB", "OUT", self.kickoff - timedelta(hours=2)),
                ("LB", "QUESTIONABLE", None),
            ],
        ):
            with self.subTest(rows=rows):
                result = self.adjustment(rows)

                self.assertEqual(result.adjustment, 0.0)
                self.assertEqual(result.coverage, "unknown")

    def test_conflicting_latest_statuses_are_not_ranked_by_severity(self):
        modified = self.kickoff - timedelta(minutes=1)
        result = self.adjustment(
            [
                ("LB", "OUT", modified),
                ("LB", "QUESTIONABLE", modified),
            ]
        )

        self.assertEqual(result.adjustment, 0.0)
        self.assertEqual(result.coverage, "unknown")
        self.assertTrue(
            any("conflicting statuses" in warning for warning in result.warnings)
        )

    def test_2024_baltimore_post_kickoff_record_does_not_change_legacy_baseline(self):
        injury_feed = pl.DataFrame(
            {
                "season": [2024],
                "game_type": ["REG"],
                "team": ["BAL"],
                "week": [1],
                "gsis_id": ["00-0031360"],
                "full_name": ["Kyle Van Noy"],
                "position": ["LB"],
                "report_status": ["Out"],
                "date_modified": [
                    datetime(2024, 9, 6, 3, 11, 40, tzinfo=timezone.utc)
                ],
            }
        )
        with patch(
            "nfl_injuries.nfl.load_injuries",
            return_value=injury_feed,
        ):
            legacy_reports, _, _ = load_injury_report_data(2024)
            report_history = load_injury_report_history(2024)

        legacy_adjustment = get_injury_adjustment(
            "BAL",
            1,
            legacy_reports,
            season=2024,
        )
        point_in_time = get_point_in_time_injury_adjustment(
            "BAL",
            1,
            2024,
            self.kickoff,
            report_history,
        )

        self.assertEqual(legacy_adjustment, 0.0)
        self.assertEqual(point_in_time.adjustment, 0.0)
        self.assertEqual(point_in_time.coverage, "unknown")
        self.assertIn(
            "00-0031360",
            report_history[(2024, 1, "BAL")],
        )
        self.assertTrue(
            any(
                "excluded post-kickoff" in warning
                for warning in point_in_time.warnings
            )
        )

    def test_feed_without_date_modified_column_is_unknown(self):
        injury_feed = pl.DataFrame(
            {
                "season": [2025],
                "season_type": ["REG"],
                "game_type": ["REG"],
                "team": ["BAL"],
                "week": [1],
                "gsis_id": ["PLAYER1"],
                "full_name": ["Player One"],
                "position": ["LB"],
                "report_status": ["Out"],
            }
        )
        with patch(
            "nfl_injuries.nfl.load_injuries",
            return_value=injury_feed,
        ):
            report_history = load_injury_report_history(2025)

        result = get_point_in_time_injury_adjustment(
            "BAL",
            1,
            2025,
            self.kickoff,
            report_history,
        )

        self.assertEqual(result.adjustment, 0.0)
        self.assertEqual(result.coverage, "unknown")
        self.assertTrue(
            any("timestamp is unavailable" in warning for warning in result.warnings)
        )


if __name__ == "__main__":
    unittest.main()
