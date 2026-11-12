import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

from nfl_injuries import (
    QBAvailabilityContext,
    get_injury_adjustment,
    get_qb_availability,
)
from nfl_model import predict_game


class QuarterbackAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.kickoff = datetime(2026, 11, 8, 18, tzinfo=timezone.utc)
        chart_time = self.kickoff - timedelta(days=2)
        self.context = QBAvailabilityContext(
            depth_charts={
                "SEA": [
                    (chart_time, "QB1", 1),
                    (chart_time, "QB2", 2),
                    (chart_time, "QB3", 3),
                    (chart_time, "QB4", 4),
                    (self.kickoff + timedelta(minutes=1), "LATER", 1),
                ]
            },
            roster_statuses={
                (2026, 9, "SEA"): {
                    "QB1": "ACT",
                    "QB2": "ACT",
                    "QB3": "ACT",
                    "QB4": "ACT",
                }
            },
            roster_names={
                (2026, 9, "SEA"): {
                    "QB1": "Starter One",
                    "QB2": "Backup Two",
                    "QB3": "Backup Three",
                    "QB4": "Backup Four",
                }
            },
            reserve_descriptions={},
        )

    def availability(self, context=None, reports=None):
        return get_qb_availability(
            "SEA",
            9,
            2026,
            self.kickoff,
            context or self.context,
            reports or {},
        )

    def test_missing_injury_report_is_reported_as_unknown(self):
        result = self.availability()
        self.assertIn("no qualifying pre-kickoff game status", result["starter_status"])
        self.assertIn("no verified pre-kickoff game status", result["uncertainty"])
        self.assertEqual(result["starter_confidence"].split()[0], "Projected")
        self.assertEqual(result["replacement_name"], "Backup Two")
        self.assertIn("not an expected starter change", result["replacement_status"])

    def test_injured_reserve_qb_is_unavailable_and_backup_is_projected(self):
        key = (2026, 9, "SEA")
        self.context.roster_statuses[key]["QB1"] = "RES"
        self.context.reserve_descriptions[key] = {"QB1": "Injured Reserve"}

        result = self.availability()

        self.assertEqual(
            result["starter_status"],
            "Confirmed unavailable: Injured reserve status",
        )
        self.assertEqual(result["replacement_name"], "Backup Two")
        self.assertIn("Projected replacement for unavailable QB1", result["replacement_status"])

    def test_generic_reserve_is_not_mislabeled_as_injured_reserve(self):
        key = (2026, 9, "SEA")
        self.context.roster_statuses[key]["QB1"] = "RES"

        result = self.availability()

        self.assertIn("subtype unverified", result["starter_status"])
        self.assertNotIn("Injured reserve", result["starter_status"])

    def test_missing_roster_data_keeps_replacement_as_unverified_candidate(self):
        self.context.roster_statuses.clear()

        result = self.availability()

        self.assertIn("roster eligibility is unverified", result["uncertainty"])
        self.assertEqual(result["replacement_name"], "Backup Two")
        self.assertIn("Replacement roster eligibility is unverified", result["uncertainty"])

    def test_player_name_lookup_uses_gsis_id_when_weekly_roster_name_is_missing(self):
        self.context.roster_names.clear()
        self.context.player_names = {
            "QB1": "Starter One from player feed",
            "QB2": "Backup Two from player feed",
        }

        result = self.availability()

        self.assertEqual(result["starter_name"], "Starter One from player feed")
        self.assertEqual(result["replacement_name"], "Backup Two from player feed")

    def test_unknown_gsis_id_keeps_id_fallback(self):
        self.context.roster_names.clear()
        self.context.player_names.clear()

        result = self.availability()

        self.assertEqual(result["starter_name"], "QB1")
        self.assertEqual(result["replacement_name"], "QB2")

    def test_only_pre_kickoff_injury_reports_affect_availability(self):
        key = (2026, 9, "SEA")
        reports = {
            key: {
                "QB1": [
                    ("OUT", self.kickoff + timedelta(minutes=1)),
                ]
            }
        }
        result = self.availability(reports=reports)
        self.assertNotIn("Confirmed absent", result["starter_status"])
        self.assertIn("post-kickoff", result["starter_status"])

        reports[key]["QB1"].append(
            ("OUT", self.kickoff - timedelta(minutes=1))
        )
        result = self.availability(reports=reports)
        self.assertIn("Confirmed absent", result["starter_status"])

    def test_replacement_selection_skips_reported_out_and_reserve_qbs(self):
        key = (2026, 9, "SEA")
        self.context.roster_statuses[key]["QB1"] = "RES"
        self.context.roster_statuses[key]["QB2"] = "RES"
        reports = {
            key: {
                "QB3": [("OUT", self.kickoff - timedelta(hours=1))]
            }
        }
        self.context.roster_statuses[key]["QB3"] = "ACT"

        result = self.availability(reports=reports)
        self.assertEqual(result["replacement_id"], "QB4")
        self.assertEqual(result["replacement_name"], "Backup Four")

    def test_stale_roster_and_injury_feed_weeks_are_reported(self):
        self.context.roster_weeks = { (2026, "SEA"): {8} }
        self.context.roster_names.clear()
        self.context.roster_names[(2026, 8, "SEA")] = {
            "QB1": "Starter One",
            "QB2": "Backup Two",
        }
        report_weeks = {(2026, "SEA"): {8}}

        result = get_qb_availability(
            "SEA",
            9,
            2026,
            self.kickoff,
            self.context,
            {},
            report_weeks,
        )

        self.assertEqual(result["latest_roster_week"], "8")
        self.assertEqual(result["latest_injury_report_week"], "8")
        self.assertEqual(result["starter_name"], "Starter One")
        self.assertEqual(result["replacement_name"], "Backup Two")
        self.assertEqual(result["current_week_roster_coverage"], "Missing")
        self.assertEqual(result["current_week_injury_coverage"], "Missing")
        self.assertIn("Roster data is from Week 8", result["coverage_warning"])
        self.assertIn("Injury reports are from Week 8", result["coverage_warning"])

    def test_lower_ranked_qbs_surface_prior_reserve_and_out_records(self):
        self.context.roster_statuses[(2026, 8, "SEA")] = {
            "QB2": "RES",
            "QB3": "ACT",
        }
        self.context.roster_names[(2026, 8, "SEA")] = {
            "QB2": "Backup Two",
            "QB3": "Backup Three",
        }
        self.context.reserve_descriptions[(2026, 8, "SEA")] = {
            "QB2": "R01",
        }
        reports = {
            (2026, 8, "SEA"): {
                "QB3": [("OUT", self.kickoff - timedelta(days=8))]
            }
        }

        result = self.availability(reports=reports)

        flags = result["prior_qb_availability_flags"]
        self.assertIn("Backup Two", flags)
        self.assertIn("Week 8 roster RES", flags)
        self.assertIn("Backup Three", flags)
        self.assertIn("Week 8 injury report OUT", flags)
        self.assertNotIn("Starter One", flags)

    def test_prior_inactive_roster_status_is_not_reported_as_reserve(self):
        self.context.roster_statuses[(2026, 8, "SEA")] = {"QB2": "INA"}
        self.context.roster_names[(2026, 8, "SEA")] = {
            "QB2": "Backup Two",
        }

        result = self.availability()

        self.assertEqual(result["prior_qb_availability_flags"], "")

    def test_depth_chart_ids_matching_latest_team_roster_have_no_mismatch(self):
        self.context.roster_weeks = {(2026, "SEA"): {9}}
        self.context.roster_player_teams = {
            (2026, 9, "QB1"): {"SEA"},
            (2026, 9, "QB2"): {"SEA"},
            (2026, 9, "QB3"): {"SEA"},
            (2026, 9, "QB4"): {"SEA"},
        }

        result = self.availability()

        self.assertEqual(result["team_assignment_warnings"], "")

    def test_depth_chart_player_rostered_elsewhere_is_confirmed_mismatch(self):
        self.context.roster_weeks = {(2026, "SEA"): {9}}
        self.context.roster_player_teams = {
            (2026, 9, "QB1"): {"LV"},
            (2026, 9, "QB2"): {"SEA"},
            (2026, 9, "QB3"): {"SEA"},
            (2026, 9, "QB4"): {"SEA"},
        }

        result = self.availability()

        self.assertIn("Confirmed team mismatch", result["team_assignment_warnings"])
        self.assertIn("Starter One", result["team_assignment_warnings"])
        self.assertIn("rostered by LV in Week 9", result["team_assignment_warnings"])

    def test_missing_roster_data_is_not_labeled_as_confirmed_mismatch(self):
        self.context.roster_weeks.clear()
        self.context.roster_player_teams.clear()

        result = self.availability()

        self.assertIn("no roster data", result["team_assignment_warnings"])
        self.assertNotIn("Confirmed team mismatch", result["team_assignment_warnings"])

    def test_stale_roster_data_is_distinguished_from_current_mismatch(self):
        self.context.roster_weeks = {(2026, "SEA"): {8}}
        self.context.roster_player_teams = {
            (2026, 8, "QB1"): {"LV"},
        }

        result = self.availability()

        self.assertIn("roster data is stale", result["team_assignment_warnings"])
        self.assertIn("latest Week 8", result["team_assignment_warnings"])
        self.assertNotIn("Confirmed team mismatch", result["team_assignment_warnings"])

    def test_diagnostics_do_not_add_a_second_qb_penalty_or_change_predictions(self):
        reports = {(2026, 9, "SEA"): {"QB1": ("QB", "OUT")}}
        fixed_adjustment = get_injury_adjustment(
            "SEA",
            9,
            reports,
            snap_history={},
            gsis_to_pfr={},
        )
        self.assertEqual(fixed_adjustment, -5.0)

        teams = pd.DataFrame(
            {
                "power": {"SEA": 2.0, "ARI": -1.0},
                "rank": {"SEA": 1, "ARI": 2},
                "off_epa": {"SEA": 0.1, "ARI": 0.0},
                "def_epa": {"SEA": 0.0, "ARI": 0.0},
            }
        )
        common = (
            "SEA",
            "ARI",
            teams,
            self.kickoff,
            9,
            reports,
            {},
            {},
            None,
        )
        with patch(
            "nfl_model.get_game_weather",
            return_value={"status": "Unavailable", "adjustment": 0.0},
        ):
            without_diagnostics = predict_game(*common)
            with_diagnostics = predict_game(
                *common,
                self.context,
                {
                    (2026, 9, "SEA"): {
                        "QB1": [("OUT", self.kickoff - timedelta(hours=1))]
                    }
                },
            )

        self.assertEqual(without_diagnostics["margin"], with_diagnostics["margin"])
        self.assertEqual(
            without_diagnostics["home_win"],
            with_diagnostics["home_win"],
        )
        self.assertEqual(
            with_diagnostics["home_injury_adjustment"],
            without_diagnostics["home_injury_adjustment"],
        )


if __name__ == "__main__":
    unittest.main()
