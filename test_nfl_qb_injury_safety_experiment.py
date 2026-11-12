import unittest
from datetime import datetime, timedelta, timezone

from nfl_qb_injury_safety_experiment import (
    QBInjuryInput,
    compare_qb_injury_penalties,
)


class QBInjurySafetyExperimentTests(unittest.TestCase):
    def setUp(self):
        self.kickoff = datetime(2026, 11, 8, 18, tzinfo=timezone.utc)
        self.source_time = self.kickoff - timedelta(hours=2)

    def compare_one(
        self,
        status="OUT",
        snap_share=1.0,
        role="starter",
        source_time=None,
    ):
        injury = QBInjuryInput(
            "QB1",
            status,
            snap_share,
            role,
            self.source_time if source_time is None else source_time,
        )
        return compare_qb_injury_penalties([injury], self.kickoff)[0]

    def test_missing_snaps_retain_full_fixed_penalty(self):
        result = self.compare_one(snap_share=None)
        self.assertEqual(result.fixed_penalty, -5.0)
        self.assertEqual(result.experimental_penalty, -5.0)

    def test_verified_backup_penalty_is_suppressed_only_experimentally(self):
        result = self.compare_one(role="backup", status="OUT")
        self.assertEqual(result.role_classification, "verified_backup")
        self.assertEqual(result.fixed_penalty, -5.0)
        self.assertEqual(result.experimental_penalty, 0.0)

    def test_verified_starter_preserves_existing_status_and_snap_calculation(self):
        result = self.compare_one(
            status="QUESTIONABLE",
            snap_share=0.6,
        )
        self.assertEqual(result.role_classification, "verified_starter")
        self.assertAlmostEqual(result.fixed_penalty, -1.5)
        self.assertAlmostEqual(result.experimental_penalty, -1.5)

    def test_ambiguous_role_is_reported_as_unknown_not_backup(self):
        result = self.compare_one(role="starter_or_backup")
        self.assertEqual(result.role_classification, "unknown")
        self.assertEqual(result.experimental_penalty, result.fixed_penalty)
        self.assertIn("ambiguous", result.role_evidence_note)

    def test_multiple_injured_qbs_are_compared_and_summed_independently(self):
        injuries = [
            QBInjuryInput("QB1", "OUT", 1.0, "starter", self.source_time),
            QBInjuryInput("QB2", "DOUBTFUL", 0.8, "backup", self.source_time),
        ]
        results = compare_qb_injury_penalties(injuries, self.kickoff)
        self.assertEqual([row.player_id for row in results], ["QB1", "QB2"])
        self.assertAlmostEqual(sum(row.fixed_penalty for row in results), -8.0)
        self.assertAlmostEqual(
            sum(row.experimental_penalty for row in results),
            -5.0,
        )

    def test_role_timestamp_must_be_strictly_before_kickoff(self):
        at_kickoff = self.compare_one(source_time=self.kickoff)
        after_kickoff = self.compare_one(
            source_time=self.kickoff + timedelta(seconds=1)
        )
        before_kickoff = self.compare_one(
            source_time=self.kickoff - timedelta(seconds=1)
        )
        self.assertEqual(at_kickoff.role_classification, "unknown")
        self.assertEqual(after_kickoff.role_classification, "unknown")
        self.assertEqual(before_kickoff.role_classification, "verified_starter")

    def test_naive_role_timestamp_cannot_verify_role(self):
        naive_timestamp = datetime(2026, 11, 8, 16)
        result = self.compare_one(source_time=naive_timestamp)
        self.assertEqual(result.role_classification, "unknown")
        self.assertIn("no timezone", result.role_evidence_note)

    def test_missing_role_timestamp_cannot_verify_backup(self):
        injury = QBInjuryInput("QB1", "OUT", 1.0, "backup", None)
        result = compare_qb_injury_penalties([injury], self.kickoff)[0]
        self.assertEqual(result.role_classification, "unknown")
        self.assertEqual(result.experimental_penalty, result.fixed_penalty)

    def test_duplicate_player_rows_are_rejected(self):
        injury = QBInjuryInput("QB1", "OUT", 1.0, "starter", self.source_time)
        with self.assertRaisesRegex(ValueError, "at most one row per player"):
            compare_qb_injury_penalties([injury, injury], self.kickoff)

    def test_naive_kickoff_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "kickoff must be timezone-aware"):
            compare_qb_injury_penalties(
                [],
                datetime(2026, 11, 8, 18),
            )


if __name__ == "__main__":
    unittest.main()
