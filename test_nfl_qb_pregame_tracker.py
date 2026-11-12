import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nfl_qb_pregame_tracker import (
    DuplicateObservationError,
    Observation,
    add_observation,
    list_observations,
)


class QBPregameTrackerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "tracker.sqlite3"
        self.kickoff = datetime(2026, 11, 8, 18, tzinfo=timezone.utc)
        self.published = self.kickoff - timedelta(hours=2)
        self.observation = Observation(
            season=2026,
            week=10,
            game_id="2026_10_SEA_LAR",
            kickoff=self.kickoff,
            team="SEA",
            player_id="00-0030565",
            source_type="official_team_announcement",
            source_url="https://example.com/team/announcement",
            source_published_at=self.published,
            source_evidence="The team announced this player will start.",
            confirmed_role="starter",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_timezone_aware_publication_before_kickoff_is_saved_as_utc(self):
        add_observation(self.observation, self.database)
        row = list_observations(self.database)[0]
        self.assertEqual(row["verification_status"], "role_timestamp_validated")
        self.assertEqual(row["source_published_utc"], self.published.isoformat())
        self.assertTrue(row["retrieved_utc"].endswith("+00:00"))

    def test_timezone_naive_timestamps_are_rejected(self):
        naive = datetime(2026, 11, 8, 16)
        with self.assertRaisesRegex(ValueError, "must include a timezone"):
            add_observation(
                Observation(
                    **{
                        **self.observation.__dict__,
                        "source_published_at": naive,
                    }
                ),
                self.database,
            )

    def test_post_kickoff_observation_is_excluded(self):
        postgame = self.kickoff + timedelta(seconds=1)
        with self.assertRaisesRegex(ValueError, "must be before kickoff"):
            add_observation(
                Observation(
                    **{
                        **self.observation.__dict__,
                        "source_published_at": postgame,
                    }
                ),
                self.database,
            )
        self.assertEqual(list_observations(self.database), [])

    def test_publication_exactly_at_kickoff_is_excluded(self):
        with self.assertRaisesRegex(ValueError, "must be before kickoff"):
            add_observation(
                Observation(
                    **{
                        **self.observation.__dict__,
                        "source_published_at": self.kickoff,
                    }
                ),
                self.database,
            )

    def test_exact_duplicate_is_rejected_without_overwriting_history(self):
        add_observation(self.observation, self.database)
        with self.assertRaises(DuplicateObservationError):
            add_observation(self.observation, self.database)
        self.assertEqual(len(list_observations(self.database)), 1)

    def test_distinct_source_updates_are_appended_as_history(self):
        add_observation(self.observation, self.database)
        later = Observation(
            **{
                **self.observation.__dict__,
                "source_published_at": self.published + timedelta(minutes=5),
                "source_evidence": "Updated team statement confirms the start.",
            }
        )
        add_observation(later, self.database)
        rows = list_observations(self.database)
        self.assertEqual(len(rows), 2)
        self.assertLess(
            rows[0]["source_published_utc"],
            rows[1]["source_published_utc"],
        )

    def test_projected_qb1_is_never_saved_as_confirmed_starter(self):
        projected = Observation(
            **{
                **self.observation.__dict__,
                "source_type": "projected_depth_chart",
                "projected_role": "QB1",
                "confirmed_role": None,
            }
        )
        add_observation(projected, self.database)
        row = list_observations(self.database)[0]
        self.assertEqual(row["projected_role"], "QB1")
        self.assertIsNone(row["confirmed_role"])
        self.assertEqual(row["verification_status"], "projected_only")

    def test_confirmed_role_requires_official_announcement_source(self):
        invalid = Observation(
            **{
                **self.observation.__dict__,
                "source_type": "projected_depth_chart",
                "projected_role": "QB1",
            }
        )
        with self.assertRaisesRegex(ValueError, "requires source_type"):
            add_observation(invalid, self.database)

    def test_confirmed_role_requires_explicit_evidence(self):
        invalid = Observation(
            **{**self.observation.__dict__, "source_evidence": " "}
        )
        with self.assertRaisesRegex(ValueError, "source_evidence are required"):
            add_observation(invalid, self.database)

    def test_filtering_by_game_and_team(self):
        add_observation(self.observation, self.database)
        other_game = Observation(
            **{
                **self.observation.__dict__,
                "game_id": "2026_11_SEA_ARI",
                "week": 11,
            }
        )
        add_observation(other_game, self.database)
        rows = list_observations(
            self.database,
            season=2026,
            week=10,
            team="sea",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["game_id"], "2026_10_SEA_LAR")

    def test_table_contains_required_audit_fields(self):
        add_observation(self.observation, self.database)
        with closing(sqlite3.connect(self.database)) as connection:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(qb_observations)")
            }
        self.assertTrue(
            {
                "source_url",
                "source_type",
                "source_published_utc",
                "retrieved_utc",
                "season",
                "week",
                "game_id",
                "team",
                "player_id",
                "projected_role",
                "confirmed_role",
                "injury_status",
                "verification_status",
            }.issubset(columns)
        )


if __name__ == "__main__":
    unittest.main()
