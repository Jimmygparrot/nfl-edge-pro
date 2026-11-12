import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nfl_qb_depth_chart_collector import (
    ScheduledGame,
    build_projections,
    collect,
    _filter_schedule_week,
    latest_team_qb_projections,
    parse_scheduled_games,
)
from nfl_qb_pregame_tracker import list_observations


class QBDepthChartCollectorTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 10, 16, tzinfo=timezone.utc)
        self.kickoff = datetime(2026, 11, 12, 1, 15, tzinfo=timezone.utc)
        self.game = ScheduledGame(
            season=2026,
            week=10,
            game_id="2026_10_WAS_NYG",
            kickoff=self.kickoff,
            home_team="NYG",
            away_team="WAS",
        )
        self.chart_rows = [
            {
                "dt": "2026-11-10T15:00:00Z",
                "team": "NYG",
                "gsis_id": "QB-OLD-1",
                "player_name": "Old Starter",
                "pos_abb": "QB",
                "pos_rank": 1,
            },
            {
                "dt": "2026-11-10T15:00:00Z",
                "team": "NYG",
                "gsis_id": "QB-OLD-2",
                "player_name": "Old Backup",
                "pos_abb": "QB",
                "pos_rank": 2,
            },
            {
                "dt": "2026-11-10T15:32:14Z",
                "team": "NYG",
                "gsis_id": "QB-NEW-1",
                "player_name": "Projected One",
                "pos_abb": "QB",
                "pos_rank": 1,
            },
            {
                "dt": "2026-11-10T15:32:14Z",
                "team": "NYG",
                "gsis_id": "QB-NEW-2",
                "player_name": "Projected Two",
                "pos_abb": "QB",
                "pos_rank": 2,
            },
            {
                "dt": "2026-11-12T02:00:00Z",
                "team": "NYG",
                "gsis_id": "QB-POST",
                "player_name": "Post Kickoff",
                "pos_abb": "QB",
                "pos_rank": 1,
            },
        ]

    def test_selects_latest_depth_snapshot_strictly_before_kickoff(self):
        projections, issues = latest_team_qb_projections(
            self.chart_rows,
            "NYG",
            self.game,
        )
        self.assertEqual(
            [(p.player_id, p.projected_role) for p in projections],
            [("QB-NEW-1", "QB1"), ("QB-NEW-2", "QB2")],
        )
        self.assertEqual(issues, [])

    def test_missing_timezone_on_chart_timestamp_is_not_used(self):
        rows = [
            {
                **self.chart_rows[0],
                "dt": "2026-11-10T15:00:00",
            }
        ]
        projections, issues = latest_team_qb_projections(rows, "NYG", self.game)
        self.assertEqual(projections, [])
        self.assertTrue(any("timezone offset" in issue for issue in issues))

    def test_ambiguous_rank_is_skipped(self):
        rows = [
            row for row in self.chart_rows
            if row["dt"] == "2026-11-10T15:32:14Z"
        ]
        rows.append(
            {
                **rows[0],
                "gsis_id": "QB-OTHER",
                "player_name": "Conflicting Rank One",
            }
        )
        projections, issues = latest_team_qb_projections(rows, "NYG", self.game)
        self.assertEqual([p.projected_role for p in projections], ["QB2"])
        self.assertTrue(any("rank 1 is ambiguous" in issue for issue in issues))

    def test_schedule_missing_or_ambiguous_kickoff_is_skipped(self):
        rows = [
            {
                "season": 2026,
                "week": 10,
                "game_type": "REG",
                "game_id": "no-time",
                "gameday": "2026-11-12",
                "gametime": None,
                "home_team": "NYG",
                "away_team": "WAS",
            },
            {
                "season": 2026,
                "week": 10,
                "game_type": "REG",
                "game_id": "already-started",
                "gameday": "2026-11-10",
                "gametime": "10:00",
                "home_team": "NYG",
                "away_team": "WAS",
            },
        ]
        games, skipped = parse_scheduled_games(rows, 2026, self.now)
        self.assertEqual(games, [])
        self.assertEqual(len(skipped), 2)

    def test_dry_run_does_not_write_to_tracker(self):
        projections, _ = latest_team_qb_projections(
            self.chart_rows,
            "NYG",
            self.game,
        )
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "tracker.sqlite3"
            results = collect(projections, database_path=database, write=False)
            self.assertTrue(all(result.action == "dry-run" for result in results))
            self.assertFalse(database.exists())

    def test_missing_publication_timestamp_skips_database_writes(self):
        projections, _ = latest_team_qb_projections(
            self.chart_rows,
            "NYG",
            self.game,
        )
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "tracker.sqlite3"
            results = collect(
                projections,
                database_path=database,
                write=True,
                publication_evidence="verified from an authoritative source",
                publication_url="https://example.com/release",
                now=self.now,
            )
            self.assertTrue(all(result.action == "skipped" for result in results))
            self.assertTrue(all("record-load timestamp" in r.note for r in results))
            self.assertFalse(database.exists())

    def test_publication_timestamp_must_be_before_kickoff_and_not_after_load(self):
        projections, _ = latest_team_qb_projections(
            self.chart_rows,
            "NYG",
            self.game,
        )
        chart_loaded_at = projections[0].chart_loaded_at
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "tracker.sqlite3"
            after_kickoff = collect(
                projections[:1],
                database_path=database,
                write=True,
                source_published_at=self.kickoff,
                publication_url="https://example.com/release",
                publication_evidence="claimed timestamp",
                now=self.now,
            )
            self.assertEqual(after_kickoff[0].action, "skipped")
            self.assertIn("before kickoff", after_kickoff[0].note)

            after_chart_load = collect(
                projections[:1],
                database_path=database,
                write=True,
                source_published_at=chart_loaded_at + timedelta(seconds=1),
                publication_url="https://example.com/release",
                publication_evidence="claimed timestamp",
                now=self.now,
            )
            self.assertEqual(after_chart_load[0].action, "skipped")
            self.assertIn("later than", after_chart_load[0].note)
            self.assertEqual(list_observations(database), [])

    def test_valid_external_publication_evidence_stores_projection_only(self):
        projections, _ = latest_team_qb_projections(
            self.chart_rows,
            "NYG",
            self.game,
        )
        published = datetime(2026, 11, 10, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "tracker.sqlite3"
            results = collect(
                projections[:1],
                database_path=database,
                write=True,
                source_published_at=published,
                publication_url="https://example.com/release",
                publication_evidence="publisher release metadata, independently checked",
                now=self.now,
            )
            self.assertEqual(results[0].action, "stored")
            row = list_observations(database)[0]
            self.assertEqual(row["projected_role"], "QB1")
            self.assertIsNone(row["confirmed_role"])
            self.assertEqual(row["verification_status"], "projected_only")

    def test_missing_publication_url_skips_write(self):
        projections, _ = latest_team_qb_projections(
            self.chart_rows,
            "NYG",
            self.game,
        )
        published = datetime(2026, 11, 10, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "tracker.sqlite3"
            results = collect(
                projections[:1],
                database_path=database,
                write=True,
                source_published_at=published,
                publication_evidence="verified publisher metadata",
                now=self.now,
            )
            self.assertEqual(results[0].action, "skipped")
            self.assertIn("timestamp, URL, and evidence", results[0].note)
            self.assertFalse(database.exists())

    def test_duplicate_projection_is_not_written_twice(self):
        projections, _ = latest_team_qb_projections(
            self.chart_rows,
            "NYG",
            self.game,
        )
        published = datetime(2026, 11, 10, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "tracker.sqlite3"
            first = collect(
                projections[:1],
                database_path=database,
                write=True,
                source_published_at=published,
                publication_url="https://example.com/release",
                publication_evidence="same verified release evidence",
                now=self.now,
            )
            second = collect(
                projections[:1],
                database_path=database,
                write=True,
                source_published_at=published,
                publication_url="https://example.com/release",
                publication_evidence="same verified release evidence",
                now=self.now,
            )
            self.assertEqual(first[0].action, "stored")
            self.assertEqual(second[0].action, "duplicate")
            self.assertEqual(len(list_observations(database)), 1)

    def test_build_projections_ignores_non_regular_games(self):
        schedules = [
            {
                "season": 2026,
                "week": 10,
                "game_type": "REG",
                "game_id": self.game.game_id,
                "gameday": "2026-11-11",
                "gametime": "20:15",
                "home_team": "NYG",
                "away_team": "WAS",
            },
            {
                "season": 2026,
                "week": 10,
                "game_type": "POST",
                "game_id": "postseason",
                "gameday": "2026-11-11",
                "gametime": "20:15",
                "home_team": "NYG",
                "away_team": "WAS",
            },
        ]
        projections, _ = build_projections(
            schedules,
            self.chart_rows,
            2026,
            self.now,
        )
        self.assertTrue(projections)
        self.assertTrue(all(p.game_id == self.game.game_id for p in projections))

    def test_week_11_filter_uses_source_week_and_keeps_regular_season_games(self):
        schedules = [
            {
                "season": 2026,
                "week": 10,
                "game_type": "REG",
                "game_id": "2026_10_WAS_NYG",
                "gameday": "2026-11-12",
                "gametime": "20:15",
                "home_team": "NYG",
                "away_team": "WAS",
            },
            {
                "season": 2026,
                "week": 11,
                "game_type": "REG",
                "game_id": "2026_11_IND_HOU",
                "gameday": "2026-11-19",
                "gametime": "20:15",
                "home_team": "HOU",
                "away_team": "IND",
            },
            {
                "season": 2026,
                "week": 11,
                "game_type": "REG",
                "game_id": "2026_11_CIN_WAS",
                "gameday": "2026-11-23",
                "gametime": "20:15",
                "home_team": "WAS",
                "away_team": "CIN",
            },
            {
                "season": 2026,
                "week": 11,
                "game_type": "POST",
                "game_id": "2026_11_POST",
                "gameday": "2026-11-22",
                "gametime": "13:00",
                "home_team": "NYG",
                "away_team": "WAS",
            },
            {
                "season": 2026,
                "week": 12,
                "game_type": "REG",
                "game_id": "2026_12_WAS_NYG",
                "gameday": "2026-11-26",
                "gametime": "20:15",
                "home_team": "NYG",
                "away_team": "WAS",
            },
        ]
        selected = _filter_schedule_week(schedules, 11)
        games, skipped = parse_scheduled_games(
            selected,
            2026,
            datetime(2026, 11, 10, tzinfo=timezone.utc),
        )

        self.assertEqual(
            [game.game_id for game in games],
            ["2026_11_IND_HOU", "2026_11_CIN_WAS"],
        )
        self.assertEqual([game.week for game in games], [11, 11])
        self.assertEqual(
            [game.kickoff.isoformat() for game in games],
            [
                "2026-11-20T01:15:00+00:00",
                "2026-11-24T01:15:00+00:00",
            ],
        )
        self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()
