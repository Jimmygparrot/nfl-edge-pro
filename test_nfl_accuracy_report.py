import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
from contextlib import closing


import nfl_accuracy_report


class AccuracyReportTests(unittest.TestCase):

    def test_multiple_snapshots_count_as_one_game(self):
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_predictions.sqlite3"

            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    CREATE TABLE predictions (
                        id INTEGER PRIMARY KEY,
                        game_id TEXT,
                        home_team TEXT,
                        away_team TEXT,
                        projected_home_score REAL,
                        projected_away_score REAL,
                        actual_home_score INTEGER,
                        actual_away_score INTEGER,
                        model_home_spread REAL,
                        market_home_spread REAL,
                        predicted_at_utc TEXT,
                        kickoff_utc TEXT
                    )
                """)

                conn.executemany("""
                    INSERT INTO predictions VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (
                        1, "game1", "KC", "BUF",
                        24, 21, 27, 20, -3, -2.5,
                        "2026-11-01T10:00:00",
                        "2026-11-01T18:00:00"
                    ),
                    (
                        2, "game1", "KC", "BUF",
                        28, 20, 27, 20, -8, -2.5,
                        "2026-11-01T12:00:00",
                        "2026-11-01T18:00:00"
                    )
                ])
            conn.close()

            with patch.object(nfl_accuracy_report, "DB_PATH", db_path):
                with patch("builtins.print") as mock_print:
                    nfl_accuracy_report.generate_accuracy_report()

            # Release SQLite connections before Windows deletes the temp folder.
            import gc
            gc.collect()

            output = "\n".join(
                str(call.args[0])
                for call in mock_print.call_args_list
                if call.args
            )

            self.assertIn("Completed games: 1", output)
            self.assertIn("Margin MAE: 1.00 points", output)


if __name__ == "__main__":
    unittest.main()