
import sqlite3
from pathlib import Path

# Store the database alongside this Python file.
DB_PATH = Path(__file__).resolve().parent / "nfl_prediction_history.sqlite3"


def initialize_database():
    """Create the prediction history table if it does not exist."""
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                kickoff_utc TEXT NOT NULL,
                predicted_at_utc TEXT NOT NULL,
                model_home_spread REAL NOT NULL,
                home_win_probability REAL NOT NULL,
                market_home_spread REAL,
                projected_home_score REAL,
                projected_away_score REAL,
                actual_home_score INTEGER,
                actual_away_score INTEGER,
                graded_at_utc TEXT
            )
        """)

def save_prediction(prediction):
    """Save a timestamped NFL prediction before kickoff."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    kickoff = datetime.fromisoformat(
        prediction["kickoff_utc"].replace("Z", "+00:00")
    )

    if kickoff.tzinfo is None:
        raise ValueError("Kickoff must include a timezone.")

    if now >= kickoff:
        print("Prediction not saved: kickoff has passed.")
        return False

    initialize_database()

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                game_id, season, week, home_team, away_team,
                kickoff_utc, predicted_at_utc,
                model_home_spread, home_win_probability,
                market_home_spread,
                projected_home_score, projected_away_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction["game_id"],
                prediction["season"],
                prediction["week"],
                prediction["home"],
                prediction["away"],
                kickoff.astimezone(timezone.utc).isoformat(),
                now.isoformat(),
                prediction["home_spread"],
                prediction["home_win"],
                prediction.get("market_spread"),
                prediction.get("home_score"),
                prediction.get("away_score"),
            ),
        )

    print(
        f"Saved pregame prediction: "
        f"{prediction['away']} at {prediction['home']}"
    )
    return True


if __name__ == "__main__":
    initialize_database()
    print(f"Prediction database ready: {DB_PATH}")


def grade_prediction(game_id, home_score, away_score):
    """Record final scores for saved predictions."""
    from datetime import datetime, timezone

    graded_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            """
            UPDATE predictions
            SET actual_home_score = ?,
                actual_away_score = ?,
                graded_at_utc = ?
            WHERE game_id = ?
              AND actual_home_score IS NULL
              AND actual_away_score IS NULL
            """,
            (home_score, away_score, graded_at, str(game_id)),
        )

        updated = cursor.rowcount

    if updated > 0:
        print(f"Graded {updated} prediction(s) for game {game_id}")

    return updated


def auto_grade_completed_games(season=2026):
    from datetime import date
    import nflreadpy as nfl
    import polars as pl

    today = date.today()
    schedules = nfl.load_schedules([season])

    with sqlite3.connect(DB_PATH) as connection:
        pending_game_ids = {
            row[0]
            for row in connection.execute(
                """
                SELECT DISTINCT game_id
                FROM predictions
                WHERE season = ?
                  AND actual_home_score IS NULL
                  AND actual_away_score IS NULL
                """,
                (season,),
            )
        }

    """Grade saved predictions using completed NFL scores."""
    from datetime import date
    import nflreadpy as nfl
    import polars as pl

    today = date.today()
    schedules = nfl.load_schedules([season])

    completed = schedules.filter(
        pl.col("game_id").is_in(list(pending_game_ids))
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
        & (pl.col("gameday").str.to_date(strict=False) < today)
    )

    total_graded = 0

    for game in completed.iter_rows(named=True):
        total_graded += grade_prediction(
            game_id=game["game_id"],
            home_score=int(game["home_score"]),
            away_score=int(game["away_score"]),
        )

    print(f"Automatic grading complete: {total_graded} predictions graded.")
    return total_graded
