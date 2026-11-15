
import sqlite3
from contextlib import closing
from pathlib import Path

DB_PATH = Path(__file__).with_name("nfl_prediction_history.sqlite3")


def generate_accuracy_report():
    """Evaluate one pregame prediction per completed game."""

    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute("""
            SELECT
                game_id,
                home_team,
                away_team,
                projected_home_score,
                projected_away_score,
                actual_home_score,
                actual_away_score,
                model_home_spread,
                market_home_spread
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY game_id
                           ORDER BY predicted_at_utc DESC, id DESC
                       ) AS prediction_rank
                FROM predictions
                WHERE predicted_at_utc < kickoff_utc
            )
            WHERE prediction_rank = 1
              AND actual_home_score IS NOT NULL
              AND actual_away_score IS NOT NULL
        """).fetchall()

    total = len(rows)
    winner_correct = 0
    winner_decisions = 0
    total_margin_error = 0
    margin_games = 0
    ats_wins = 0
    ats_losses = 0
    ats_pushes = 0

    for row in rows:
        (
            game_id, home, away,
            projected_home, projected_away,
            actual_home, actual_away,
            model_spread, market_spread
        ) = row

        if None in (projected_home, projected_away):
            continue

        predicted_margin = projected_home - projected_away
        actual_margin = actual_home - actual_away

        total_margin_error += abs(predicted_margin - actual_margin)
        margin_games += 1

        if predicted_margin != 0 and actual_margin != 0:
            winner_decisions += 1
            if (predicted_margin > 0) == (actual_margin > 0):
                winner_correct += 1

        if market_spread is not None and model_spread is not None:
            # Both spreads must use the home-team convention.
            edge = market_spread - model_spread
            ats_result = actual_margin + market_spread

            if abs(edge) < 1e-9:
                continue

            if abs(ats_result) < 1e-9:
                ats_pushes += 1
            elif (edge > 0 and ats_result > 0) or (
                edge < 0 and ats_result < 0
            ):
                ats_wins += 1
            else:
                ats_losses += 1

    print("\nNFL EDGE PRO — ACCURACY REPORT")
    print("=" * 40)
    print(f"Completed games: {total}")
    print(f"Winner accuracy: {winner_correct}/{winner_decisions}")

    if winner_decisions:
        print(f"Winner percentage: {winner_correct / winner_decisions:.1%}")
    else:
        print("Winner percentage: N/A")

    if margin_games:
        print(f"Margin MAE: {total_margin_error / margin_games:.2f} points")
    else:
        print("Margin MAE: N/A")

    print(f"ATS record: {ats_wins}-{ats_losses}-{ats_pushes}")
    print(f"ATS decisions: {ats_wins + ats_losses}")


if __name__ == "__main__":
    generate_accuracy_report()
