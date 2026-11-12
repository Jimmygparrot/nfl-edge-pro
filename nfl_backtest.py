"""Chronological, ratings-only NFL backtest for regular-season games."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import nflreadpy as nfl
import pandas as pd
from scipy.stats import norm

from nfl_model import (
    HOME_ADVANTAGE,
    SPREAD_STD,
    calculate_rankings,
    prepare_plays,
)

SEASONS = tuple(range(2020, 2026))
MIN_PRIOR_TEAM_GAMES = 2
RESULTS_PATH = Path(__file__).with_name("nfl_backtest_results.csv")
SUMMARY_PATH = Path(__file__).with_name("nfl_backtest_summary.csv")


def _completed_regular_season_games(schedules: pd.DataFrame) -> pd.DataFrame:
    required = {
        "game_id",
        "season",
        "game_type",
        "gameday",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }
    missing = required.difference(schedules.columns)
    if missing:
        raise ValueError(f"Schedule is missing required columns: {sorted(missing)}")

    games = schedules.loc[schedules["game_type"] == "REG"].copy()
    games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
    games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")
    final_scores = (
        np.isfinite(games["home_score"])
        & np.isfinite(games["away_score"])
    )
    games = games.loc[final_scores].copy()
    games["_game_date"] = pd.to_datetime(
        games["gameday"], errors="coerce"
    ).dt.date
    return games.loc[games["_game_date"].notna()].copy()


def _prior_completed_games(
    completed_games: pd.DataFrame,
    prediction_date: date,
) -> pd.DataFrame:
    """Return only final games from dates strictly before the prediction date."""
    return completed_games.loc[
        completed_games["_game_date"] < prediction_date
    ]


def _team_game_counts(games: pd.DataFrame) -> pd.Series:
    home_counts = games["home_team"].value_counts()
    away_counts = games["away_team"].value_counts()
    return home_counts.add(away_counts, fill_value=0)


def _prediction_row(game: pd.Series, ratings: pd.DataFrame) -> dict:
    home = game["home_team"]
    away = game["away_team"]
    margin = float(ratings.loc[home, "power"] - ratings.loc[away, "power"])
    margin += HOME_ADVANTAGE
    home_win_probability = float(norm.cdf(margin / SPREAD_STD))

    home_score = float(game["home_score"])
    away_score = float(game["away_score"])
    actual_margin = home_score - away_score
    if actual_margin > 0:
        actual_winner = home
    elif actual_margin < 0:
        actual_winner = away
    else:
        actual_winner = None

    return {
        "season": int(game["season"]),
        "game_id": game["game_id"],
        "week": game.get("week"),
        "gameday": game["_game_date"].isoformat(),
        "home_team": home,
        "away_team": away,
        "home_power": float(ratings.loc[home, "power"]),
        "away_power": float(ratings.loc[away, "power"]),
        "predicted_margin": margin,
        "predicted_home_spread": -margin,
        "home_win_probability": home_win_probability,
        "predicted_winner": home if margin >= 0 else away,
        "home_score": home_score,
        "away_score": away_score,
        "actual_margin": actual_margin,
        "actual_winner": actual_winner,
        "margin_error": margin - actual_margin,
    }


def backtest_season(
    season: int,
    schedules: pd.DataFrame,
    pbp: pd.DataFrame,
    min_prior_team_games: int = MIN_PRIOR_TEAM_GAMES,
) -> pd.DataFrame:
    """Predict games in one season, using only prior-date completed games."""
    completed_games = _completed_regular_season_games(schedules)
    if completed_games.empty:
        return pd.DataFrame()

    valid_plays = prepare_plays(pbp, schedules)
    predictions: list[dict] = []

    for prediction_date, target_games in completed_games.groupby(
        "_game_date", sort=True
    ):
        prior_games = _prior_completed_games(completed_games, prediction_date)
        if prior_games.empty:
            continue

        games_played = _team_game_counts(prior_games)
        eligible_games = target_games.loc[
            target_games.apply(
                lambda game: (
                    games_played.get(game["home_team"], 0)
                    >= min_prior_team_games
                    and games_played.get(game["away_team"], 0)
                    >= min_prior_team_games
                ),
                axis=1,
            )
        ]
        if eligible_games.empty:
            continue

        prior_game_ids = set(prior_games["game_id"])
        prior_plays = valid_plays.loc[
            valid_plays["game_id"].isin(prior_game_ids)
        ]
        if prior_plays.empty:
            continue

        ratings = calculate_rankings(prior_plays)
        for _, game in eligible_games.iterrows():
            home = game["home_team"]
            away = game["away_team"]
            if (
                home not in ratings.index
                or away not in ratings.index
            ):
                continue
            predictions.append(_prediction_row(game, ratings))

    return pd.DataFrame(predictions)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Return per-season and pooled prediction metrics."""
    if results.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "games_predicted",
                "decisive_games",
                "ties_excluded",
                "straight_up_correct",
                "straight_up_accuracy",
                "margin_mae",
                "margin_rmse",
                "brier_score",
            ]
        )

    summaries = [
        _summarize_group(season, group)
        for season, group in results.groupby("season", sort=True)
    ]
    summaries.append(_summarize_group("ALL", results))
    return pd.DataFrame(summaries)


def _summarize_group(season: int | str, games: pd.DataFrame) -> dict:
    decisive = games.loc[games["actual_winner"].notna()]
    correct = decisive["predicted_winner"] == decisive["actual_winner"]
    errors = games["margin_error"].astype(float)
    brier = (
        (
            decisive["home_win_probability"].astype(float)
            - (decisive["actual_margin"] > 0).astype(float)
        ) ** 2
    )
    return {
        "season": season,
        "games_predicted": len(games),
        "decisive_games": len(decisive),
        "ties_excluded": len(games) - len(decisive),
        "straight_up_correct": int(correct.sum()),
        "straight_up_accuracy": float(correct.mean()) if len(decisive) else np.nan,
        "margin_mae": float(errors.abs().mean()),
        "margin_rmse": float(np.sqrt((errors**2).mean())),
        "brier_score": float(brier.mean()) if len(decisive) else np.nan,
    }


def run_backtest() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, evaluate, and release one season of source data at a time."""
    season_results = []
    for season in SEASONS:
        print(f"Backtesting {season}...")
        schedules = nfl.load_schedules(season).to_pandas()
        pbp = nfl.load_pbp(season).to_pandas()
        predictions = backtest_season(season, schedules, pbp)
        if predictions.empty:
            print(f"No games met the minimum-history requirement for {season}.")
        else:
            season_results.append(predictions)
            print(f"Evaluated {len(predictions)} games in {season}.")
        del pbp, schedules

    results = (
        pd.concat(season_results, ignore_index=True)
        if season_results
        else pd.DataFrame()
    )
    summary = summarize_results(results)
    results.to_csv(RESULTS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"Predictions saved to {RESULTS_PATH}")
    print(f"Summaries saved to {SUMMARY_PATH}")
    if not summary.empty:
        print(summary.to_string(index=False))
    return results, summary


if __name__ == "__main__":
    run_backtest()
