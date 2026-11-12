"""Expanding-window calibration experiment for NFL team power ratings.

The original model and its historical CSVs are not modified. Each season's
rating features are constructed using only completed games strictly before
the target game's calendar date, following nfl_backtest.py. Candidate weights
are selected using earlier seasons only; closing spreads are evaluation-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import nflreadpy as nfl
import pandas as pd
from scipy.stats import norm

from nfl_backtest import MIN_PRIOR_TEAM_GAMES, SEASONS
from nfl_model import SPREAD_STD, calculate_rankings, prepare_plays

BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_PATH = BASE_DIR / "nfl_power_calibration_predictions.csv"
SUMMARY_PATH = BASE_DIR / "nfl_power_calibration_summary.csv"
SELECTIONS_PATH = BASE_DIR / "nfl_power_calibration_selections.csv"

EPA_WEIGHTS = (10.0, 15.0, 20.0, 25.0, 30.0)
SOS_WEIGHTS = (0.0, 5.0, 10.0, 15.0)
RECENT_WEIGHTS = (0.0, 2.5, 5.0, 7.5)
HOME_ADVANTAGES = (0.0, 1.5, 3.0)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 8675309

FEATURE_COLUMNS = (
    "home_net_epa",
    "away_net_epa",
    "home_sos",
    "away_sos",
    "home_recent_delta",
    "away_recent_delta",
)


@dataclass(frozen=True)
class Weights:
    epa: float
    sos: float
    recent: float
    home_advantage: float

    @property
    def key(self) -> tuple[float, float, float, float]:
        return (self.epa, self.sos, self.recent, self.home_advantage)


BASELINE = Weights(20.0, 10.0, 5.0, 1.5)
CANDIDATES = tuple(
    Weights(*values)
    for values in product(
        EPA_WEIGHTS,
        SOS_WEIGHTS,
        RECENT_WEIGHTS,
        HOME_ADVANTAGES,
    )
)


def _finished_regular_games(schedules: pd.DataFrame) -> pd.DataFrame:
    required = {
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
    }
    missing = required.difference(schedules.columns)
    if missing:
        raise ValueError(f"Schedule missing columns: {sorted(missing)}")
    games = schedules.loc[schedules["game_type"] == "REG"].copy()
    games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
    games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")
    games["spread_line"] = pd.to_numeric(games["spread_line"], errors="coerce")
    games = games.loc[
        np.isfinite(games["home_score"]) & np.isfinite(games["away_score"])
    ].copy()
    games["_game_date"] = pd.to_datetime(
        games["gameday"], errors="coerce"
    ).dt.date
    return games.loc[games["_game_date"].notna()].copy()


def _team_game_counts(games: pd.DataFrame) -> pd.Series:
    return games["home_team"].value_counts().add(
        games["away_team"].value_counts(), fill_value=0
    )


def _extract_game_features(
    season: int,
    schedules: pd.DataFrame,
    pbp: pd.DataFrame,
    min_prior_team_games: int = MIN_PRIOR_TEAM_GAMES,
) -> pd.DataFrame:
    """Create one row per eligible game from prior-date team-rating features."""
    completed = _finished_regular_games(schedules)
    if completed.empty:
        return pd.DataFrame()
    valid_plays = prepare_plays(pbp, schedules)
    feature_rows: list[dict] = []

    for prediction_date, target_games in completed.groupby(
        "_game_date", sort=True
    ):
        prior_games = completed.loc[completed["_game_date"] < prediction_date]
        if prior_games.empty:
            continue
        game_counts = _team_game_counts(prior_games)
        eligible = target_games.loc[
            target_games.apply(
                lambda game: (
                    game_counts.get(game["home_team"], 0) >= min_prior_team_games
                    and game_counts.get(game["away_team"], 0) >= min_prior_team_games
                ),
                axis=1,
            )
        ]
        if eligible.empty:
            continue

        prior_ids = set(prior_games["game_id"])
        prior_plays = valid_plays.loc[valid_plays["game_id"].isin(prior_ids)]
        if prior_plays.empty:
            continue
        ratings = calculate_rankings(prior_plays)

        for _, game in eligible.iterrows():
            home = game["home_team"]
            away = game["away_team"]
            if home not in ratings.index or away not in ratings.index:
                continue
            home_row = ratings.loc[home]
            away_row = ratings.loc[away]
            feature_rows.append(
                {
                    "season": int(season),
                    "game_id": game["game_id"],
                    "week": int(game["week"]),
                    "gameday": game["_game_date"].isoformat(),
                    "home_team": home,
                    "away_team": away,
                    "home_net_epa": float(home_row["net_epa"]),
                    "away_net_epa": float(away_row["net_epa"]),
                    "home_sos": float(home_row["sos"]),
                    "away_sos": float(away_row["sos"]),
                    "home_recent_delta": float(
                        home_row["recent_off_epa"] - home_row["off_epa"]
                    ),
                    "away_recent_delta": float(
                        away_row["recent_off_epa"] - away_row["off_epa"]
                    ),
                    "actual_margin": float(game["home_score"] - game["away_score"]),
                    "home_score": float(game["home_score"]),
                    "away_score": float(game["away_score"]),
                    "spread_line_source": game["spread_line"],
                }
            )

    return pd.DataFrame(feature_rows)


def _predict_margins(features: pd.DataFrame, weights: Weights) -> np.ndarray:
    return (
        (features["home_net_epa"].to_numpy() - features["away_net_epa"].to_numpy())
        * weights.epa
        + (features["home_sos"].to_numpy() - features["away_sos"].to_numpy())
        * weights.sos
        + (
            features["home_recent_delta"].to_numpy()
            - features["away_recent_delta"].to_numpy()
        )
        * weights.recent
        + weights.home_advantage
    )


def _candidate_errors(
    season_features: dict[int, pd.DataFrame],
) -> tuple[dict[tuple[float, float, float, float], float], int]:
    errors: dict[tuple[float, float, float, float], list[np.ndarray]] = {
        weights.key: [] for weights in CANDIDATES
    }
    total_games = 0
    for features in season_features.values():
        if features.empty:
            continue
        total_games += len(features)
        actual = features["actual_margin"].to_numpy()
        for weights in CANDIDATES:
            errors[weights.key].append(
                np.abs(_predict_margins(features, weights) - actual)
            )
    mae = {
        key: float(np.concatenate(values).mean())
        for key, values in errors.items()
        if values
    }
    return mae, total_games


def _select_weights(
    season_features: dict[int, pd.DataFrame],
    seasons_before_test: tuple[int, ...],
) -> tuple[Weights, float, int]:
    training = {
        season: season_features[season]
        for season in seasons_before_test
        if season in season_features and not season_features[season].empty
    }
    if not training:
        raise ValueError("No earlier-season walk-forward predictions to tune on")
    candidate_mae, sample_size = _candidate_errors(training)
    # Deterministic tie-break: nearest to the established baseline, then
    # smallest total departure from it.
    best_key = min(
        candidate_mae,
        key=lambda key: (
            candidate_mae[key],
            sum((left - right) ** 2 for left, right in zip(key, BASELINE.key)),
            key,
        ),
    )
    selected = Weights(*best_key)
    return selected, candidate_mae[best_key], sample_size


def _grade_ats(
    predicted_margin: np.ndarray,
    actual_margin: np.ndarray,
    schedule_spread_line: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Grade direction versus closing spread; spread_line is + for home favorite."""
    closing_home_line = -schedule_spread_line
    model_home_line = -predicted_margin
    model_edge_home = model_home_line - closing_home_line
    pick = np.where(model_edge_home < 0, "HOME", np.where(model_edge_home > 0, "AWAY", "PASS"))
    cover_margin = actual_margin + closing_home_line
    result = np.where(
        pick == "PASS",
        "NO_BET",
        np.where(
            np.isclose(cover_margin, 0.0),
            "PUSH",
            np.where(
                ((pick == "HOME") & (cover_margin > 0))
                | ((pick == "AWAY") & (cover_margin < 0)),
                "WIN",
                "LOSS",
            ),
        ),
    )
    return pick, result


def _prediction_rows(
    features: pd.DataFrame,
    method: str,
    weights: Weights,
    tuned_for_season: int | None,
    tuning_seasons: tuple[int, ...],
    training_mae: float | None,
) -> list[dict]:
    margins = _predict_margins(features, weights)
    actual = features["actual_margin"].to_numpy()
    schedule_spread = features["spread_line_source"].to_numpy(dtype=float)
    predicted_pick, ats_result = _grade_ats(margins, actual, schedule_spread)
    home_probability = norm.cdf(margins / SPREAD_STD)
    rows = []
    for index, (_, game) in enumerate(features.iterrows()):
        actual_winner = (
            game["home_team"]
            if actual[index] > 0
            else game["away_team"]
            if actual[index] < 0
            else None
        )
        rows.append(
            {
                "season": int(game["season"]),
                "game_id": game["game_id"],
                "week": int(game["week"]),
                "gameday": game["gameday"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "method": method,
                "epa_weight": weights.epa,
                "sos_weight": weights.sos,
                "recent_offense_weight": weights.recent,
                "home_advantage": weights.home_advantage,
                "tuned_for_season": tuned_for_season,
                "tuning_seasons": ",".join(map(str, tuning_seasons)),
                "training_mae": training_mae,
                "predicted_margin": float(margins[index]),
                "predicted_home_spread": float(-margins[index]),
                "home_win_probability": float(home_probability[index]),
                "predicted_winner": (
                    game["home_team"] if margins[index] >= 0 else game["away_team"]
                ),
                "actual_margin": float(actual[index]),
                "actual_winner": actual_winner,
                "absolute_margin_error": float(abs(margins[index] - actual[index])),
                "closing_home_spread": float(-schedule_spread[index]),
                "model_vs_market_spread": float(
                    -margins[index] - (-schedule_spread[index])
                ),
                "model_ats_pick": predicted_pick[index],
                "model_ats_result": ats_result[index],
            }
        )
    return rows


def _metric_summary(
    rows: pd.DataFrame,
    season: int | str,
    method: str,
    tuning_seasons: str = "",
    training_mae: float | None = None,
) -> dict:
    actual_decisive = rows["actual_winner"].notna()
    winner_correct = (
        rows.loc[actual_decisive, "predicted_winner"]
        == rows.loc[actual_decisive, "actual_winner"]
    )
    ats_decisive = rows["model_ats_result"].isin(["WIN", "LOSS"])
    ats_wins = int((rows["model_ats_result"] == "WIN").sum())
    ats_losses = int((rows["model_ats_result"] == "LOSS").sum())
    ats_pushes = int((rows["model_ats_result"] == "PUSH").sum())
    probabilities = rows.loc[actual_decisive, "home_win_probability"].astype(float)
    outcomes = (
        rows.loc[actual_decisive, "actual_margin"].astype(float) > 0
    ).astype(float)
    return {
        "season": season,
        "method": method,
        "games": len(rows),
        "winner_decisions": int(actual_decisive.sum()),
        "straight_up_wins": int(winner_correct.sum()),
        "straight_up_accuracy": float(winner_correct.mean())
        if len(winner_correct)
        else np.nan,
        "margin_mae": float(rows["absolute_margin_error"].mean())
        if len(rows)
        else np.nan,
        "brier_score": float(((probabilities - outcomes) ** 2).mean())
        if len(probabilities)
        else np.nan,
        "ats_wins": ats_wins,
        "ats_losses": ats_losses,
        "ats_pushes": ats_pushes,
        "ats_decisions": int(ats_decisive.sum()),
        "ats_accuracy": ats_wins / int(ats_decisive.sum())
        if ats_decisive.any()
        else np.nan,
        "tuning_seasons": tuning_seasons,
        "training_margin_mae": training_mae,
        "candidate_count": len(CANDIDATES),
        "selection_metric": "training walk-forward margin MAE",
    }


def _bootstrap_difference(
    baseline_values: np.ndarray,
    tuned_values: np.ndarray,
    statistic: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if len(baseline_values) != len(tuned_values) or not len(tuned_values):
        return np.nan, np.nan
    differences = []
    sample_count = len(tuned_values)
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, sample_count, size=sample_count)
        base = baseline_values[indices]
        tuned = tuned_values[indices]
        if statistic == "mean":
            differences.append(float(np.mean(tuned - base)))
        else:
            differences.append(float(np.mean(tuned) - np.mean(base)))
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return float(lower), float(upper)


def _comparison_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for season in sorted(predictions["season"].unique()):
        pair = predictions.loc[predictions["season"] == season]
        if set(pair["method"]) != {"baseline", "tuned"}:
            continue
        base = pair.loc[pair["method"] == "baseline"].set_index("game_id")
        tuned = pair.loc[pair["method"] == "tuned"].set_index("game_id")
        common = base.index.intersection(tuned.index)
        base = base.loc[common]
        tuned = tuned.loc[common]
        rows.append(
            _comparison_summary(
                int(season), base, tuned, rng
            )
        )

    test = predictions.loc[predictions["season"] >= min(SEASONS) + 1]
    base = test.loc[test["method"] == "baseline"].set_index("game_id")
    tuned = test.loc[test["method"] == "tuned"].set_index("game_id")
    common = base.index.intersection(tuned.index)
    if len(common):
        rows.append(
            _comparison_summary(
                "HELD_OUT_ALL",
                base.loc[common],
                tuned.loc[common],
                rng,
            )
        )
    return pd.DataFrame(rows)


def _comparison_summary(
    season: int | str,
    baseline: pd.DataFrame,
    tuned: pd.DataFrame,
    rng: np.random.Generator,
) -> dict:
    base_errors = baseline["absolute_margin_error"].to_numpy(dtype=float)
    tuned_errors = tuned["absolute_margin_error"].to_numpy(dtype=float)
    margin_ci = _bootstrap_difference(base_errors, tuned_errors, "mean", rng)

    base_winner = (
        baseline["predicted_winner"].to_numpy()
        == baseline["actual_winner"].to_numpy()
    )
    tuned_winner = (
        tuned["predicted_winner"].to_numpy()
        == tuned["actual_winner"].to_numpy()
    )
    decisive = baseline["actual_winner"].notna().to_numpy()
    base_su = base_winner[decisive].astype(float)
    tuned_su = tuned_winner[decisive].astype(float)
    su_ci = _bootstrap_difference(base_su, tuned_su, "rate", rng)

    base_ats_valid = baseline["model_ats_result"].isin(["WIN", "LOSS"]).to_numpy()
    tuned_ats_valid = tuned["model_ats_result"].isin(["WIN", "LOSS"]).to_numpy()
    # The market push is independent of model selection. Keep the paired sample
    # where both prediction methods have a non-push ATS outcome.
    ats_valid = base_ats_valid & tuned_ats_valid
    base_ats = (
        baseline["model_ats_result"].to_numpy()[ats_valid] == "WIN"
    ).astype(float)
    tuned_ats = (
        tuned["model_ats_result"].to_numpy()[ats_valid] == "WIN"
    ).astype(float)
    ats_ci = _bootstrap_difference(base_ats, tuned_ats, "rate", rng)

    return {
        "season": season,
        "baseline_games": len(baseline),
        "tuned_games": len(tuned),
        "paired_games": len(baseline),
        "baseline_margin_mae": float(base_errors.mean()),
        "tuned_margin_mae": float(tuned_errors.mean()),
        "tuned_minus_baseline_margin_mae": float(
            tuned_errors.mean() - base_errors.mean()
        ),
        "margin_mae_difference_ci95_low": margin_ci[0],
        "margin_mae_difference_ci95_high": margin_ci[1],
        "baseline_straight_up_accuracy": float(base_su.mean()) if len(base_su) else np.nan,
        "tuned_straight_up_accuracy": float(tuned_su.mean()) if len(tuned_su) else np.nan,
        "tuned_minus_baseline_straight_up_pp": (
            100 * (tuned_su.mean() - base_su.mean()) if len(base_su) else np.nan
        ),
        "straight_up_difference_ci95_low_pp": 100 * su_ci[0],
        "straight_up_difference_ci95_high_pp": 100 * su_ci[1],
        "straight_up_decisions": len(base_su),
        "baseline_ats_accuracy": float(base_ats.mean()) if len(base_ats) else np.nan,
        "tuned_ats_accuracy": float(tuned_ats.mean()) if len(tuned_ats) else np.nan,
        "tuned_minus_baseline_ats_pp": (
            100 * (tuned_ats.mean() - base_ats.mean()) if len(base_ats) else np.nan
        ),
        "ats_difference_ci95_low_pp": 100 * ats_ci[0],
        "ats_difference_ci95_high_pp": 100 * ats_ci[1],
        "ats_paired_decisions": len(base_ats),
        "inference_note": (
            "Paired game-level bootstrap 95% intervals; seasons are descriptive. "
            "Only the aggregate held-out comparison is the primary comparison. "
            "Per-season intervals are exploratory; no per-season significance claims."
        ),
    }


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load one season at a time, tune on past seasons, and evaluate forward."""
    features_by_season: dict[int, pd.DataFrame] = {}
    prediction_rows: list[dict] = []
    selection_rows: list[dict] = []

    for season in SEASONS:
        print(f"Loading walk-forward features for {season}...")
        schedules = nfl.load_schedules(season).to_pandas()
        pbp = nfl.load_pbp(season).to_pandas()
        features = _extract_game_features(season, schedules, pbp)
        features_by_season[season] = features
        del pbp, schedules

        if season == min(SEASONS):
            selected = None
            training_mae = None
            training_seasons: tuple[int, ...] = ()
        else:
            training_seasons = tuple(year for year in SEASONS if year < season)
            selected, training_mae, training_games = _select_weights(
                features_by_season, training_seasons
            )
            selection_rows.append(
                {
                    "test_season": season,
                    "selected_epa_weight": selected.epa,
                    "selected_sos_weight": selected.sos,
                    "selected_recent_offense_weight": selected.recent,
                    "selected_home_advantage": selected.home_advantage,
                    "baseline_epa_weight": BASELINE.epa,
                    "baseline_sos_weight": BASELINE.sos,
                    "baseline_recent_offense_weight": BASELINE.recent,
                    "baseline_home_advantage": BASELINE.home_advantage,
                    "training_seasons": ",".join(map(str, training_seasons)),
                    "training_games": training_games,
                    "training_margin_mae": training_mae,
                    "candidate_count": len(CANDIDATES),
                    "selection_metric": "pooled prior-season walk-forward margin MAE",
                    "test_season_used_for_selection": False,
                    "multiple_comparison_control": (
                        "Candidate grid fixed in advance; all candidates scored only "
                        "on seasons before test_season. Test season is never used to select."
                    ),
                }
            )

        if features.empty:
            print(f"No eligible walk-forward games in {season}.")
            continue
        prediction_rows.extend(
            _prediction_rows(
                features,
                "baseline",
                BASELINE,
                season,
                (),
                None,
            )
        )
        if selected is not None:
            prediction_rows.extend(
                _prediction_rows(
                    features,
                    "tuned",
                    selected,
                    season,
                    training_seasons,
                    training_mae,
                )
            )
        print(
            f"{season}: {len(features)} games; "
            + (
                "baseline only (no earlier season available for tuning)."
                if selected is None
                else f"selected {selected.key} using {training_games} prior-season games."
            )
        )

    predictions = pd.DataFrame(prediction_rows)
    summary_rows = []
    for (season, method), group in predictions.groupby(
        ["season", "method"], sort=True
    ):
        metadata = group.iloc[0]
        summary_rows.append(
            _metric_summary(
                group,
                int(season),
                method,
                metadata["tuning_seasons"],
                metadata["training_mae"],
            )
        )
    summaries = pd.DataFrame(summary_rows)
    comparisons = _comparison_rows(predictions)
    if not comparisons.empty:
        comparisons.insert(0, "record_type", "baseline_vs_tuned_comparison")
    if not summaries.empty:
        summaries.insert(0, "record_type", "season_method_metrics")
    combined_summary = pd.concat(
        [summaries, comparisons], ignore_index=True, sort=False
    )

    predictions.to_csv(PREDICTIONS_PATH, index=False)
    combined_summary.to_csv(SUMMARY_PATH, index=False)
    selections = pd.DataFrame(selection_rows)
    selections.to_csv(SELECTIONS_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PATH}")
    print(f"Summary saved to {SUMMARY_PATH}")
    print(f"Weight-selection audit saved to {SELECTIONS_PATH}")
    print(
        combined_summary.loc[
            combined_summary["record_type"] == "season_method_metrics"
        ].to_string(index=False)
    )
    print(
        combined_summary.loc[
            combined_summary["record_type"] == "baseline_vs_tuned_comparison"
        ].to_string(index=False)
    )
    return predictions, combined_summary, selections


if __name__ == "__main__":
    run_experiment()
