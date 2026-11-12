"""Expanding-window test of decaying prior-season NFL power ratings.

This experiment is independent of the live model and existing backtest files.
It follows nfl_backtest.py's strict prior-calendar-date cutoff and minimum
two-current-games baseline eligibility. A prior-season rating is available
before the next season starts and is carried across known team-code changes.
The prior is exponentially discounted as each team accumulates current-season
games. Closing spreads are used only to score ATS performance.
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
from nfl_model import HOME_ADVANTAGE, SPREAD_STD, calculate_rankings, prepare_plays

BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_PATH = BASE_DIR / "nfl_preseason_prior_predictions.csv"
SUMMARY_PATH = BASE_DIR / "nfl_preseason_prior_summary.csv"
SELECTIONS_PATH = BASE_DIR / "nfl_preseason_prior_selections.csv"

PRIOR_INITIAL_WEIGHTS = (0.25, 0.50, 0.75)
PRIOR_HALF_LIVES = (2.0, 4.0, 8.0, 16.0)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 9042026

# These are franchise identity aliases, not an adjustment for roster turnover.
FRANCHISE_ALIASES = {
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "JAC": "JAX",
    "KAN": "KC",
    "LA": "LAR",
    "LAR": "LAR",
    "OAK": "LV",
    "LV": "LV",
    "LVR": "LV",
    "SD": "LAC",
    "LAC": "LAC",
    "STL": "LAR",
    "WSH": "WAS",
    "WAS": "WAS",
}


@dataclass(frozen=True)
class BlendParameters:
    initial_prior_weight: float
    half_life_games: float

    @property
    def key(self) -> tuple[float, float]:
        return self.initial_prior_weight, self.half_life_games


PARAMETER_GRID = tuple(
    BlendParameters(*values)
    for values in product(PRIOR_INITIAL_WEIGHTS, PRIOR_HALF_LIVES)
)
SEED_PARAMETERS = BlendParameters(0.50, 8.0)


def _franchise(team: object) -> str | None:
    if not isinstance(team, str) or not team.strip():
        return None
    code = team.strip().upper()
    return FRANCHISE_ALIASES.get(code, code)


def _completed_regular_games(schedules: pd.DataFrame) -> pd.DataFrame:
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
    games["_home_franchise"] = games["home_team"].map(_franchise)
    games["_away_franchise"] = games["away_team"].map(_franchise)
    return games.loc[
        games["_game_date"].notna()
        & games["_home_franchise"].notna()
        & games["_away_franchise"].notna()
    ].copy()


def _final_season_ratings(
    schedules: pd.DataFrame,
    pbp: pd.DataFrame,
) -> dict[str, float]:
    completed = _completed_regular_games(schedules)
    if completed.empty:
        return {}
    plays = prepare_plays(pbp, schedules)
    valid_ids = set(completed["game_id"])
    plays = plays.loc[plays["game_id"].isin(valid_ids)]
    if plays.empty:
        return {}
    ratings = calculate_rankings(plays)
    mapped: dict[str, list[float]] = {}
    for team, row in ratings.iterrows():
        franchise = _franchise(team)
        if franchise is not None:
            mapped.setdefault(franchise, []).append(float(row["power"]))
    return {
        franchise: float(np.mean(values))
        for franchise, values in mapped.items()
    }


def _team_game_counts(prior_games: pd.DataFrame) -> pd.Series:
    return prior_games["home_team"].value_counts().add(
        prior_games["away_team"].value_counts(), fill_value=0
    )


def extract_season_features(
    season: int,
    schedules: pd.DataFrame,
    pbp: pd.DataFrame,
    prior_season_ratings: dict[str, float],
    min_prior_team_games: int = MIN_PRIOR_TEAM_GAMES,
) -> pd.DataFrame:
    """Build per-game ratings using prior dates only plus prior-year ratings."""
    completed = _completed_regular_games(schedules)
    if completed.empty:
        return pd.DataFrame()
    valid_plays = prepare_plays(pbp, schedules)
    rows: list[dict] = []

    for prediction_date, target_games in completed.groupby(
        "_game_date", sort=True
    ):
        prior_games = completed.loc[completed["_game_date"] < prediction_date]
        game_counts = _team_game_counts(prior_games)
        prior_ids = set(prior_games["game_id"])
        prior_plays = valid_plays.loc[valid_plays["game_id"].isin(prior_ids)]
        current_ratings = (
            calculate_rankings(prior_plays)
            if not prior_plays.empty
            else pd.DataFrame()
        )

        for _, game in target_games.iterrows():
            home_code, away_code = game["home_team"], game["away_team"]
            home_franchise = game["_home_franchise"]
            away_franchise = game["_away_franchise"]
            prior_home = prior_season_ratings.get(home_franchise)
            prior_away = prior_season_ratings.get(away_franchise)
            if prior_home is None or prior_away is None:
                continue

            home_games = int(game_counts.get(home_code, 0))
            away_games = int(game_counts.get(away_code, 0))
            has_current_home = (
                not current_ratings.empty and home_code in current_ratings.index
            )
            has_current_away = (
                not current_ratings.empty and away_code in current_ratings.index
            )
            home_current = (
                float(current_ratings.loc[home_code, "power"])
                if has_current_home
                else np.nan
            )
            away_current = (
                float(current_ratings.loc[away_code, "power"])
                if has_current_away
                else np.nan
            )
            baseline_eligible = (
                home_games >= min_prior_team_games
                and away_games >= min_prior_team_games
                and has_current_home
                and has_current_away
            )
            rows.append(
                {
                    "season": int(season),
                    "game_id": game["game_id"],
                    "week": int(game["week"]),
                    "gameday": game["_game_date"].isoformat(),
                    "home_team": home_code,
                    "away_team": away_code,
                    "home_franchise": home_franchise,
                    "away_franchise": away_franchise,
                    "home_games_before": home_games,
                    "away_games_before": away_games,
                    "home_prior_power": float(prior_home),
                    "away_prior_power": float(prior_away),
                    "home_current_power": home_current,
                    "away_current_power": away_current,
                    "baseline_eligible": bool(baseline_eligible),
                    "actual_margin": float(game["home_score"] - game["away_score"]),
                    "home_score": float(game["home_score"]),
                    "away_score": float(game["away_score"]),
                    "spread_line_source": game["spread_line"],
                }
            )

    return pd.DataFrame(rows)


def _prior_share(games_before: int, parameters: BlendParameters) -> float:
    return parameters.initial_prior_weight * 2 ** (
        -games_before / parameters.half_life_games
    )


def _blended_power(
    prior_power: float,
    current_power: float,
    games_before: int,
    parameters: BlendParameters,
) -> tuple[float, float]:
    prior_share = _prior_share(games_before, parameters)
    if not np.isfinite(current_power):
        return float(prior_power), 1.0
    return (
        prior_share * prior_power + (1.0 - prior_share) * current_power,
        prior_share,
    )


def _ats_grade(
    predicted_margin: float,
    actual_margin: float,
    spread_line: float,
) -> tuple[str | None, str | None]:
    if not np.isfinite(spread_line):
        return None, None
    closing_home_line = -spread_line
    model_home_line = -predicted_margin
    edge_home = model_home_line - closing_home_line
    if edge_home < 0:
        pick = "HOME"
    elif edge_home > 0:
        pick = "AWAY"
    else:
        pick = "PASS"
    cover_margin = actual_margin + closing_home_line
    if pick == "PASS":
        outcome = "NO_BET"
    elif np.isclose(cover_margin, 0.0):
        outcome = "PUSH"
    elif (pick == "HOME" and cover_margin > 0) or (
        pick == "AWAY" and cover_margin < 0
    ):
        outcome = "WIN"
    else:
        outcome = "LOSS"
    return pick, outcome


def _method_rows(
    features: pd.DataFrame,
    method: str,
    parameters: BlendParameters | None,
    parameter_season: int | None,
    tuning_seasons: tuple[int, ...],
    training_games: int | None,
) -> list[dict]:
    rows = []
    for _, game in features.iterrows():
        if method == "baseline":
            if not game["baseline_eligible"]:
                continue
            home_power = float(game["home_current_power"])
            away_power = float(game["away_current_power"])
            home_prior_share = 0.0
            away_prior_share = 0.0
        else:
            assert parameters is not None
            home_power, home_prior_share = _blended_power(
                game["home_prior_power"],
                game["home_current_power"],
                int(game["home_games_before"]),
                parameters,
            )
            away_power, away_prior_share = _blended_power(
                game["away_prior_power"],
                game["away_current_power"],
                int(game["away_games_before"]),
                parameters,
            )

        margin = home_power - away_power + HOME_ADVANTAGE
        actual = float(game["actual_margin"])
        actual_winner = (
            game["home_team"] if actual > 0 else game["away_team"] if actual < 0 else None
        )
        ats_pick, ats_result = _ats_grade(
            margin, actual, float(game["spread_line_source"])
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
                "evaluation_scope": (
                    "paired_baseline_eligible"
                    if game["baseline_eligible"]
                    else "newly_covered"
                ),
                "initial_prior_weight": (
                    parameters.initial_prior_weight if parameters else 0.0
                ),
                "prior_half_life_games": (
                    parameters.half_life_games if parameters else 0.0
                ),
                "home_games_before": int(game["home_games_before"]),
                "away_games_before": int(game["away_games_before"]),
                "home_prior_share": home_prior_share,
                "away_prior_share": away_prior_share,
                "home_prior_power": float(game["home_prior_power"]),
                "away_prior_power": float(game["away_prior_power"]),
                "home_current_power": game["home_current_power"],
                "away_current_power": game["away_current_power"],
                "home_power_used": home_power,
                "away_power_used": away_power,
                "parameter_selected_for_season": parameter_season,
                "parameter_training_seasons": ",".join(map(str, tuning_seasons)),
                "parameter_training_games": training_games,
                "predicted_margin": margin,
                "predicted_home_spread": -margin,
                "home_win_probability": float(norm.cdf(margin / SPREAD_STD)),
                "predicted_winner": (
                    game["home_team"] if margin >= 0 else game["away_team"]
                ),
                "actual_margin": actual,
                "actual_winner": actual_winner,
                "absolute_margin_error": abs(margin - actual),
                "spread_line_source": game["spread_line_source"],
                "closing_home_spread": (
                    -float(game["spread_line_source"])
                    if np.isfinite(game["spread_line_source"])
                    else np.nan
                ),
                "model_ats_pick": ats_pick,
                "model_ats_result": ats_result,
            }
        )
    return rows


def _select_parameters(
    prior_features: dict[int, pd.DataFrame],
    train_seasons: tuple[int, ...],
) -> tuple[BlendParameters, float, int]:
    eligible = [
        prior_features[season]
        for season in train_seasons
        if season in prior_features and not prior_features[season].empty
    ]
    if not eligible:
        raise ValueError("No previous-season games available for parameter selection")
    training = pd.concat(eligible, ignore_index=True)
    scores = []
    for candidate in PARAMETER_GRID:
        errors = []
        for _, game in training.iterrows():
            home_power, _ = _blended_power(
                float(game["home_prior_power"]),
                float(game["home_current_power"]),
                int(game["home_games_before"]),
                candidate,
            )
            away_power, _ = _blended_power(
                float(game["away_prior_power"]),
                float(game["away_current_power"]),
                int(game["away_games_before"]),
                candidate,
            )
            margin = home_power - away_power + HOME_ADVANTAGE
            errors.append(abs(margin - float(game["actual_margin"])))
        scores.append((float(np.mean(errors)), candidate.key, candidate))
    best_score, _, selected = min(scores, key=lambda row: (row[0], row[1]))
    return selected, best_score, len(training)


def _summary_metrics(
    frame: pd.DataFrame,
    season: int | str,
    method: str,
    scope: str,
) -> dict:
    decisive = frame["actual_winner"].notna()
    winner_correct = (
        frame.loc[decisive, "predicted_winner"]
        == frame.loc[decisive, "actual_winner"]
    )
    probabilities = frame.loc[decisive, "home_win_probability"].astype(float)
    outcomes = (frame.loc[decisive, "actual_margin"].astype(float) > 0).astype(float)
    ats = frame["model_ats_result"].isin(["WIN", "LOSS"])
    wins = int((frame["model_ats_result"] == "WIN").sum())
    losses = int((frame["model_ats_result"] == "LOSS").sum())
    pushes = int((frame["model_ats_result"] == "PUSH").sum())
    return {
        "record_type": "season_metrics",
        "season": season,
        "method": method,
        "evaluation_scope": scope,
        "games": len(frame),
        "straight_up_decisions": int(decisive.sum()),
        "straight_up_wins": int(winner_correct.sum()),
        "straight_up_accuracy": float(winner_correct.mean())
        if len(winner_correct)
        else np.nan,
        "margin_mae": float(frame["absolute_margin_error"].mean())
        if len(frame)
        else np.nan,
        "brier_score": float(((probabilities - outcomes) ** 2).mean())
        if len(probabilities)
        else np.nan,
        "ats_spread_games": int(frame["model_ats_result"].notna().sum()),
        "ats_wins": wins,
        "ats_losses": losses,
        "ats_pushes": pushes,
        "ats_decisions": int(ats.sum()),
        "ats_accuracy": wins / int(ats.sum()) if ats.any() else np.nan,
        "initial_prior_weight": (
            float(frame["initial_prior_weight"].iloc[0]) if len(frame) else np.nan
        ),
        "prior_half_life_games": (
            float(frame["prior_half_life_games"].iloc[0]) if len(frame) else np.nan
        ),
    }


def _paired_comparison(
    season: int | str,
    baseline: pd.DataFrame,
    blend: pd.DataFrame,
    scope: str,
) -> dict:
    base = baseline.set_index("game_id")
    prior = blend.set_index("game_id")
    common = base.index.intersection(prior.index)
    base = base.loc[common]
    prior = prior.loc[common]
    return {
        "record_type": "paired_comparison",
        "season": season,
        "evaluation_scope": scope,
        "paired_games": len(common),
        "baseline_games": len(base),
        "blend_games": len(prior),
        "baseline_margin_mae": float(base["absolute_margin_error"].mean())
        if len(base)
        else np.nan,
        "blend_margin_mae": float(prior["absolute_margin_error"].mean())
        if len(prior)
        else np.nan,
        "blend_minus_baseline_margin_mae": float(
            prior["absolute_margin_error"].mean()
            - base["absolute_margin_error"].mean()
        )
        if len(base)
        else np.nan,
        "baseline_straight_up_accuracy": float(
            (
                base["predicted_winner"] == base["actual_winner"]
            )[base["actual_winner"].notna()].mean()
        )
        if base["actual_winner"].notna().any()
        else np.nan,
        "blend_straight_up_accuracy": float(
            (
                prior["predicted_winner"] == prior["actual_winner"]
            )[prior["actual_winner"].notna()].mean()
        )
        if prior["actual_winner"].notna().any()
        else np.nan,
        "baseline_brier_score": _brier(base),
        "blend_brier_score": _brier(prior),
        "baseline_ats_accuracy": _ats_accuracy(base),
        "blend_ats_accuracy": _ats_accuracy(prior),
        "multiple_comparison_note": (
            "Parameters selected using earlier seasons only; primary inference "
            "is pooled held-out paired comparison, not separate season tests."
        ),
    }


def _brier(frame: pd.DataFrame) -> float:
    decisive = frame["actual_winner"].notna()
    if not decisive.any():
        return np.nan
    outcome = (frame.loc[decisive, "actual_margin"] > 0).astype(float)
    return float(
        (
            (frame.loc[decisive, "home_win_probability"].astype(float) - outcome) ** 2
        ).mean()
    )


def _ats_accuracy(frame: pd.DataFrame) -> float:
    decisions = frame["model_ats_result"].isin(["WIN", "LOSS"])
    if not decisions.any():
        return np.nan
    return float((frame.loc[decisions, "model_ats_result"] == "WIN").mean())


def _bootstrap_metric_difference(
    baseline: pd.DataFrame,
    blend: pd.DataFrame,
    metric: str,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    base = baseline.set_index("game_id")
    prior = blend.set_index("game_id")
    common = base.index.intersection(prior.index)
    base = base.loc[common]
    prior = prior.loc[common]
    if metric == "margin_mae":
        base_values = base["absolute_margin_error"].to_numpy(dtype=float)
        blend_values = prior["absolute_margin_error"].to_numpy(dtype=float)
    elif metric == "straight_up_accuracy":
        valid = base["actual_winner"].notna()
        base = base.loc[valid]
        prior = prior.loc[valid]
        base_values = (
            base["predicted_winner"].to_numpy()
            == base["actual_winner"].to_numpy()
        ).astype(float)
        blend_values = (
            prior["predicted_winner"].to_numpy()
            == prior["actual_winner"].to_numpy()
        ).astype(float)
    elif metric == "brier_score":
        valid = base["actual_winner"].notna()
        base = base.loc[valid]
        prior = prior.loc[valid]
        outcomes = (base["actual_margin"].to_numpy(dtype=float) > 0).astype(float)
        base_values = (base["home_win_probability"].to_numpy() - outcomes) ** 2
        blend_values = (prior["home_win_probability"].to_numpy() - outcomes) ** 2
    else:
        valid = (
            base["model_ats_result"].isin(["WIN", "LOSS"])
            & prior["model_ats_result"].isin(["WIN", "LOSS"])
        )
        base = base.loc[valid]
        prior = prior.loc[valid]
        base_values = (base["model_ats_result"].to_numpy() == "WIN").astype(float)
        blend_values = (prior["model_ats_result"].to_numpy() == "WIN").astype(float)

    count = len(base_values)
    if count == 0:
        return np.nan, np.nan, 0
    differences = np.empty(BOOTSTRAP_REPLICATES)
    for iteration in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, count, size=count)
        differences[iteration] = np.mean(
            blend_values[indices] - base_values[indices]
        )
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return float(lower), float(upper), count


def _paired_with_uncertainty(
    season: int | str,
    baseline: pd.DataFrame,
    blend: pd.DataFrame,
    rng: np.random.Generator,
) -> dict:
    paired = _paired_comparison(season, baseline, blend, "same_games")
    base = baseline.set_index("game_id")
    tuned = blend.set_index("game_id")
    common = base.index.intersection(tuned.index)
    base, tuned = base.loc[common], tuned.loc[common]
    result = dict(paired)
    result.update(
        {
            "margin_mae_diff_ci95_low": np.nan,
            "margin_mae_diff_ci95_high": np.nan,
            "straight_up_diff_ci95_low": np.nan,
            "straight_up_diff_ci95_high": np.nan,
            "straight_up_diff_n": 0,
            "brier_diff_ci95_low": np.nan,
            "brier_diff_ci95_high": np.nan,
            "brier_diff_n": 0,
            "ats_diff_ci95_low": np.nan,
            "ats_diff_ci95_high": np.nan,
            "ats_diff_n": 0,
        }
    )
    for metric, prefix in (
        ("margin_mae", "margin_mae"),
        ("straight_up_accuracy", "straight_up"),
        ("brier_score", "brier"),
        ("ats_accuracy", "ats"),
    ):
        low, high, count = _bootstrap_metric_difference(
            base.reset_index(),
            tuned.reset_index(),
            metric,
            rng,
        )
        result[f"{prefix}_diff_ci95_low"] = low
        result[f"{prefix}_diff_ci95_high"] = high
        result[f"{prefix}_diff_n"] = count
    result["blend_minus_baseline_straight_up_accuracy"] = (
        result["blend_straight_up_accuracy"]
        - result["baseline_straight_up_accuracy"]
    )
    result["blend_minus_baseline_brier_score"] = (
        result["blend_brier_score"] - result["baseline_brier_score"]
    )
    result["blend_minus_baseline_ats_accuracy"] = (
        result["blend_ats_accuracy"] - result["baseline_ats_accuracy"]
    )
    result["paired_games"] = len(common)
    return result


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prior_ratings: dict[str, float] = {}
    previous_season = min(SEASONS) - 1
    print(f"Loading prior-season ratings for {previous_season}...")
    previous_schedules = nfl.load_schedules(previous_season).to_pandas()
    previous_pbp = nfl.load_pbp(previous_season).to_pandas()
    prior_ratings = _final_season_ratings(previous_schedules, previous_pbp)
    del previous_schedules, previous_pbp

    features_by_season: dict[int, pd.DataFrame] = {}
    prediction_rows: list[dict] = []
    selection_rows: list[dict] = []

    for season in SEASONS:
        print(f"Processing {season}...")
        schedules = nfl.load_schedules(season).to_pandas()
        pbp = nfl.load_pbp(season).to_pandas()
        features = extract_season_features(season, schedules, pbp, prior_ratings)
        features_by_season[season] = features

        if season == min(SEASONS):
            parameters = SEED_PARAMETERS
            training_seasons: tuple[int, ...] = ()
            training_games = 0
            training_mae = np.nan
            selection_basis = "predeclared_seed_parameters; not selected on test season"
        else:
            training_seasons = tuple(year for year in SEASONS if year < season)
            parameters, training_mae, training_games = _select_parameters(
                features_by_season, training_seasons
            )
            selection_basis = "expanding prior seasons; pooled margin MAE"

        selection_rows.append(
            {
                "test_season": season,
                "initial_prior_weight": parameters.initial_prior_weight,
                "prior_half_life_games": parameters.half_life_games,
                "training_seasons": ",".join(map(str, training_seasons)),
                "training_games": training_games,
                "training_margin_mae": training_mae,
                "parameter_grid_size": len(PARAMETER_GRID),
                "selection_basis": selection_basis,
                "test_season_used_for_selection": False,
                "franchise_mapping_note": (
                    "Known team-code aliases mapped to franchises; player/coach "
                    "turnover and offseason roster changes are not modeled."
                ),
            }
        )

        if not features.empty:
            prediction_rows.extend(
                _method_rows(
                    features,
                    "baseline",
                    None,
                    season,
                    (),
                    None,
                )
            )
            prediction_rows.extend(
                _method_rows(
                    features,
                    "preseason_blend",
                    parameters,
                    season,
                    training_seasons,
                    training_games,
                )
            )
        prior_ratings = _final_season_ratings(schedules, pbp)
        del pbp, schedules

    predictions = pd.DataFrame(prediction_rows)
    summary_rows = []
    for (season, method, scope), group in predictions.groupby(
        ["season", "method", "evaluation_scope"], sort=True
    ):
        summary_rows.append(
            _summary_metrics(group, int(season), method, scope)
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparison_rows = []
    heldout_base, heldout_blend = [], []
    for season in SEASONS:
        season_rows = predictions.loc[predictions["season"] == season]
        baseline = season_rows.loc[
            (season_rows["method"] == "baseline")
            & (season_rows["evaluation_scope"] == "paired_baseline_eligible")
        ]
        blend = season_rows.loc[season_rows["method"] == "preseason_blend"]
        blend_paired = blend.loc[
            blend["evaluation_scope"] == "paired_baseline_eligible"
        ]
        blend_new = blend.loc[blend["evaluation_scope"] == "newly_covered"]
        comparison_rows.append(
            _paired_with_uncertainty(season, baseline, blend_paired, rng)
        )
        if not blend_new.empty:
            comparison_rows.append(
                _summary_metrics(
                    blend_new,
                    int(season),
                    "preseason_blend",
                    "newly_covered_games_only",
                )
            )
        if season > min(SEASONS):
            heldout_base.append(baseline)
            heldout_blend.append(blend_paired)

    if heldout_base:
        comparison_rows.append(
            _paired_with_uncertainty(
                "HELD_OUT_ALL",
                pd.concat(heldout_base, ignore_index=True),
                pd.concat(heldout_blend, ignore_index=True),
                rng,
            )
        )

    summary = pd.DataFrame(summary_rows + comparison_rows)
    selections = pd.DataFrame(selection_rows)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    selections.to_csv(SELECTIONS_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PATH}")
    print(f"Summary saved to {SUMMARY_PATH}")
    print(f"Parameter-selection audit saved to {SELECTIONS_PATH}")
    print(
        summary.loc[
            summary["record_type"] == "season_metrics",
            [
                "season",
                "method",
                "evaluation_scope",
                "games",
                "straight_up_accuracy",
                "margin_mae",
                "brier_score",
                "ats_wins",
                "ats_losses",
                "ats_pushes",
                "ats_accuracy",
            ],
        ].to_string(index=False)
    )
    print(
        summary.loc[
            (summary["record_type"] == "paired_comparison")
            & (summary["season"] == "HELD_OUT_ALL")
        ].to_string(index=False)
    )
    return predictions, summary, selections


if __name__ == "__main__":
    run_experiment()
