"""Grade existing NFL predictions against documented historical closing spreads.

Closing lines come from ``nflreadpy.load_schedules(season)`` and its
``spread_line`` field. The nflfastR data dictionary describes ``spread_line``
as the closing point spread, sourced from Pro-Football-Reference, with positive
values indicating the home team is favored:
https://github.com/nflverse/nflfastR/blob/master/data-raw/variable_list.txt
The nflreadr schedule loader documentation is at:
https://nflreadr.nflverse.com/reference/load_schedules.html

The model's existing predictions are consumed as-is; this script does not
recompute them, modify model files, estimate missing lines, or calculate ROI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import nflreadpy as nfl
import pandas as pd

SEASONS = tuple(range(2020, 2026))
DATE_TOLERANCE_DAYS = 1
PREDICTIONS_PATH = Path(__file__).with_name("nfl_backtest_results.csv")
RESULTS_PATH = Path(__file__).with_name("nfl_ats_results.csv")
SUMMARY_PATH = Path(__file__).with_name("nfl_ats_summary.csv")
SPREAD_SOURCE = "Pro-Football-Reference via nflverse nflreadpy schedules"
PREDICTION_COLUMNS = (
    "season",
    "game_id",
    "gameday",
    "home_team",
    "away_team",
    "predicted_home_spread",
    "actual_margin",
)
RESULT_COLUMNS = (
    "season",
    "game_id",
    "gameday",
    "schedule_gameday",
    "home_team",
    "away_team",
    "predicted_home_spread",
    "closing_home_spread",
    "model_vs_market_spread",
    "model_edge_home_points",
    "actual_margin",
    "actual_cover_margin",
    "market_ats_outcome",
    "model_ats_pick",
    "model_ats_result",
    "spread_source",
)

# nflverse generally uses current abbreviations; accept common historical and
# alternate forms in case a prediction CSV and schedule feed differ.
TEAM_ALIASES = {
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "JAC": "JAX",
    "KAN": "KC",
    "LAR": "LA",
    "LVR": "LV",
    "NWE": "NE",
    "NO": "NO",
    "OAK": "LV",
    "PHO": "ARI",
    "SD": "LAC",
    "SFO": "SF",
    "STL": "LA",
    "WSH": "WAS",
}


def _normalize_team(team: object) -> str | None:
    if not isinstance(team, str) or not team.strip():
        return None
    abbreviation = team.strip().upper()
    return TEAM_ALIASES.get(abbreviation, abbreviation)


def _prepare_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    missing = set(PREDICTION_COLUMNS).difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {sorted(missing)}")
    frame = predictions.loc[:, PREDICTION_COLUMNS].copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="coerce").dt.normalize()
    frame["predicted_home_spread"] = pd.to_numeric(
        frame["predicted_home_spread"], errors="coerce"
    )
    frame["actual_margin"] = pd.to_numeric(frame["actual_margin"], errors="coerce")
    frame["_home_key"] = frame["home_team"].map(_normalize_team)
    frame["_away_key"] = frame["away_team"].map(_normalize_team)
    return frame


def _prepare_schedules(schedules: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season",
        "game_type",
        "gameday",
        "home_team",
        "away_team",
        "spread_line",
    }
    missing = required.difference(schedules.columns)
    if missing:
        raise ValueError(f"Schedule data is missing required columns: {sorted(missing)}")
    frame = schedules.loc[schedules["game_type"] == "REG", list(required)].copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="coerce").dt.normalize()
    frame["spread_line"] = pd.to_numeric(frame["spread_line"], errors="coerce")
    frame["_home_key"] = frame["home_team"].map(_normalize_team)
    frame["_away_key"] = frame["away_team"].map(_normalize_team)
    return frame


def _market_outcome(actual_cover_margin: float) -> str:
    if actual_cover_margin > 0:
        return "HOME_COVER"
    if actual_cover_margin < 0:
        return "AWAY_COVER"
    return "PUSH"


def _grade_model_pick(
    pick: str,
    actual_cover_margin: float,
) -> str:
    if pick == "PASS":
        return "NO_BET"
    if actual_cover_margin == 0:
        return "PUSH"
    if (pick == "HOME" and actual_cover_margin > 0) or (
        pick == "AWAY" and actual_cover_margin < 0
    ):
        return "WIN"
    return "LOSS"


def match_and_grade(
    predictions: pd.DataFrame,
    schedules: pd.DataFrame,
    date_tolerance_days: int = DATE_TOLERANCE_DAYS,
) -> tuple[pd.DataFrame, int]:
    """Match by season/date/oriented teams and grade only sourced spreads.

    Exact-date matches take precedence. A unique game with the same oriented
    teams within ``date_tolerance_days`` is accepted to accommodate schedule
    date discrepancies; ambiguous or reused schedule rows are excluded.
    """
    predicted = _prepare_predictions(predictions)
    market = _prepare_schedules(schedules)
    market = market.loc[market["spread_line"].notna()].reset_index(drop=True)

    market_index: dict[tuple[int, str, str], list[int]] = {}
    for index, row in market.iterrows():
        if pd.isna(row["season"]) or pd.isna(row["gameday"]):
            continue
        if row["_home_key"] is None or row["_away_key"] is None:
            continue
        key = (int(row["season"]), row["_home_key"], row["_away_key"])
        market_index.setdefault(key, []).append(index)

    used_market_rows: set[int] = set()
    graded: list[dict] = []
    for _, prediction in predicted.iterrows():
        if (
            pd.isna(prediction["season"])
            or pd.isna(prediction["gameday"])
            or pd.isna(prediction["predicted_home_spread"])
            or pd.isna(prediction["actual_margin"])
            or prediction["_home_key"] is None
            or prediction["_away_key"] is None
        ):
            continue

        key = (
            int(prediction["season"]),
            prediction["_home_key"],
            prediction["_away_key"],
        )
        candidates = [
            index for index in market_index.get(key, [])
            if index not in used_market_rows
        ]
        if not candidates:
            continue

        date_differences = {
            index: abs(
                (market.at[index, "gameday"] - prediction["gameday"]).days
            )
            for index in candidates
        }
        exact = [
            index for index in candidates if date_differences[index] == 0
        ]
        if len(exact) == 1:
            match_index = exact[0]
        elif len(exact) > 1:
            continue
        else:
            close = [
                index for index in candidates
                if date_differences[index] <= date_tolerance_days
            ]
            if len(close) != 1:
                continue
            match_index = close[0]

        used_market_rows.add(match_index)
        schedule = market.loc[match_index]
        # nflverse spread_line is positive when the home team is favored;
        # this output uses the conventional signed home line (negative=favored).
        closing_home_spread = -float(schedule["spread_line"])
        actual_cover_margin = (
            float(prediction["actual_margin"]) + closing_home_spread
        )
        model_spread = float(prediction["predicted_home_spread"])
        model_vs_market = model_spread - closing_home_spread
        model_edge_home = -model_vs_market
        if model_edge_home > 0:
            pick = "HOME"
        elif model_edge_home < 0:
            pick = "AWAY"
        else:
            pick = "PASS"

        graded.append(
            {
                "season": int(prediction["season"]),
                "game_id": prediction["game_id"],
                "gameday": prediction["gameday"].date().isoformat(),
                "schedule_gameday": schedule["gameday"].date().isoformat(),
                "home_team": prediction["home_team"],
                "away_team": prediction["away_team"],
                "predicted_home_spread": model_spread,
                "closing_home_spread": closing_home_spread,
                "model_vs_market_spread": model_vs_market,
                "model_edge_home_points": model_edge_home,
                "actual_margin": float(prediction["actual_margin"]),
                "actual_cover_margin": actual_cover_margin,
                "market_ats_outcome": _market_outcome(actual_cover_margin),
                "model_ats_pick": pick,
                "model_ats_result": _grade_model_pick(pick, actual_cover_margin),
                "spread_source": SPREAD_SOURCE,
            }
        )

    results = pd.DataFrame(graded, columns=RESULT_COLUMNS)
    unmatched = len(predicted) - len(results)
    return results, unmatched


def _summarize_season(
    season: int | str,
    predictions_count: int,
    results: pd.DataFrame,
    unmatched: int,
) -> dict:
    bets = results.loc[results["model_ats_pick"] != "PASS"]
    wins = int((bets["model_ats_result"] == "WIN").sum())
    losses = int((bets["model_ats_result"] == "LOSS").sum())
    pushes = int((bets["model_ats_result"] == "PUSH").sum())
    decisions = wins + losses
    differences = results["model_vs_market_spread"].astype(float)
    return {
        "season": season,
        "predictions": predictions_count,
        "matched_games": len(results),
        "unmatched_games": unmatched,
        "spread_coverage": (
            len(results) / predictions_count if predictions_count else np.nan
        ),
        "model_ats_bets": len(bets),
        "ats_wins": wins,
        "ats_losses": losses,
        "ats_pushes": pushes,
        "ats_decisions": decisions,
        "ats_accuracy": wins / decisions if decisions else np.nan,
        "market_home_covers": int(
            (results["market_ats_outcome"] == "HOME_COVER").sum()
        ),
        "market_away_covers": int(
            (results["market_ats_outcome"] == "AWAY_COVER").sum()
        ),
        "market_pushes": int((results["market_ats_outcome"] == "PUSH").sum()),
        "mean_model_vs_market_spread": (
            float(differences.mean()) if len(differences) else np.nan
        ),
        "mean_abs_model_vs_market_spread": (
            float(differences.abs().mean()) if len(differences) else np.nan
        ),
        "spread_source": SPREAD_SOURCE,
        "profitability_assessed": False,
    }


def summarize_results(
    predictions: pd.DataFrame,
    results: pd.DataFrame,
    unmatched_by_season: dict[int, int],
) -> pd.DataFrame:
    """Summarize ATS records and coverage by season and across all seasons."""
    rows = []
    for season in SEASONS:
        season_predictions = predictions.loc[predictions["season"] == season]
        season_results = results.loc[results["season"] == season]
        rows.append(
            _summarize_season(
                season,
                len(season_predictions),
                season_results,
                unmatched_by_season.get(season, len(season_predictions) - len(season_results)),
            )
        )

    rows.append(
        _summarize_season(
            "ALL",
            sum(row["predictions"] for row in rows),
            results,
            sum(unmatched_by_season.values()),
        )
    )
    return pd.DataFrame(rows)


def run_ats_backtest() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load predictions once and schedule/spread data one season at a time."""
    predictions = _prepare_predictions(pd.read_csv(PREDICTIONS_PATH))
    predictions = predictions.loc[predictions["season"].isin(SEASONS)].copy()

    all_results = []
    unmatched_by_season: dict[int, int] = {}
    for season in SEASONS:
        season_predictions = predictions.loc[predictions["season"] == season]
        if season_predictions.empty:
            unmatched_by_season[season] = 0
            continue

        print(f"Matching closing spreads for {season}...")
        schedule = nfl.load_schedules(season).select(
            ["season", "game_type", "gameday", "home_team", "away_team", "spread_line"]
        ).to_pandas()
        season_results, unmatched = match_and_grade(
            season_predictions, schedule
        )
        all_results.append(season_results)
        unmatched_by_season[season] = unmatched
        print(
            f"Matched {len(season_results)} of {len(season_predictions)} "
            f"predictions; {unmatched} unmatched."
        )
        del schedule

    results = (
        pd.concat(all_results, ignore_index=True)
        if all_results
        else pd.DataFrame(columns=RESULT_COLUMNS)
    )
    summary = summarize_results(predictions, results, unmatched_by_season)
    results.to_csv(RESULTS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"ATS results saved to {RESULTS_PATH}")
    print(f"ATS summaries saved to {SUMMARY_PATH}")
    print(summary.to_string(index=False))
    print("ATS accuracy is not a profitability estimate; no returns are modeled.")
    return results, summary


if __name__ == "__main__":
    run_ats_backtest()
