"""Summarize NFL backtest and ATS performance by season-week segment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

BASE_DIR = Path(__file__).resolve().parent
BACKTEST_PATH = BASE_DIR / "nfl_backtest_results.csv"
ATS_PATH = BASE_DIR / "nfl_ats_results.csv"
OUTPUT_PATH = BASE_DIR / "nfl_week_analysis.csv"

WEEK_GROUPS = (
    ("Weeks 1-4", 1, 4),
    ("Weeks 5-8", 5, 8),
    ("Weeks 9-12", 9, 12),
    ("Weeks 13-16", 13, 16),
    ("Weeks 17-18", 17, 18),
)

BACKTEST_COLUMNS = [
    "season",
    "game_id",
    "week",
    "home_team",
    "away_team",
    "predicted_winner",
    "actual_winner",
    "margin_error",
]
ATS_COLUMNS = [
    "season",
    "game_id",
    "model_ats_result",
    "model_vs_market_spread",
]
OUTPUT_COLUMNS = [
    "record_type",
    "season",
    "week_group",
    "week_start",
    "week_end",
    "game_count",
    "winner_decisions",
    "straight_up_wins",
    "straight_up_accuracy",
    "straight_up_ci95_low",
    "straight_up_ci95_high",
    "average_absolute_margin_error",
    "ats_matched_games",
    "ats_wins",
    "ats_losses",
    "ats_pushes",
    "ats_decisions",
    "ats_win_percentage",
    "ats_ci95_low",
    "ats_ci95_high",
    "average_absolute_spread_disagreement",
    "median_absolute_spread_disagreement",
    "late_minus_earlier_straight_up_pp",
    "straight_up_fisher_pvalue",
    "late_minus_earlier_ats_pp",
    "ats_fisher_pvalue",
    "late_game_count",
    "earlier_game_count",
    "late_winner_decisions",
    "earlier_winner_decisions",
    "late_ats_decisions",
    "earlier_ats_decisions",
    "late_ats_pushes",
    "earlier_ats_pushes",
    "late_average_absolute_margin_error",
    "earlier_average_absolute_margin_error",
    "late_minus_earlier_absolute_margin_error",
    "late_average_absolute_spread_disagreement",
    "earlier_average_absolute_spread_disagreement",
    "comparison_sample_sizes",
    "interpretation",
]


def _wilson_interval(wins: int, trials: int, z: float = 1.959963984540054):
    if trials == 0:
        return np.nan, np.nan
    proportion = wins / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / trials
            + z**2 / (4 * trials**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def load_joined_results(
    backtest_path: Path = BACKTEST_PATH,
    ats_path: Path = ATS_PATH,
) -> pd.DataFrame:
    """Load only required fields and join ATS results to predictions by game key."""
    backtest = pd.read_csv(backtest_path, usecols=BACKTEST_COLUMNS)
    ats = pd.read_csv(ats_path, usecols=ATS_COLUMNS)
    for frame in (backtest, ats):
        frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
        frame["game_id"] = frame["game_id"].astype(str)
    backtest["week"] = pd.to_numeric(backtest["week"], errors="coerce")
    backtest["margin_error"] = pd.to_numeric(
        backtest["margin_error"], errors="coerce"
    )
    ats["model_vs_market_spread"] = pd.to_numeric(
        ats["model_vs_market_spread"], errors="coerce"
    )

    duplicated_predictions = backtest.duplicated(["season", "game_id"]).any()
    duplicated_ats = ats.duplicated(["season", "game_id"]).any()
    if duplicated_predictions or duplicated_ats:
        raise ValueError("Duplicate season/game_id keys in prediction or ATS CSV")

    joined = backtest.merge(
        ats,
        on=["season", "game_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_ats = int((joined["_merge"] != "both").sum())
    if missing_ats:
        raise ValueError(f"{missing_ats} prediction rows have no matching ATS result")
    joined = joined.drop(columns="_merge")
    joined["absolute_margin_error"] = joined["margin_error"].abs()
    joined["absolute_spread_disagreement"] = (
        joined["model_vs_market_spread"].abs()
    )
    joined["straight_up_decision"] = joined["actual_winner"].notna()
    joined["straight_up_correct"] = (
        joined["predicted_winner"] == joined["actual_winner"]
    ) & joined["straight_up_decision"]
    return joined


def _summarize(
    frame: pd.DataFrame,
    record_type: str,
    season: int | str,
    week_group: str,
    week_start: int | None,
    week_end: int | None,
) -> dict:
    winner_games = frame.loc[frame["straight_up_decision"]]
    winner_wins = int(winner_games["straight_up_correct"].sum())
    winner_trials = len(winner_games)
    su_low, su_high = _wilson_interval(winner_wins, winner_trials)

    ats_wins = int((frame["model_ats_result"] == "WIN").sum())
    ats_losses = int((frame["model_ats_result"] == "LOSS").sum())
    ats_pushes = int((frame["model_ats_result"] == "PUSH").sum())
    ats_trials = ats_wins + ats_losses
    ats_low, ats_high = _wilson_interval(ats_wins, ats_trials)

    spreads = frame["absolute_spread_disagreement"].dropna()
    errors = frame["absolute_margin_error"].dropna()
    return {
        "record_type": record_type,
        "season": season,
        "week_group": week_group,
        "week_start": week_start,
        "week_end": week_end,
        "game_count": len(frame),
        "winner_decisions": winner_trials,
        "straight_up_wins": winner_wins,
        "straight_up_accuracy": winner_wins / winner_trials
        if winner_trials
        else np.nan,
        "straight_up_ci95_low": su_low,
        "straight_up_ci95_high": su_high,
        "average_absolute_margin_error": float(errors.mean())
        if len(errors)
        else np.nan,
        "ats_matched_games": len(spreads),
        "ats_wins": ats_wins,
        "ats_losses": ats_losses,
        "ats_pushes": ats_pushes,
        "ats_decisions": ats_trials,
        "ats_win_percentage": ats_wins / ats_trials if ats_trials else np.nan,
        "ats_ci95_low": ats_low,
        "ats_ci95_high": ats_high,
        "average_absolute_spread_disagreement": float(spreads.mean())
        if len(spreads)
        else np.nan,
        "median_absolute_spread_disagreement": float(spreads.median())
        if len(spreads)
        else np.nan,
        "late_minus_earlier_straight_up_pp": np.nan,
        "straight_up_fisher_pvalue": np.nan,
        "late_minus_earlier_ats_pp": np.nan,
        "ats_fisher_pvalue": np.nan,
        "late_game_count": np.nan,
        "earlier_game_count": np.nan,
        "late_winner_decisions": np.nan,
        "earlier_winner_decisions": np.nan,
        "late_ats_decisions": np.nan,
        "earlier_ats_decisions": np.nan,
        "late_ats_pushes": np.nan,
        "earlier_ats_pushes": np.nan,
        "late_average_absolute_margin_error": np.nan,
        "earlier_average_absolute_margin_error": np.nan,
        "late_minus_earlier_absolute_margin_error": np.nan,
        "late_average_absolute_spread_disagreement": np.nan,
        "earlier_average_absolute_spread_disagreement": np.nan,
        "comparison_sample_sizes": None,
        "interpretation": None,
    }


def build_week_analysis(joined: pd.DataFrame) -> pd.DataFrame:
    """Create pooled and by-season summaries plus a sample-aware late comparison."""
    rows = []
    for label, start, end in WEEK_GROUPS:
        segment = joined.loc[joined["week"].between(start, end)]
        rows.append(_summarize(segment, "overall_week_group", "ALL", label, start, end))
        for season, season_games in segment.groupby("season", sort=True):
            rows.append(
                _summarize(
                    season_games,
                    "season_week_group",
                    int(season),
                    label,
                    start,
                    end,
                )
            )

    # Week 17-18 is compared to all available earlier prediction weeks (3-16
    # in this data; Weeks 1-2 are absent from the source backtest).
    late = joined.loc[joined["week"].between(17, 18)]
    earlier = joined.loc[joined["week"].between(1, 16)]
    late_decisions = late.loc[late["straight_up_decision"]]
    earlier_decisions = earlier.loc[earlier["straight_up_decision"]]
    late_su_wins = int(late_decisions["straight_up_correct"].sum())
    earlier_su_wins = int(earlier_decisions["straight_up_correct"].sum())
    late_ats = late.loc[late["model_ats_result"].isin(["WIN", "LOSS"])]
    earlier_ats = earlier.loc[earlier["model_ats_result"].isin(["WIN", "LOSS"])]
    late_ats_wins = int((late_ats["model_ats_result"] == "WIN").sum())
    earlier_ats_wins = int((earlier_ats["model_ats_result"] == "WIN").sum())

    su_p = (
        fisher_exact(
            [
                [late_su_wins, len(late_decisions) - late_su_wins],
                [earlier_su_wins, len(earlier_decisions) - earlier_su_wins],
            ]
        ).pvalue
        if len(late_decisions) and len(earlier_decisions)
        else np.nan
    )
    ats_p = (
        fisher_exact(
            [
                [late_ats_wins, len(late_ats) - late_ats_wins],
                [earlier_ats_wins, len(earlier_ats) - earlier_ats_wins],
            ]
        ).pvalue
        if len(late_ats) and len(earlier_ats)
        else np.nan
    )
    late_su_rate = late_su_wins / len(late_decisions) if len(late_decisions) else np.nan
    earlier_su_rate = (
        earlier_su_wins / len(earlier_decisions) if len(earlier_decisions) else np.nan
    )
    late_ats_rate = late_ats_wins / len(late_ats) if len(late_ats) else np.nan
    earlier_ats_rate = (
        earlier_ats_wins / len(earlier_ats) if len(earlier_ats) else np.nan
    )
    late_margin_error = late["absolute_margin_error"].mean()
    earlier_margin_error = earlier["absolute_margin_error"].mean()
    late_spread_diff = late["absolute_spread_disagreement"].mean()
    earlier_spread_diff = earlier["absolute_spread_disagreement"].mean()

    comparison = _summarize(
        joined,
        "late_vs_earlier_comparison",
        "ALL",
        "Weeks 17-18 vs Weeks 1-16",
        1,
        18,
    )
    comparison.update(
        {
            "late_minus_earlier_straight_up_pp": (
                100 * (late_su_rate - earlier_su_rate)
                if pd.notna(late_su_rate) and pd.notna(earlier_su_rate)
                else np.nan
            ),
            "straight_up_fisher_pvalue": su_p,
            "late_minus_earlier_ats_pp": (
                100 * (late_ats_rate - earlier_ats_rate)
                if pd.notna(late_ats_rate) and pd.notna(earlier_ats_rate)
                else np.nan
            ),
            "ats_fisher_pvalue": ats_p,
            "late_game_count": len(late),
            "earlier_game_count": len(earlier),
            "late_winner_decisions": len(late_decisions),
            "earlier_winner_decisions": len(earlier_decisions),
            "late_ats_decisions": len(late_ats),
            "earlier_ats_decisions": len(earlier_ats),
            "late_ats_pushes": int(
                (late["model_ats_result"] == "PUSH").sum()
            ),
            "earlier_ats_pushes": int(
                (earlier["model_ats_result"] == "PUSH").sum()
            ),
            "late_average_absolute_margin_error": late_margin_error,
            "earlier_average_absolute_margin_error": earlier_margin_error,
            "late_minus_earlier_absolute_margin_error": (
                late_margin_error - earlier_margin_error
            ),
            "late_average_absolute_spread_disagreement": late_spread_diff,
            "earlier_average_absolute_spread_disagreement": earlier_spread_diff,
            "comparison_sample_sizes": (
                f"SU: late {len(late_decisions)}, earlier {len(earlier_decisions)}; "
                f"ATS decisions: late {len(late_ats)}, earlier {len(earlier_ats)}; "
                f"late ATS pushes {int((late['model_ats_result'] == 'PUSH').sum())}"
            ),
            "interpretation": (
                "Two-sided Fisher exact tests compare win/loss counts; "
                "95% Wilson intervals are included for each group's rates. "
                "A positive late-minus-earlier margin-error difference means "
                "late predictions had larger errors. "
                "These are descriptive historical comparisons, not proof of a "
                "week-of-season effect."
            ),
        }
    )
    rows.append(comparison)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def run_analysis() -> pd.DataFrame:
    joined = load_joined_results()
    analysis = build_week_analysis(joined)
    analysis.to_csv(OUTPUT_PATH, index=False)
    print(f"Week analysis saved to {OUTPUT_PATH}")
    print(
        analysis.loc[analysis["record_type"] == "overall_week_group", [
            "week_group",
            "game_count",
            "straight_up_wins",
            "winner_decisions",
            "straight_up_accuracy",
            "average_absolute_margin_error",
            "ats_wins",
            "ats_losses",
            "ats_pushes",
            "ats_win_percentage",
            "average_absolute_spread_disagreement",
            "median_absolute_spread_disagreement",
        ]].to_string(index=False)
    )
    print(
        analysis.loc[analysis["record_type"] == "late_vs_earlier_comparison", [
            "late_minus_earlier_straight_up_pp",
            "straight_up_fisher_pvalue",
            "late_minus_earlier_ats_pp",
            "ats_fisher_pvalue",
            "comparison_sample_sizes",
            "interpretation",
        ]].to_string(index=False)
    )
    return analysis


if __name__ == "__main__":
    run_analysis()
