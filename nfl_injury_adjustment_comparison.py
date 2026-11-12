"""Generate a read-only comparison of legacy and point-in-time injury effects."""

from collections import Counter
from pathlib import Path
from typing import Iterable

import nflreadpy as nfl
import pandas as pd
from nflreadpy.utils_date import get_current_season

from nfl_injuries import (
    QB_ADJUSTMENT_SCALE,
    InjuryReportHistory,
    InjuryReports,
    QBContext,
    SnapHistory,
    _game_is_complete,
    _schedule_kickoff,
    get_injury_adjustment,
    get_point_in_time_injury_adjustment,
    load_injury_report_comparison_data,
    load_qb_context,
    load_snap_history,
)

OUTPUT_PATH = Path(__file__).with_name("nfl_injury_adjustment_comparison.csv")
FIRST_INJURY_SEASON = 2009
COMPARISON_COLUMNS = [
    "game_id",
    "season",
    "week",
    "gameday",
    "kickoff_utc",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "legacy_home_injury_adjustment",
    "home_legacy_coverage",
    "home_legacy_warnings",
    "point_in_time_home_injury_adjustment",
    "home_point_in_time_coverage",
    "home_point_in_time_warnings",
    "legacy_away_injury_adjustment",
    "away_legacy_coverage",
    "away_legacy_warnings",
    "point_in_time_away_injury_adjustment",
    "away_point_in_time_coverage",
    "away_point_in_time_warnings",
    "legacy_injury_margin_effect",
    "point_in_time_injury_margin_effect",
    "home_adjustment_difference",
    "away_adjustment_difference",
    "predicted_margin_difference_point_in_time_minus_legacy",
]


def compare_historical_games(
    schedule_rows: Iterable[dict],
    season: int,
    legacy_reports: InjuryReports,
    report_history: InjuryReportHistory,
    snap_history: SnapHistory | None = None,
    gsis_to_pfr: dict[str, str] | None = None,
    qb_context: QBContext | None = None,
    qb_adjustment_scale: float | None = QB_ADJUSTMENT_SCALE,
) -> pd.DataFrame:
    """Compare injury-only margin effects for completed regular-season games."""
    rows = []
    for game in schedule_rows:
        if game.get("season") != season or game.get("game_type") != "REG":
            continue
        if not _game_is_complete(game):
            continue

        kickoff = _schedule_kickoff(game)
        if kickoff is None:
            continue

        try:
            week = int(game["week"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue

        home_team = game.get("home_team")
        away_team = game.get("away_team")
        if not isinstance(home_team, str) or not isinstance(away_team, str):
            continue

        legacy_home = get_injury_adjustment(
            home_team,
            week,
            legacy_reports,
            season,
            snap_history,
            gsis_to_pfr,
            qb_context,
            kickoff,
            qb_adjustment_scale,
        )
        legacy_away = get_injury_adjustment(
            away_team,
            week,
            legacy_reports,
            season,
            snap_history,
            gsis_to_pfr,
            qb_context,
            kickoff,
            qb_adjustment_scale,
        )
        point_in_time_home = get_point_in_time_injury_adjustment(
            home_team,
            week,
            season,
            kickoff,
            report_history,
            snap_history,
            gsis_to_pfr,
            qb_context,
            qb_adjustment_scale,
        )
        point_in_time_away = get_point_in_time_injury_adjustment(
            away_team,
            week,
            season,
            kickoff,
            report_history,
            snap_history,
            gsis_to_pfr,
            qb_context,
            qb_adjustment_scale,
        )

        home_key = (season, week, home_team)
        away_key = (season, week, away_team)
        home_legacy_coverage, home_legacy_warnings = _legacy_coverage(
            home_key,
            legacy_reports,
            report_history,
        )
        away_legacy_coverage, away_legacy_warnings = _legacy_coverage(
            away_key,
            legacy_reports,
            report_history,
        )
        legacy_margin_effect = legacy_home - legacy_away
        point_in_time_margin_effect = (
            point_in_time_home.adjustment - point_in_time_away.adjustment
        )
        rows.append(
            {
                "game_id": game.get("game_id"),
                "season": season,
                "week": week,
                "gameday": str(game.get("gameday")),
                "kickoff_utc": kickoff.isoformat(),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": game.get("home_score"),
                "away_score": game.get("away_score"),
                "legacy_home_injury_adjustment": legacy_home,
                "home_legacy_coverage": home_legacy_coverage,
                "home_legacy_warnings": home_legacy_warnings,
                "point_in_time_home_injury_adjustment": (
                    point_in_time_home.adjustment
                ),
                "home_point_in_time_coverage": point_in_time_home.coverage,
                "home_point_in_time_warnings": " | ".join(
                    point_in_time_home.warnings
                ),
                "legacy_away_injury_adjustment": legacy_away,
                "away_legacy_coverage": away_legacy_coverage,
                "away_legacy_warnings": away_legacy_warnings,
                "point_in_time_away_injury_adjustment": (
                    point_in_time_away.adjustment
                ),
                "away_point_in_time_coverage": point_in_time_away.coverage,
                "away_point_in_time_warnings": " | ".join(
                    point_in_time_away.warnings
                ),
                "legacy_injury_margin_effect": legacy_margin_effect,
                "point_in_time_injury_margin_effect": point_in_time_margin_effect,
                "home_adjustment_difference": (
                    point_in_time_home.adjustment - legacy_home
                ),
                "away_adjustment_difference": (
                    point_in_time_away.adjustment - legacy_away
                ),
                "predicted_margin_difference_point_in_time_minus_legacy": (
                    point_in_time_margin_effect - legacy_margin_effect
                ),
            }
        )

    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def _legacy_coverage(
    key: tuple[int, int, str],
    legacy_reports: InjuryReports,
    report_history: InjuryReportHistory,
) -> tuple[str, str]:
    if legacy_reports.get(key):
        return "weighted_statuses_present", ""
    if report_history.get(key):
        return (
            "feed_rows_without_legacy_statuses",
            "Feed rows exist, but no weighted status is present in the legacy "
            "map; zero is not confirmation of health.",
        )
    return (
        "unknown",
        "No qualifying legacy injury status is present; zero is not "
        "confirmation of health.",
    )


def summarize_comparison(
    comparison: pd.DataFrame,
    largest_count: int = 5,
) -> dict:
    """Summarize team-side coverage, diagnostics, and largest margin changes."""
    coverage_counts = Counter(
        coverage
        for column in (
            "home_point_in_time_coverage",
            "away_point_in_time_coverage",
        )
        for coverage in comparison[column].tolist()
    )
    legacy_coverage_counts = Counter(
        coverage
        for column in ("home_legacy_coverage", "away_legacy_coverage")
        for coverage in comparison[column].tolist()
    )
    warning_counts = Counter()
    for column in (
        "home_point_in_time_warnings",
        "away_point_in_time_warnings",
        "home_legacy_warnings",
        "away_legacy_warnings",
    ):
        for warning_text in comparison[column].fillna("").astype(str):
            for warning in filter(None, warning_text.split(" | ")):
                if "timestamp is unavailable" in warning:
                    warning_counts["missing_or_unverifiable_timestamps"] += 1
                if "excluded post-kickoff" in warning:
                    warning_counts["post_kickoff_updates_excluded"] += 1
                if (
                    "coverage is unknown" in warning
                    or "unsupported injury position" in warning
                ):
                    warning_counts["incomplete_injury_data"] += 1

    largest_differences = []
    if not comparison.empty:
        ranked = comparison.assign(
            absolute_margin_difference=pd.to_numeric(
                comparison[
                    "predicted_margin_difference_point_in_time_minus_legacy"
                ],
                errors="coerce",
            ).abs()
        ).nlargest(largest_count, "absolute_margin_difference")
        largest_differences = [
            {
                "game_id": row["game_id"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "margin_difference": row[
                    "predicted_margin_difference_point_in_time_minus_legacy"
                ],
                "absolute_margin_difference": row[
                    "absolute_margin_difference"
                ],
            }
            for _, row in ranked.iterrows()
        ]

    return {
        "games_compared": len(comparison),
        "team_side_coverage": {
            status: coverage_counts[status]
            for status in ("verified", "partial", "unknown")
        },
        "legacy_team_side_coverage": {
            status: legacy_coverage_counts[status]
            for status in (
                "weighted_statuses_present",
                "feed_rows_without_legacy_statuses",
                "unknown",
            )
        },
        "diagnostic_warning_counts": {
            category: warning_counts[category]
            for category in (
                "missing_or_unverifiable_timestamps",
                "post_kickoff_updates_excluded",
                "incomplete_injury_data",
            )
        },
        "largest_margin_differences": largest_differences,
    }


def run_injury_adjustment_comparison(
    seasons: Iterable[int] | None = None,
    output_path: Path = OUTPUT_PATH,
) -> tuple[pd.DataFrame, dict]:
    """Write the comparison CSV without changing prediction or dashboard flows."""
    selected_seasons = (
        list(seasons)
        if seasons is not None
        else list(range(FIRST_INJURY_SEASON, get_current_season() + 1))
    )
    season_results = []

    for season in selected_seasons:
        print(f"Comparing injury adjustments for {season}...")
        schedule = nfl.load_schedules(season)
        legacy_reports, report_history = load_injury_report_comparison_data(
            season
        )
        snap_history, gsis_to_pfr = load_snap_history(season)
        qb_context = (
            load_qb_context(season, schedule.to_pandas())
            if QB_ADJUSTMENT_SCALE is not None
            else None
        )
        result = compare_historical_games(
            schedule.to_dicts(),
            season,
            legacy_reports,
            report_history,
            snap_history,
            gsis_to_pfr,
            qb_context,
            QB_ADJUSTMENT_SCALE,
        )
        if not result.empty:
            season_results.append(result)

    comparison = (
        pd.concat(season_results, ignore_index=True)
        if season_results
        else pd.DataFrame(columns=COMPARISON_COLUMNS)
    )
    comparison.to_csv(output_path, index=False)
    summary = summarize_comparison(comparison)
    print(f"\nComparison CSV saved to {output_path}")
    print(f"Games compared: {summary['games_compared']}")
    print("Team-side coverage:")
    for status, count in summary["team_side_coverage"].items():
        print(f"  {status}: {count}")
    print("Legacy team-side coverage:")
    for status, count in summary["legacy_team_side_coverage"].items():
        print(f"  {status}: {count}")
    print("Diagnostic warnings:")
    for category, count in summary["diagnostic_warning_counts"].items():
        print(f"  {category}: {count}")
    print("Largest predicted-margin differences (points, home perspective):")
    for difference in summary["largest_margin_differences"]:
        print(
            f"  {difference['game_id']} "
            f"{difference['away_team']} at {difference['home_team']}: "
            f"{difference['margin_difference']:+.3f}"
        )
    return comparison, summary


if __name__ == "__main__":
    run_injury_adjustment_comparison()
