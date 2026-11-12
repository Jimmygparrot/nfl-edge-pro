"""Audit historical NFL prediction and ATS CSVs without changing either."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
BACKTEST_PATH = BASE_DIR / "nfl_backtest_results.csv"
ATS_PATH = BASE_DIR / "nfl_ats_results.csv"
REPORT_PATH = BASE_DIR / "nfl_model_diagnostics.csv"
BACKTEST_CODE_PATH = BASE_DIR / "nfl_backtest.py"
TOLERANCE = 1e-6

BACKTEST_COLUMNS = [
    "season",
    "game_id",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "predicted_margin",
    "predicted_home_spread",
    "home_score",
    "away_score",
    "actual_margin",
    "margin_error",
]
ATS_COLUMNS = [
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
]

REPORT_COLUMNS = [
    "record_type",
    "rank",
    "check_id",
    "status",
    "finding",
    "season",
    "week",
    "game_id",
    "gameday",
    "schedule_gameday",
    "home_team",
    "away_team",
    "predicted_margin",
    "actual_margin",
    "absolute_margin_error",
    "predicted_home_spread",
    "closing_home_spread",
    "model_vs_market_spread",
    "absolute_spread_disagreement",
    "details",
]


def _read_inputs(
    backtest_path: Path = BACKTEST_PATH,
    ats_path: Path = ATS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    backtest = pd.read_csv(backtest_path, usecols=BACKTEST_COLUMNS)
    ats = pd.read_csv(ats_path, usecols=ATS_COLUMNS)
    for frame in (backtest, ats):
        frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
        frame["game_id"] = frame["game_id"].astype("string")
    for column in (
        "predicted_margin",
        "predicted_home_spread",
        "home_score",
        "away_score",
        "actual_margin",
        "margin_error",
    ):
        backtest[column] = pd.to_numeric(backtest[column], errors="coerce")
    for column in (
        "predicted_home_spread",
        "closing_home_spread",
        "model_vs_market_spread",
        "model_edge_home_points",
        "actual_margin",
        "actual_cover_margin",
    ):
        ats[column] = pd.to_numeric(ats[column], errors="coerce")
    for frame, column in (
        (backtest, "gameday"),
        (ats, "gameday"),
        (ats, "schedule_gameday"),
    ):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    backtest["week"] = pd.to_numeric(backtest["week"], errors="coerce")
    return backtest, ats


def _source_cutoff_checks(source_path: Path = BACKTEST_CODE_PATH) -> dict[str, bool]:
    """Statically verify the walk-forward cutoff and prior-play rating path."""
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return {"strict_prior_date_cutoff": False, "ratings_use_prior_plays": False}

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    prior_func = functions.get("_prior_completed_games")
    strict_cutoff = False
    if prior_func is not None:
        for node in ast.walk(prior_func):
            if not isinstance(node, ast.Compare) or len(node.ops) != 1:
                continue
            left = node.left
            if not (
                isinstance(left, ast.Subscript)
                and isinstance(left.value, ast.Name)
                and left.value.id == "completed_games"
                and isinstance(left.slice, ast.Constant)
                and left.slice.value == "_game_date"
            ):
                continue
            if (
                isinstance(node.ops[0], ast.Lt)
                and node.comparators
                and isinstance(node.comparators[0], ast.Name)
                and node.comparators[0].id == "prediction_date"
            ):
                strict_cutoff = True
                break

    season_func = functions.get("backtest_season")
    uses_prior_plays = False
    if season_func is not None:
        has_prior_selector = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prior_completed_games"
            for node in ast.walk(season_func)
        )
        has_prior_rating = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "calculate_rankings"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "prior_plays"
            for node in ast.walk(season_func)
        )
        uses_prior_plays = has_prior_selector and has_prior_rating

    return {
        "strict_prior_date_cutoff": strict_cutoff,
        "ratings_use_prior_plays": uses_prior_plays,
    }


def _append_check(
    rows: list[dict],
    check_id: str,
    status: str,
    finding: str,
    details: str,
) -> None:
    rows.append(
        {
            "record_type": "validation_check",
            "rank": None,
            "check_id": check_id,
            "status": status,
            "finding": finding,
            "details": details,
        }
    )


def _diagnostic_game_row(
    record_type: str,
    rank: int,
    row: pd.Series,
) -> dict:
    return {
        "record_type": record_type,
        "rank": rank,
        "check_id": None,
        "status": "verified_record",
        "finding": (
            "Largest absolute prediction error"
            if record_type == "largest_margin_error"
            else "Largest model-versus-market spread disagreement"
        ),
        "season": int(row["season"]),
        "week": int(row["week"]) if pd.notna(row["week"]) else None,
        "game_id": row["game_id"],
        "gameday": row["gameday"].date().isoformat()
        if pd.notna(row["gameday"])
        else None,
        "schedule_gameday": row["schedule_gameday"].date().isoformat()
        if pd.notna(row["schedule_gameday"])
        else None,
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "predicted_margin": row["predicted_margin"],
        "actual_margin": row["actual_margin"],
        "absolute_margin_error": row["absolute_margin_error"],
        "predicted_home_spread": row["predicted_home_spread"],
        "closing_home_spread": row["closing_home_spread"],
        "model_vs_market_spread": row["model_vs_market_spread"],
        "absolute_spread_disagreement": row["absolute_spread_disagreement"],
        "details": None,
    }


def build_diagnostic_report(
    backtest: pd.DataFrame,
    ats: pd.DataFrame,
    source_path: Path = BACKTEST_CODE_PATH,
) -> pd.DataFrame:
    """Cross-check records and return checks plus the two requested top-25 lists."""
    rows: list[dict] = []
    duplicate_backtest = int(backtest.duplicated(["season", "game_id"], keep=False).sum())
    duplicate_ats = int(ats.duplicated(["season", "game_id"], keep=False).sum())
    _append_check(
        rows,
        "input_counts",
        "verified",
        "Input CSV row counts",
        f"Backtest rows={len(backtest)}; ATS rows={len(ats)}.",
    )
    _append_check(
        rows,
        "duplicate_game_keys",
        "pass" if duplicate_backtest == 0 and duplicate_ats == 0 else "flag",
        "Duplicate season/game_id keys",
        f"Backtest duplicate rows={duplicate_backtest}; ATS duplicate rows={duplicate_ats}.",
    )

    clean_backtest = backtest.drop_duplicates(
        ["season", "game_id"], keep=False
    )
    clean_ats = ats.drop_duplicates(["season", "game_id"], keep=False)
    joined = clean_backtest.merge(
        clean_ats,
        on=["season", "game_id"],
        how="outer",
        suffixes=("_bt", "_ats"),
        indicator=True,
        validate="one_to_one",
    )
    matched = joined.loc[joined["_merge"] == "both"].copy()
    backtest_only = int((joined["_merge"] == "left_only").sum())
    ats_only = int((joined["_merge"] == "right_only").sum())
    _append_check(
        rows,
        "game_id_match_coverage",
        "pass" if backtest_only == 0 and ats_only == 0 else "flag",
        "Backtest/ATS season+game_id matching",
        f"Matched={len(matched)}; backtest-only={backtest_only}; ATS-only={ats_only}.",
    )

    if not matched.empty:
        same_teams = (
            matched["home_team_bt"].astype("string")
            == matched["home_team_ats"].astype("string")
        ) & (
            matched["away_team_bt"].astype("string")
            == matched["away_team_ats"].astype("string")
        )
        same_dates = matched["gameday_bt"] == matched["gameday_ats"]
        schedule_days = (
            matched["schedule_gameday"] - matched["gameday_bt"]
        ).dt.days.abs()
        team_mismatch = int((~same_teams).sum())
        date_mismatch = int((~same_dates).sum())
        schedule_date_over_1 = int((schedule_days > 1).sum())
        _append_check(
            rows,
            "match_identity",
            "pass"
            if team_mismatch == 0 and date_mismatch == 0 and schedule_date_over_1 == 0
            else "flag",
            "Team orientation and date consistency",
            f"Matched rows with different home/away teams={team_mismatch}; "
            f"prediction date mismatches={date_mismatch}; schedule date gaps >1 day={schedule_date_over_1}.",
        )

        model_spread_error = (
            matched["predicted_home_spread_bt"]
            - matched["predicted_home_spread_ats"]
        ).abs()
        actual_margin_error = (
            matched["actual_margin_bt"] - matched["actual_margin_ats"]
        ).abs()
        _append_check(
            rows,
            "prediction_csv_consistency",
            "pass"
            if (model_spread_error <= TOLERANCE).all()
            and (actual_margin_error <= TOLERANCE).all()
            else "flag",
            "Prediction line and actual-margin equality across files",
            f"Model spread mismatches={int((model_spread_error > TOLERANCE).sum())}; "
            f"actual margin mismatches={int((actual_margin_error > TOLERANCE).sum())}.",
        )

        predicted_line_error = (
            matched["predicted_home_spread_bt"] + matched["predicted_margin"]
        ).abs()
        score_margin_error = (
            matched["home_score"] - matched["away_score"] - matched["actual_margin_bt"]
        ).abs()
        prediction_error_error = (
            matched["predicted_margin"] - matched["actual_margin_bt"]
            - matched["margin_error"]
        ).abs()
        _append_check(
            rows,
            "backtest_margin_arithmetic",
            "pass"
            if (predicted_line_error <= TOLERANCE).all()
            and (score_margin_error <= TOLERANCE).all()
            and (prediction_error_error <= TOLERANCE).all()
            else "flag",
            "Predicted/actual margin arithmetic",
            f"Predicted spread != -predicted margin={int((predicted_line_error > TOLERANCE).sum())}; "
            f"score-derived actual margin mismatches={int((score_margin_error > TOLERANCE).sum())}; "
            f"stored margin-error mismatches={int((prediction_error_error > TOLERANCE).sum())}.",
        )

        market_line_formula_error = (
            matched["model_vs_market_spread"]
            - (
                matched["predicted_home_spread_ats"]
                - matched["closing_home_spread"]
            )
        ).abs()
        home_edge_formula_error = (
            matched["model_edge_home_points"]
            + matched["model_vs_market_spread"]
        ).abs()
        cover_formula_error = (
            matched["actual_cover_margin"]
            - (matched["actual_margin_ats"] + matched["closing_home_spread"])
        ).abs()
        cover_expected = np.where(
            matched["actual_cover_margin"] > TOLERANCE,
            "HOME_COVER",
            np.where(
                matched["actual_cover_margin"] < -TOLERANCE,
                "AWAY_COVER",
                "PUSH",
            ),
        )
        cover_label_mismatch = (
            matched["market_ats_outcome"].astype(str).to_numpy() != cover_expected
        )
        _append_check(
            rows,
            "spread_sign_and_ats_arithmetic",
            "pass"
            if (market_line_formula_error <= TOLERANCE).all()
            and (home_edge_formula_error <= TOLERANCE).all()
            and (cover_formula_error <= TOLERANCE).all()
            and not cover_label_mismatch.any()
            else "flag",
            "Signed home-spread convention and ATS arithmetic",
            "Expected convention: negative home spread means home favored; "
            "model_vs_market = model home line - closing home line; "
            "home edge = negative line difference; cover margin = actual margin + signed home line. "
            f"Line-difference mismatches={int((market_line_formula_error > TOLERANCE).sum())}; "
            f"edge-sign mismatches={int((home_edge_formula_error > TOLERANCE).sum())}; "
            f"cover-margin mismatches={int((cover_formula_error > TOLERANCE).sum())}; "
            f"cover-label mismatches={int(cover_label_mismatch.sum())}.",
        )

        days_from_scheduled = (
            matched["schedule_gameday"] - matched["gameday_bt"]
        ).dt.days
        _append_check(
            rows,
            "schedule_date_alignment",
            "pass" if (days_from_scheduled.abs() <= 1).all() else "flag",
            "ATS schedule-date match tolerance",
            f"Absolute schedule/prediction date offset counts: "
            f"0d={(days_from_scheduled == 0).sum()}, "
            f"1d={(days_from_scheduled.abs() == 1).sum()}, "
            f">1d={(days_from_scheduled.abs() > 1).sum()}.",
        )

        merged_for_top = matched.rename(
            columns={
                "week": "week",
                "gameday_bt": "gameday",
                "home_team_bt": "home_team",
                "away_team_bt": "away_team",
                "predicted_margin": "predicted_margin",
                "actual_margin_bt": "actual_margin",
                "predicted_home_spread_bt": "predicted_home_spread",
                "closing_home_spread": "closing_home_spread",
                "model_vs_market_spread": "model_vs_market_spread",
                "schedule_gameday": "schedule_gameday",
                "game_id": "game_id",
                "season": "season",
            }
        ).copy()
        merged_for_top["absolute_margin_error"] = (
            merged_for_top["margin_error"].abs()
        )
        merged_for_top["absolute_spread_disagreement"] = (
            merged_for_top["model_vs_market_spread"].abs()
        )
        for column in (
            "season",
            "week",
            "gameday",
            "schedule_gameday",
            "home_team",
            "away_team",
            "predicted_margin",
            "actual_margin",
            "absolute_margin_error",
            "predicted_home_spread",
            "closing_home_spread",
            "model_vs_market_spread",
            "absolute_spread_disagreement",
            "game_id",
        ):
            if column not in merged_for_top:
                raise ValueError(f"Unable to construct diagnostic field {column}")

        for record_type, value_column in (
            ("largest_margin_error", "absolute_margin_error"),
            ("largest_market_disagreement", "absolute_spread_disagreement"),
        ):
            top = merged_for_top.nlargest(25, value_column).reset_index(drop=True)
            rows.extend(
                _diagnostic_game_row(record_type, index + 1, row)
                for index, row in top.iterrows()
            )

    cutoff_checks = _source_cutoff_checks(source_path)
    cutoff_ok = all(cutoff_checks.values())
    _append_check(
        rows,
        "pregame_information_cutoff",
        "verified_from_source" if cutoff_ok else "not_verified",
        "Walk-forward source code cutoff review",
        "Static AST check of nfl_backtest.py: "
        f"strict prior-date comparison={cutoff_checks['strict_prior_date_cutoff']}; "
        f"ratings calculated from prior_plays after prior-game selection={cutoff_checks['ratings_use_prior_plays']}. "
        "The backtest excludes same-calendar-day games; the CSVs do not store per-game source-game IDs "
        "or a prediction timestamp, so individual predictions cannot be independently replayed from these CSVs.",
    )
    _append_check(
        rows,
        "closing_spread_convention_source",
        "documented",
        "Documented spread-line semantics",
        "nfl_ats_backtest.py documents nflverse/nflfastR spread_line as a closing spread sourced from "
        "Pro-Football-Reference and converts its positive-home-favorite encoding to signed home line "
        "(negative means home favored). CSV arithmetic is checked separately above.",
    )
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def run_diagnostics() -> pd.DataFrame:
    backtest, ats = _read_inputs()
    report = build_diagnostic_report(backtest, ats)
    report.to_csv(REPORT_PATH, index=False)
    print(f"Diagnostic report saved to {REPORT_PATH}")
    checks = report.loc[report["record_type"] == "validation_check"]
    print(checks[["check_id", "status", "details"]].to_string(index=False))
    top_error = report.loc[report["record_type"] == "largest_margin_error"]
    top_spread = report.loc[report["record_type"] == "largest_market_disagreement"]
    print(
        f"Top-game rows: {len(top_error)} largest margin errors; "
        f"{len(top_spread)} largest market disagreements."
    )
    return report


if __name__ == "__main__":
    run_diagnostics()
