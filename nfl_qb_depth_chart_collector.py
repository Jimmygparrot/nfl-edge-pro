"""Optional, isolated nflreadpy QB depth-chart projection collector.

The nflverse depth-chart ``dt`` field is the time the data record was loaded,
not the source publication time. This module uses ``dt`` only to select the
latest pre-kickoff projection. Database writes require a separately verified
publication timestamp and evidence supplied by the operator.

PowerShell examples:
    .\\.venv\\Scripts\\python.exe .\\nfl_qb_depth_chart_collector.py --season 2026 --dry-run
    .\\.venv\\Scripts\\python.exe .\\nfl_qb_depth_chart_collector.py --season 2026 --write --source-published-at 2026-11-10T10:00:00-05:00 --publication-url https://example.com/release-metadata --publication-evidence "Verified metadata for this exact chart snapshot says published at 10:00 ET."
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import nflreadpy as nfl
import polars as pl

from nfl_qb_pregame_tracker import (
    DEFAULT_DB,
    DuplicateObservationError,
    Observation,
    add_observation,
)

DEPTH_CHART_RELEASE_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "depth_charts/depth_charts_{season}.rds"
)
EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class ScheduledGame:
    season: int
    week: int
    game_id: str
    kickoff: datetime
    home_team: str
    away_team: str


@dataclass(frozen=True)
class QBProjection:
    season: int
    week: int
    game_id: str
    kickoff: datetime
    team: str
    player_id: str
    player_name: str
    projected_role: str
    depth_rank: int
    chart_loaded_at: datetime


@dataclass(frozen=True)
class CollectionResult:
    projection: QBProjection
    action: str
    note: str


def _parse_aware_timestamp(value: object, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{field_name} is not a valid ISO timestamp") from error
    else:
        raise ValueError(f"{field_name} is missing or is not a timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def _parse_kickoff(row: dict[str, object]) -> datetime:
    game_day = row.get("gameday")
    game_time = row.get("gametime")
    if not isinstance(game_day, str) or not isinstance(game_time, str):
        raise ValueError("schedule row is missing gameday or gametime")
    try:
        parsed_day = date.fromisoformat(game_day[:10])
        parsed_time = time.fromisoformat(game_time[:8])
    except ValueError as error:
        raise ValueError("schedule gameday/gametime could not be parsed") from error
    return datetime.combine(parsed_day, parsed_time, tzinfo=EASTERN).astimezone(
        timezone.utc
    )


def parse_scheduled_games(
    schedule_rows: Iterable[dict[str, object]],
    season: int,
    now: datetime,
) -> tuple[list[ScheduledGame], list[str]]:
    """Keep only regular-season games with a parseable, future kickoff."""
    now_utc = _parse_aware_timestamp(now, "current time")
    games: list[ScheduledGame] = []
    skipped: list[str] = []
    for row in schedule_rows:
        if row.get("season") != season or row.get("game_type") != "REG":
            continue
        game_id = row.get("game_id")
        try:
            kickoff = _parse_kickoff(row)
        except ValueError as error:
            skipped.append(f"{game_id or '<unknown game>'}: {error}")
            continue
        if kickoff <= now_utc:
            skipped.append(f"{game_id}: kickoff is not in the future")
            continue
        week = row.get("week")
        home_team = row.get("home_team")
        away_team = row.get("away_team")
        if (
            not isinstance(game_id, str)
            or not isinstance(week, int)
            or not isinstance(home_team, str)
            or not isinstance(away_team, str)
        ):
            skipped.append(f"{game_id or '<unknown game>'}: incomplete game identity")
            continue
        games.append(
            ScheduledGame(
                season=season,
                week=week,
                game_id=game_id,
                kickoff=kickoff,
                home_team=home_team.upper(),
                away_team=away_team.upper(),
            )
        )
    return games, skipped


def _filter_schedule_week(
    schedule_rows: Iterable[dict[str, object]],
    week: int,
) -> list[dict[str, object]]:
    return [row for row in schedule_rows if row.get("week") == week]


def latest_team_qb_projections(
    chart_rows: Iterable[dict[str, object]],
    team: str,
    game: ScheduledGame,
) -> tuple[list[QBProjection], list[str]]:
    """Return unambiguous QB1/QB2 ranks from the latest loaded pre-kickoff chart."""
    team_rows: list[tuple[datetime, dict[str, object]]] = []
    issues: list[str] = []
    for row in chart_rows:
        if row.get("team") != team or row.get("pos_abb") != "QB":
            continue
        try:
            loaded_at = _parse_aware_timestamp(row.get("dt"), "depth chart dt")
        except ValueError as error:
            issues.append(f"{team}: ignored chart row: {error}")
            continue
        if loaded_at < game.kickoff:
            team_rows.append((loaded_at, row))

    if not team_rows:
        return [], issues + [f"{team}: no QB depth-chart snapshot before kickoff"]

    latest_loaded_at = max(loaded_at for loaded_at, _ in team_rows)
    latest_rows = [
        row for loaded_at, row in team_rows if loaded_at == latest_loaded_at
    ]
    projections: list[QBProjection] = []
    for target_rank, role in ((1, "QB1"), (2, "QB2")):
        candidates: dict[str, dict[str, object]] = {}
        for row in latest_rows:
            try:
                rank = int(row["pos_rank"])
            except (KeyError, TypeError, ValueError):
                continue
            if rank != target_rank:
                continue
            player_id = row.get("gsis_id")
            player_name = row.get("player_name")
            if isinstance(player_id, str) and player_id.strip():
                candidates[player_id.strip()] = {
                    "player_name": (
                        player_name.strip()
                        if isinstance(player_name, str)
                        else ""
                    ),
                }

        if len(candidates) != 1:
            if len(candidates) > 1:
                issues.append(
                    f"{team}: rank {target_rank} is ambiguous in latest snapshot"
                )
            continue

        player_id, details = next(iter(candidates.items()))
        projections.append(
            QBProjection(
                season=game.season,
                week=game.week,
                game_id=game.game_id,
                kickoff=game.kickoff,
                team=team,
                player_id=player_id,
                player_name=str(details["player_name"]),
                projected_role=role,
                depth_rank=target_rank,
                chart_loaded_at=latest_loaded_at,
            )
        )

    if not projections:
        issues.append(f"{team}: no unambiguous QB1/QB2 ranks in latest snapshot")
    return projections, issues


def build_projections(
    schedule_rows: Iterable[dict[str, object]],
    chart_rows: Iterable[dict[str, object]],
    season: int,
    now: datetime,
) -> tuple[list[QBProjection], list[str]]:
    games, skipped = parse_scheduled_games(schedule_rows, season, now)
    rows = list(chart_rows)
    projections: list[QBProjection] = []
    for game in games:
        for team in (game.away_team, game.home_team):
            team_projections, issues = latest_team_qb_projections(
                rows,
                team,
                game,
            )
            projections.extend(team_projections)
            skipped.extend(f"{game.game_id}: {issue}" for issue in issues)
    return projections, skipped


def _make_observation(
    projection: QBProjection,
    source_published_at: datetime,
    publication_url: str,
    publication_evidence: str,
) -> Observation:
    return Observation(
        season=projection.season,
        week=projection.week,
        game_id=projection.game_id,
        kickoff=projection.kickoff,
        team=projection.team,
        player_id=projection.player_id,
        source_type="projected_depth_chart",
        source_url=DEPTH_CHART_RELEASE_URL.format(season=projection.season),
        source_published_at=source_published_at,
        source_evidence=(
            f"Projected {projection.projected_role} (rank "
            f"{projection.depth_rank}) from nflreadpy depth chart; "
            f"player={projection.player_name or projection.player_id}; "
            f"chart record loaded at {projection.chart_loaded_at.isoformat()}. "
            f"Separately verified publication evidence: "
            f"{publication_evidence.strip()}; publication metadata URL: "
            f"{publication_url.strip()}"
        ),
        projected_role=projection.projected_role,
        confirmed_role=None,
    )


def collect(
    projections: Iterable[QBProjection],
    *,
    database_path: str | Path = DEFAULT_DB,
    write: bool = False,
    source_published_at: datetime | None = None,
    publication_url: str | None = None,
    publication_evidence: str | None = None,
    now: datetime | None = None,
) -> list[CollectionResult]:
    """Report projections or append them with independent publication evidence."""
    current_time = _parse_aware_timestamp(
        now or datetime.now(timezone.utc),
        "current time",
    )
    parsed_source_time: datetime | None = None
    source_error: str | None = None
    if source_published_at is not None:
        try:
            parsed_source_time = _parse_aware_timestamp(
                source_published_at,
                "source publication timestamp",
            )
        except ValueError as error:
            source_error = str(error)

    evidence = publication_evidence.strip() if publication_evidence else ""
    results: list[CollectionResult] = []
    for projection in projections:
        if not write:
            results.append(
                CollectionResult(
                    projection,
                    "dry-run",
                    "no database write requested",
                )
            )
            continue
        if source_error:
            results.append(CollectionResult(projection, "skipped", source_error))
            continue
        verified_publication_url = (
            publication_url.strip() if publication_url else ""
        )
        if parsed_source_time is None or not evidence or not verified_publication_url:
            results.append(
                CollectionResult(
                    projection,
                    "skipped",
                    "no separately verified source publication timestamp, URL, "
                    "and evidence; "
                    "nflreadpy dt is only the record-load timestamp",
                )
            )
            continue
        if parsed_source_time >= projection.kickoff:
            results.append(
                CollectionResult(
                    projection,
                    "skipped",
                    "source publication timestamp is not strictly before kickoff",
                )
            )
            continue
        if parsed_source_time > projection.chart_loaded_at:
            results.append(
                CollectionResult(
                    projection,
                    "skipped",
                    "publication timestamp is later than the selected chart record "
                    "load time; snapshot provenance is inconsistent",
                )
            )
            continue
        if parsed_source_time > current_time:
            results.append(
                CollectionResult(
                    projection,
                    "skipped",
                    "source publication timestamp is in the future",
                )
            )
            continue
        try:
            observation_id = add_observation(
                _make_observation(
                    projection,
                    parsed_source_time,
                    verified_publication_url,
                    evidence,
                ),
                database_path,
                retrieved_at=current_time,
            )
        except DuplicateObservationError:
            results.append(
                CollectionResult(projection, "duplicate", "observation already exists")
            )
        except ValueError as error:
            results.append(CollectionResult(projection, "skipped", str(error)))
        else:
            results.append(
                CollectionResult(
                    projection,
                    "stored",
                    f"observation_id={observation_id}",
                )
            )
    return results


def load_inputs(
    season: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Load only the chart and schedule columns needed for projections."""
    charts = nfl.load_depth_charts(season)
    schedule = nfl.load_schedules(season)
    chart_columns = {"dt", "team", "gsis_id", "player_name", "pos_abb", "pos_rank"}
    schedule_columns = {
        "game_id",
        "season",
        "game_type",
        "week",
        "gameday",
        "gametime",
        "away_team",
        "home_team",
    }
    missing_chart_columns = chart_columns.difference(charts.columns)
    missing_schedule_columns = schedule_columns.difference(schedule.columns)
    if missing_chart_columns or missing_schedule_columns:
        raise ValueError(
            "nflreadpy schema mismatch: "
            f"depth charts missing {sorted(missing_chart_columns)}; "
            f"schedules missing {sorted(missing_schedule_columns)}"
        )
    chart_rows = (
        charts.filter(pl.col("pos_abb") == "QB")
        .select(sorted(chart_columns))
        .to_dicts()
    )
    schedule_rows = (
        schedule.select(sorted(schedule_columns))
        .filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
        .to_dicts()
    )
    return schedule_rows, chart_rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print or optionally store projected QB1/QB2 depth-chart observations. "
            "No confirmed-starter roles are created."
        )
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, help="optional NFL week filter")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="print projections only; this is the default",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="append rows only when separate publication-time evidence is supplied",
    )
    parser.add_argument(
        "--source-published-at",
        help="verified publication timestamp for the exact chart snapshot (ISO with timezone)",
    )
    parser.add_argument(
        "--publication-evidence",
        help="explicit evidence for the source publication timestamp",
    )
    parser.add_argument(
        "--publication-url",
        help="URL where the verified publication timestamp can be checked",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.week is not None and not 1 <= args.week <= 22:
            raise ValueError("week must be between 1 and 22")
        schedule_rows, chart_rows = load_inputs(args.season)
        if args.week is not None:
            schedule_rows = _filter_schedule_week(schedule_rows, args.week)
        now = datetime.now(timezone.utc)
        projections, skipped = build_projections(
            schedule_rows,
            chart_rows,
            args.season,
            now,
        )
        try:
            publication_time = (
                _parse_aware_timestamp(
                    args.source_published_at,
                    "source publication timestamp",
                )
                if args.source_published_at
                else None
            )
        except ValueError as error:
            publication_time = None
            publication_error = str(error)
        else:
            publication_error = None

        results = collect(
            projections,
            database_path=args.db,
            write=args.write,
            source_published_at=publication_time,
            publication_url=args.publication_url,
            publication_evidence=args.publication_evidence,
            now=now,
        )
        mode = "WRITE" if args.write else "DRY RUN"
        print(f"{mode}: season {args.season}; projections={len(projections)}")
        for projection, result in zip(projections, results):
            print(
                f"{result.action.upper()}: {projection.game_id} "
                f"{projection.team} {projection.player_name} "
                f"({projection.player_id}) {projection.projected_role}; "
                f"chart_loaded_at={projection.chart_loaded_at.isoformat()}; "
                f"scheduled_kickoff="
                f"{projection.kickoff.astimezone(EASTERN).isoformat()}; "
                f"kickoff_utc={projection.kickoff.isoformat()}; {result.note}"
            )
        for issue in skipped:
            print(f"SKIPPED: {issue}")
        if publication_error:
            print(f"PUBLICATION TIME UNVERIFIED: {publication_error}")
        if args.write and publication_time is None and not publication_error:
            print(
                "No observations were stored because no separately verified "
                "source publication timestamp was supplied."
            )
        if args.write and (
            not args.publication_evidence or not args.publication_url
        ):
            print(
                "Rows lacking a publication URL or explicit publication evidence "
                "were skipped; "
                "nflreadpy dt is not treated as publication time."
            )
        return 0
    except Exception as error:
        print(f"Collector failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
