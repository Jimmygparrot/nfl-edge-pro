import warnings
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from math import isfinite
from zoneinfo import ZoneInfo

import nflreadpy as nfl

INJURY_WEIGHTS = {
    "QB": -5.0,
    "RB": -0.5,
    "WR": -1.0,
    "TE": -0.5,
    "OL": -0.8,
    "DL": -0.7,
    "LB": -0.5,
    "CB": -0.7,
    "S": -0.5,
}

STATUS_WEIGHTS = {
    "OUT": 1.0,
    "DOUBTFUL": 0.75,
    "QUESTIONABLE": 0.5,
}

POSITION_GROUPS = {
    "T": "OL",
    "G": "OL",
    "C": "OL",
    "DT": "DL",
    "DE": "DL",
}

InjuryReports = dict[tuple[int, int, str], dict[str, tuple[str, str]]]
InjuryReportHistory = dict[
    tuple[int, int, str],
    dict[str, list[tuple[str, str | None, datetime | None]]],
]
SnapHistory = dict[tuple[str, str], list[tuple[int, float | None, float | None]]]
QBStats = list[tuple[datetime, str, str, str, float, int]]
QBReportHistory = dict[
    tuple[int, int, str],
    dict[str, list[tuple[str | None, datetime | None]]],
]
InjuryReportWeeks = dict[tuple[int, str], set[int]]

QB_PRIOR_DROPBACKS = 200
QB_STATS_SEASONS = 3
QB_ADJUSTMENT_SCALE: float | None = None
OFFENSIVE_POSITIONS = {"QB", "RB", "WR", "TE", "OL"}
SNAP_LOOKBACK_GAMES = 5


@dataclass
class QBContext:
    depth_charts: dict[str, list[tuple[datetime, str, int]]]
    roster_players: dict[tuple[int, int, str], set[str]]
    passing_stats: QBStats


@dataclass
class QBAvailabilityContext:
    depth_charts: dict[str, list[tuple[datetime, str, int]]]
    roster_statuses: dict[tuple[int, int, str], dict[str, str]]
    roster_names: dict[tuple[int, int, str], dict[str, str]]
    reserve_descriptions: dict[tuple[int, int, str], dict[str, str]]
    roster_weeks: dict[tuple[int, str], set[int]] = field(default_factory=dict)
    player_names: dict[str, str] = field(default_factory=dict)
    roster_player_teams: dict[tuple[int, int, str], set[str]] = field(
        default_factory=dict
    )


@dataclass
class PointInTimeInjuryAdjustment:
    """Adjustment result; coverage describes feed rows, not the full roster."""

    adjustment: float
    coverage: str
    warnings: list[str]


def _as_utc_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_utc_datetime_if_aware(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _schedule_kickoff(row: dict) -> datetime | None:
    gameday = row.get("gameday")
    gametime = row.get("gametime")
    if gameday is None:
        return None

    if isinstance(gameday, datetime):
        day = gameday.date()
    else:
        try:
            day = datetime.fromisoformat(str(gameday)[:10]).date()
        except ValueError:
            return None

    if gametime is None:
        return None
    if isinstance(gametime, time):
        local_time = gametime
    else:
        try:
            local_time = time.fromisoformat(str(gametime)[:5])
        except ValueError:
            return None

    return datetime.combine(day, local_time).replace(
        tzinfo=ZoneInfo("America/New_York")
    ).astimezone(timezone.utc)


def _game_is_complete(row: dict) -> bool:
    try:
        return isfinite(float(row["home_score"])) and isfinite(
            float(row["away_score"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def load_qb_context(season: int, current_games) -> QBContext | None:
    first_stats_season = max(2009, season - QB_STATS_SEASONS)
    prior_seasons = list(range(first_stats_season, season))
    stats_seasons = prior_seasons + [season]

    try:
        depth_charts = nfl.load_depth_charts(season).select(
            ["dt", "team", "gsis_id", "pos_abb", "pos_rank"]
        )
        rosters = nfl.load_rosters_weekly(season).select(
            ["season", "week", "team", "position", "gsis_id", "status"]
        )
        player_stats = nfl.load_player_stats(
            stats_seasons,
            summary_level="week",
        ).select(
            [
                "season",
                "week",
                "season_type",
                "game_id",
                "team",
                "player_id",
                "position",
                "attempts",
                "sacks_suffered",
                "passing_epa",
            ]
        )
        prior_schedules = (
            nfl.load_schedules(prior_seasons)
            if prior_seasons
            else None
        )
    except Exception as error:
        warnings.warn(
            f"Unable to load QB depth, roster, or passing data; "
            f"existing QB injury penalties will be retained: {error}",
            RuntimeWarning,
        )
        return None

    schedule_rows = (
        prior_schedules.to_dicts() if prior_schedules is not None else []
    )
    schedule_rows.extend(current_games.to_dict("records"))
    completed_games = {}
    for row in schedule_rows:
        if not _game_is_complete(row):
            continue
        kickoff = _schedule_kickoff(row)
        if kickoff is not None:
            completed_games[(int(row["season"]), row["game_id"])] = kickoff

    chart_entries: dict[str, list[tuple[datetime, str, int]]] = {}
    for row in depth_charts.iter_rows(named=True):
        if (
            row["pos_abb"] != "QB"
            or not isinstance(row["team"], str)
            or not isinstance(row["gsis_id"], str)
            or row["pos_rank"] is None
        ):
            continue
        timestamp = _as_utc_datetime(row["dt"])
        if timestamp is None:
            continue
        chart_entries.setdefault(row["team"], []).append(
            (timestamp, row["gsis_id"], int(row["pos_rank"]))
        )

    roster_players: dict[tuple[int, int, str], set[str]] = {}
    for row in rosters.iter_rows(named=True):
        if (
            row["position"] != "QB"
            or row["status"] != "ACT"
            or row["week"] is None
            or not isinstance(row["team"], str)
            or not isinstance(row["gsis_id"], str)
        ):
            continue
        key = (int(row["season"]), int(row["week"]), row["team"])
        roster_players.setdefault(key, set()).add(row["gsis_id"])

    passing_stats: QBStats = []
    for row in player_stats.iter_rows(named=True):
        if (
            row["season_type"] != "REG"
            or row["position"] != "QB"
            or not isinstance(row["team"], str)
            or not isinstance(row["player_id"], str)
            or row["game_id"] is None
            or row["passing_epa"] is None
        ):
            continue
        kickoff = completed_games.get((int(row["season"]), row["game_id"]))
        if kickoff is None:
            continue
        try:
            dropbacks = int(row["attempts"] or 0) + int(
                row["sacks_suffered"] or 0
            )
            epa = float(row["passing_epa"])
        except (TypeError, ValueError):
            continue
        if dropbacks <= 0 or not isfinite(epa):
            continue
        passing_stats.append(
            (
                kickoff,
                row["game_id"],
                row["team"],
                row["player_id"],
                epa,
                dropbacks,
            )
        )

    return QBContext(chart_entries, roster_players, passing_stats)


def load_qb_availability_context(season: int) -> QBAvailabilityContext | None:
    try:
        depth_charts = nfl.load_depth_charts(season)
        rosters = nfl.load_rosters_weekly(season)
    except Exception as error:
        warnings.warn(
            f"Unable to load {season} QB depth charts or weekly rosters; "
            f"availability diagnostics will be incomplete: {error}",
            RuntimeWarning,
        )
        return None

    chart_columns = {"dt", "team", "gsis_id", "pos_abb", "pos_rank"}
    missing_chart_columns = chart_columns.difference(depth_charts.columns)
    if missing_chart_columns:
        warnings.warn(
            "QB depth-chart data is missing required columns "
            f"{sorted(missing_chart_columns)}; QB starter diagnostics "
            "will be unavailable.",
            RuntimeWarning,
        )
        return None

    roster_columns = {
        "season",
        "week",
        "team",
        "position",
        "gsis_id",
        "status",
    }
    missing_roster_columns = roster_columns.difference(rosters.columns)
    if missing_roster_columns:
        warnings.warn(
            "Weekly roster data is missing required columns "
            f"{sorted(missing_roster_columns)}; QB roster status will be "
            "unverified.",
            RuntimeWarning,
        )

    chart_entries: dict[str, list[tuple[datetime, str, int]]] = {}
    for row in depth_charts.iter_rows(named=True):
        if (
            row["pos_abb"] != "QB"
            or not isinstance(row["team"], str)
            or not isinstance(row["gsis_id"], str)
            or row["pos_rank"] is None
        ):
            continue
        timestamp = _as_utc_datetime(row["dt"])
        if timestamp is None:
            continue
        chart_entries.setdefault(row["team"], []).append(
            (timestamp, row["gsis_id"], int(row["pos_rank"]))
        )

    roster_statuses: dict[tuple[int, int, str], dict[str, str]] = {}
    roster_names: dict[tuple[int, int, str], dict[str, str]] = {}
    reserve_descriptions: dict[tuple[int, int, str], dict[str, str]] = {}
    roster_weeks: dict[tuple[int, str], set[int]] = {}
    roster_player_teams: dict[tuple[int, int, str], set[str]] = {}
    if not missing_roster_columns:
        for row in rosters.iter_rows(named=True):
            if (
                row["week"] is not None
                and isinstance(row["team"], str)
                and row["season"] is not None
            ):
                if isinstance(row["gsis_id"], str) and row["gsis_id"].strip():
                    roster_player_teams.setdefault(
                        (int(row["season"]), int(row["week"]), row["gsis_id"]),
                        set(),
                    ).add(row["team"])
                roster_weeks.setdefault(
                    (int(row["season"]), row["team"]), set()
                ).add(int(row["week"]))
            if (
                row["position"] != "QB"
                or row["week"] is None
                or not isinstance(row["team"], str)
                or not isinstance(row["gsis_id"], str)
            ):
                continue
            key = (int(row["season"]), int(row["week"]), row["team"])
            player_id = row["gsis_id"]
            status = row["status"]
            if isinstance(status, str) and status.strip():
                roster_statuses.setdefault(key, {})[player_id] = status.strip().upper()
            name = row.get("full_name")
            if isinstance(name, str) and name.strip():
                roster_names.setdefault(key, {})[player_id] = name.strip()
            reserve_description = row.get("status_description_abbr")
            if isinstance(reserve_description, str) and reserve_description.strip():
                reserve_descriptions.setdefault(key, {})[player_id] = (
                    reserve_description.strip()
                )

    player_names: dict[str, str] = {}
    try:
        players = nfl.load_players()
    except Exception as error:
        warnings.warn(
            f"Unable to load player names for {season}; QB names may fall "
            f"back to their GSIS IDs: {error}",
            RuntimeWarning,
        )
    else:
        if {"gsis_id", "display_name"}.issubset(players.columns):
            player_names = {
                row["gsis_id"]: row["display_name"].strip()
                for row in players.iter_rows(named=True)
                if isinstance(row["gsis_id"], str)
                and row["gsis_id"].strip()
                and isinstance(row["display_name"], str)
                and row["display_name"].strip()
            }

    return QBAvailabilityContext(
        chart_entries,
        roster_statuses,
        roster_names,
        reserve_descriptions,
        roster_weeks,
        player_names,
        roster_player_teams,
    )


def _point_in_time_history_from_frame(
    season: int,
    injuries,
) -> InjuryReportHistory:
    required_columns = {
        "season",
        "game_type",
        "team",
        "week",
        "full_name",
        "position",
        "report_status",
    }
    missing_columns = required_columns.difference(injuries.columns)
    if missing_columns:
        warnings.warn(
            "Point-in-time NFL injury data is missing required columns "
            f"{sorted(missing_columns)}; coverage will be unknown.",
            RuntimeWarning,
        )
        return {}

    report_history: InjuryReportHistory = {}
    has_season_type = "season_type" in injuries.columns
    for row in injuries.iter_rows(named=True):
        if (
            row["season"] != season
            or row["game_type"] != "REG"
            or (has_season_type and row["season_type"] != "REG")
        ):
            continue

        team = row["team"]
        week = row["week"]
        position_value = row["position"]
        if (
            not isinstance(team, str)
            or week is None
            or not isinstance(position_value, str)
        ):
            continue

        position = POSITION_GROUPS.get(
            position_value.strip().upper(),
            position_value.strip().upper(),
        )
        if position not in INJURY_WEIGHTS:
            continue

        player_id = row.get("gsis_id") or row.get("full_name")
        if not isinstance(player_id, str) or not player_id.strip():
            continue

        status_value = row["report_status"]
        status = (
            status_value.strip().upper()
            if isinstance(status_value, str) and status_value.strip()
            else None
        )
        key = (season, int(week), team)
        report_history.setdefault(key, {}).setdefault(player_id, []).append(
            (
                position,
                status,
                _as_utc_datetime_if_aware(row.get("date_modified")),
            )
        )

    return report_history


def _load_injury_report_data(
    season: int,
) -> tuple[InjuryReports, QBReportHistory, InjuryReportWeeks, InjuryReportHistory]:
    try:
        injuries = nfl.load_injuries(season)
    except Exception as error:
        warnings.warn(
            f"Unable to load {season} NFL injury reports; "
            f"injury adjustments will be zero: {error}",
            RuntimeWarning,
        )
        return {}, {}, {}, {}

    if injuries.is_empty():
        warnings.warn(
            f"No {season} NFL injury reports were returned; "
            "injury adjustments will be zero.",
            RuntimeWarning,
        )
        return {}, {}, {}, {}

    required_columns = {
        "season",
        "season_type",
        "game_type",
        "team",
        "week",
        "gsis_id",
        "full_name",
        "position",
        "report_status",
    }
    missing_columns = required_columns.difference(injuries.columns)
    if missing_columns:
        warnings.warn(
            "Legacy NFL injury data is missing required columns "
            f"{sorted(missing_columns)}; legacy injury adjustments will be zero.",
            RuntimeWarning,
        )
        return (
            {},
            {},
            {},
            _point_in_time_history_from_frame(season, injuries),
        )

    reports: InjuryReports = {}
    qb_report_history: QBReportHistory = {}
    injury_report_weeks: InjuryReportWeeks = {}
    injury_report_history: InjuryReportHistory = {}
    status_severity = {
        status: severity
        for severity, status in enumerate(
            ("QUESTIONABLE", "DOUBTFUL", "OUT"),
            start=1,
        )
    }

    for row in injuries.iter_rows(named=True):
        if (
            row["season"] != season
            or row["season_type"] != "REG"
            or row["game_type"] != "REG"
        ):
            continue

        team = row["team"]
        week = row["week"]
        position_value = row["position"]
        if (
            not isinstance(team, str)
            or week is None
            or not isinstance(position_value, str)
        ):
            continue

        position = position_value.strip().upper()
        player_id = row["gsis_id"] or row["full_name"]
        if not isinstance(player_id, str) or not player_id.strip():
            continue

        key = (season, int(week), team)
        injury_report_weeks.setdefault((season, team), set()).add(int(week))
        status_value = row["report_status"]
        status = (
            status_value.strip().upper()
            if isinstance(status_value, str) and status_value.strip()
            else None
        )
        position = POSITION_GROUPS.get(position, position)
        if position not in INJURY_WEIGHTS:
            continue

        injury_report_history.setdefault(key, {}).setdefault(
            player_id, []
        ).append(
            (
                position,
                status,
                _as_utc_datetime_if_aware(row.get("date_modified")),
            )
        )

        if position == "QB":
            qb_report_history.setdefault(key, {}).setdefault(
                player_id, []
            ).append((status, _as_utc_datetime(row.get("date_modified"))))

        if status not in STATUS_WEIGHTS:
            continue

        players = reports.setdefault(key, {})
        existing = players.get(player_id)
        if (
            existing is None
            or status_severity[status] > status_severity[existing[1]]
        ):
            players[player_id] = (position, status)

    return reports, qb_report_history, injury_report_weeks, injury_report_history


def load_injury_report_data(
    season: int,
) -> tuple[InjuryReports, QBReportHistory, InjuryReportWeeks]:
    reports, qb_report_history, injury_report_weeks, _ = _load_injury_report_data(
        season
    )
    return reports, qb_report_history, injury_report_weeks


def load_injury_report_history(season: int) -> InjuryReportHistory:
    """Load injury-feed rows with timestamps preserved for point-in-time use."""
    _, _, _, report_history = _load_injury_report_data(season)
    return report_history


def load_injury_report_comparison_data(
    season: int,
) -> tuple[InjuryReports, InjuryReportHistory]:
    """Load legacy and timestamped injury views from one feed snapshot."""
    reports, _, _, report_history = _load_injury_report_data(season)
    return reports, report_history


def _qb_report_status_as_of(
    report_rows: list[tuple[str | None, datetime | None]],
    kickoff: datetime,
) -> tuple[str | None, str]:
    timestamped = [
        (status, modified)
        for status, modified in report_rows
        if modified is not None
    ]
    eligible = [
        (status, modified)
        for status, modified in timestamped
        if modified <= kickoff
    ]
    if eligible:
        status, _ = max(eligible, key=lambda row: row[1])
        return status, "timestamped report available before kickoff"
    if timestamped:
        return None, "only post-kickoff report updates are available"
    return None, "report timestamp unavailable"


def _qb_roster_label(
    status: str | None,
    reserve_description: str | None,
) -> tuple[str, bool]:
    if status == "ACT":
        return "Active roster (ACT)", True
    if status == "RES":
        normalized_description = (
            reserve_description.strip().upper().replace("_", " ")
            if reserve_description
            else ""
        )
        if normalized_description in {"IR", "INJURED RESERVE"}:
            return "Injured reserve (RES)", False
        return "Reserve (RES; subtype unverified)", False
    if status == "RSN":
        return "Non-football injury reserve (RSN)", False
    if status in {
        "INA",
        "PUP",
        "SUS",
        "EXE",
        "CUT",
        "NWT",
        "RET",
        "UFA",
        "TRC",
        "TRD",
        "TRL",
        "TRT",
    }:
        return f"Unavailable roster status ({status})", False
    if status == "DEV":
        return "Practice squad (DEV; game eligibility unverified)", False
    if status:
        return f"Roster status {status} (availability unverified)", False
    return "Roster status unavailable", False


def get_qb_availability(
    team: str,
    week: int,
    season: int,
    kickoff: datetime,
    context: QBAvailabilityContext | None,
    report_history: QBReportHistory,
    injury_report_weeks: InjuryReportWeeks | None = None,
) -> dict[str, str]:
    result = {
        "starter_id": "",
        "starter_name": "Unknown",
        "starter_confidence": "Unconfirmed",
        "starter_status": "Unavailable",
        "replacement_id": "",
        "replacement_name": "Unknown",
        "replacement_status": "Unavailable",
        "uncertainty": "",
        "latest_roster_week": "Unknown",
        "latest_injury_report_week": "Unknown",
        "current_week_roster_coverage": "Missing",
        "current_week_injury_coverage": "Missing",
        "coverage_warning": "",
        "prior_qb_availability_flags": "",
        "team_assignment_warnings": "",
    }
    timestamp = _as_utc_datetime(kickoff)
    key = (season, week, team)
    roster_weeks = (
        context.roster_weeks.get((season, team), set())
        if context is not None
        else set()
    )
    report_weeks = (injury_report_weeks or {}).get((season, team), set())
    eligible_roster_weeks = [
        prior_week for prior_week in roster_weeks if prior_week <= week
    ]
    eligible_report_weeks = [
        prior_week for prior_week in report_weeks if prior_week <= week
    ]
    latest_roster_week = (
        max(eligible_roster_weeks) if eligible_roster_weeks else None
    )
    latest_report_week = (
        max(eligible_report_weeks) if eligible_report_weeks else None
    )
    result.update(
        latest_roster_week=(
            str(latest_roster_week) if latest_roster_week is not None else "None"
        ),
        latest_injury_report_week=(
            str(latest_report_week) if latest_report_week is not None else "None"
        ),
        current_week_roster_coverage=(
            "Available" if week in roster_weeks else "Missing"
        ),
        current_week_injury_coverage=(
            "Available" if week in report_weeks else "Missing"
        ),
    )
    coverage_warnings: list[str] = []
    if latest_roster_week is None:
        coverage_warnings.append("No roster week available")
    elif latest_roster_week < week:
        coverage_warnings.append(
            f"Roster data is from Week {latest_roster_week}; "
            f"Week {week} roster coverage is missing"
        )
    if latest_report_week is None:
        coverage_warnings.append("No injury-report week available")
    elif latest_report_week < week:
        coverage_warnings.append(
            f"Injury reports are from Week {latest_report_week}; "
            f"Week {week} report coverage is missing"
        )
    result["coverage_warning"] = "; ".join(coverage_warnings)
    if context is None or timestamp is None:
        result["uncertainty"] = "QB roster/depth data unavailable"
        return result

    eligible_charts = [
        entry
        for entry in context.depth_charts.get(team, [])
        if entry[0] <= timestamp
    ]
    if not eligible_charts:
        result["uncertainty"] = "No timestamped pre-kickoff depth chart"
        return result

    latest_snapshot = max(entry[0] for entry in eligible_charts)
    depth_order = sorted(
        {
            player_id: rank
            for entry_time, player_id, rank in eligible_charts
            if entry_time == latest_snapshot
        }.items(),
        key=lambda item: (item[1], item[0]),
    )
    if not depth_order:
        result["uncertainty"] = "No QB entries in the latest depth chart"
        return result

    roster = context.roster_statuses.get(key, {})
    names = context.roster_names.get(key, {})
    reserve_descriptions = context.reserve_descriptions.get(key, {})
    reports = report_history.get(key, {})

    def player_name(player_id: str) -> str:
        name = names.get(player_id)
        if name:
            return name
        name = context.player_names.get(player_id)
        if name:
            return name
        prior_roster_keys = sorted(
            (
                roster_key
                for roster_key in context.roster_names
                if roster_key[0] == season
                and roster_key[1] < week
                and roster_key[2] == team
            ),
            key=lambda roster_key: roster_key[1],
            reverse=True,
        )
        for roster_key in prior_roster_keys:
            name = context.roster_names[roster_key].get(player_id)
            if name:
                return name
        return player_id

    assignment_warnings: list[str] = []
    latest_roster_week = (
        max(eligible_roster_weeks) if eligible_roster_weeks else None
    )
    if latest_roster_week is None:
        assignment_warnings.append("Team assignment cannot be verified: no roster data")
    elif latest_roster_week < week:
        assignment_warnings.append(
            f"Team assignment cannot be verified: roster data is stale "
            f"(latest Week {latest_roster_week}; target Week {week})"
        )
    else:
        for player_id, rank in depth_order:
            player_teams = context.roster_player_teams.get(
                (season, latest_roster_week, player_id), set()
            )
            other_teams = sorted(player_teams - {team})
            if other_teams:
                assignment_warnings.append(
                    f"Confirmed team mismatch: {player_name(player_id)} "
                    f"(GSIS {player_id}), depth-chart rank {rank} for {team} "
                    f"but rostered by {', '.join(other_teams)} in Week "
                    f"{latest_roster_week}"
                )
            elif team not in player_teams:
                assignment_warnings.append(
                    f"Team assignment unverified: {player_name(player_id)} "
                    f"(GSIS {player_id}) is absent from the Week "
                    f"{latest_roster_week} roster data"
                )
    result["team_assignment_warnings"] = "; ".join(assignment_warnings)

    def player_details(
        player_id: str,
    ) -> tuple[str, str, bool, bool, str]:
        roster_label, active = _qb_roster_label(
            roster.get(player_id),
            reserve_descriptions.get(player_id),
        )
        report_rows = reports.get(player_id, [])
        report_status, report_source = _qb_report_status_as_of(
            report_rows,
            timestamp,
        )
        if report_status == "OUT":
            availability = "Confirmed absent: OUT report"
            active = False
        elif report_status in {"DOUBTFUL", "QUESTIONABLE"}:
            availability = f"{report_status.title()} (pre-kickoff report)"
        elif report_status is not None:
            availability = f"{report_status.title()} (pre-kickoff report)"
        else:
            availability = roster_label
        if report_rows and report_status is None:
            availability += f"; {report_source}"
        elif report_status is None:
            availability += "; no qualifying pre-kickoff game status"
        roster_status = roster.get(player_id)
        unavailable = (
            roster_status
            in {
                "RES",
                "RSN",
                "INA",
                "PUP",
                "SUS",
                "EXE",
                "CUT",
                "NWT",
                "RET",
                "UFA",
                "TRC",
                "TRD",
                "TRL",
                "TRT",
            }
            or report_status == "OUT"
        )
        return (
            availability,
            roster_label,
            active,
            unavailable,
            report_status or "",
        )

    starter_id = depth_order[0][0]
    (
        starter_status,
        starter_roster_status,
        starter_active,
        starter_unavailable,
        starter_report,
    ) = player_details(starter_id)
    if starter_roster_status.startswith("Injured reserve"):
        starter_status = "Confirmed unavailable: Injured reserve status"
    elif starter_roster_status.startswith("Reserve ("):
        starter_status = "Unavailable: reserve status; subtype unverified"
    elif starter_roster_status.startswith("Unavailable roster"):
        starter_status = starter_roster_status
    result.update(
        starter_id=starter_id,
        starter_name=player_name(starter_id),
        starter_confidence="Projected QB1 (depth-chart rank 1; not game-confirmed)",
        starter_status=starter_status,
    )

    uncertainty: list[str] = []
    if not starter_active and not starter_unavailable:
        uncertainty.append("QB1 roster eligibility is unverified")
    if starter_report in {"DOUBTFUL", "QUESTIONABLE"}:
        uncertainty.append("QB1 availability is uncertain")
    elif not starter_report:
        uncertainty.append("QB1 has no verified pre-kickoff game status")

    replacement_candidate = None
    for player_id, _ in depth_order[1:]:
        status, roster_status, active, unavailable, report_status = (
            player_details(player_id)
        )
        if unavailable:
            continue
        replacement_candidate = (
            player_id,
            status,
            roster_status,
            active,
            report_status,
        )
        break

    if replacement_candidate is not None:
        replacement_id, status, roster_status, active, report_status = (
            replacement_candidate
        )
        if starter_unavailable:
            prefix = "Projected replacement for unavailable QB1"
        elif starter_report in {"DOUBTFUL", "QUESTIONABLE"}:
            prefix = "Contingent replacement; QB1 availability uncertain"
        else:
            prefix = "Next depth-chart candidate; not an expected starter change"
        result.update(
            replacement_id=replacement_id,
            replacement_name=player_name(replacement_id),
            replacement_status=f"{prefix}; {status}",
        )
        if roster_status == "Roster status unavailable" or not active:
            uncertainty.append("Replacement roster eligibility is unverified")
        if report_status in {"DOUBTFUL", "QUESTIONABLE"}:
            uncertainty.append("Replacement availability is uncertain")
        elif not report_status:
            uncertainty.append(
                "Replacement has no verified pre-kickoff game status"
            )
    else:
        result["replacement_status"] = (
            "No eligible replacement could be identified"
        )
        uncertainty.append("No pre-kickoff active replacement found")

    reserve_roster_statuses = {
        "RES",
        "RSN",
    }
    prior_flags: list[str] = []
    for player_id, _ in depth_order[1:]:
        player_flags: list[str] = []
        prior_weeks = sorted(
            {
                roster_week
                for roster_season, roster_week, roster_team in (
                    set(context.roster_statuses)
                    | set(report_history)
                )
                if roster_season == season
                and roster_team == team
                and roster_week < week
            }
        )
        player_name_value = player_name(player_id)
        for prior_week in prior_weeks:
            prior_status = context.roster_statuses.get(
                (season, prior_week, team), {}
            ).get(player_id)
            if prior_status in reserve_roster_statuses:
                description = context.reserve_descriptions.get(
                    (season, prior_week, team), {}
                ).get(player_id)
                detail = (
                    f" ({description})"
                    if description and prior_status == "RES"
                    else ""
                )
                player_flags.append(
                    f"Week {prior_week} roster {prior_status}{detail}"
                )

            prior_report_rows = report_history.get(
                (season, prior_week, team), {}
            ).get(player_id, [])
            if any(status == "OUT" for status, _ in prior_report_rows):
                player_flags.append(f"Week {prior_week} injury report OUT")

        if player_flags:
            prior_flags.append(
                f"{player_name_value} ({player_id}): "
                + ", ".join(dict.fromkeys(player_flags))
            )

    result["prior_qb_availability_flags"] = "; ".join(prior_flags)
    result["uncertainty"] = "; ".join(dict.fromkeys(uncertainty))
    return result


def _smoothed_qb_rates(
    context: QBContext,
    team: str,
    kickoff: datetime,
) -> tuple[dict[str, float], float] | None:
    player_totals: dict[str, list[float]] = {}
    team_games: dict[str, tuple[datetime, int]] = {}
    league_totals: list[float] = [0.0, 0.0]

    for (
        game_kickoff,
        game_id,
        game_team,
        player_id,
        epa,
        dropbacks,
    ) in context.passing_stats:
        if game_kickoff >= kickoff:
            continue
        player_total = player_totals.setdefault(player_id, [0.0, 0.0])
        player_total[0] += epa
        player_total[1] += dropbacks
        league_totals[0] += epa
        league_totals[1] += dropbacks
        if game_team == team:
            previous = team_games.get(game_id)
            team_games[game_id] = (
                game_kickoff,
                (previous[1] if previous is not None else 0) + dropbacks,
            )

    if league_totals[1] <= 0:
        return None

    league_rate = league_totals[0] / league_totals[1]
    player_rates = {
        player_id: (total[0] + QB_PRIOR_DROPBACKS * league_rate)
        / (total[1] + QB_PRIOR_DROPBACKS)
        for player_id, total in player_totals.items()
    }

    recent_team_volumes = [
        volume
        for _, volume in sorted(
            team_games.values(),
            key=lambda game: game[0],
        )[-5:]
    ]
    if recent_team_volumes:
        expected_dropbacks = sum(recent_team_volumes) / len(recent_team_volumes)
    else:
        league_game_volumes: dict[tuple[str, str], tuple[datetime, int]] = {}
        for game_kickoff, game_id, game_team, _, _, dropbacks in context.passing_stats:
            if game_kickoff < kickoff:
                game_key = (game_id, game_team)
                previous = league_game_volumes.get(game_key)
                league_game_volumes[game_key] = (
                    game_kickoff,
                    (previous[1] if previous is not None else 0) + dropbacks,
                )
        if not league_game_volumes:
            return None
        expected_dropbacks = sum(
            volume for _, volume in league_game_volumes.values()
        ) / len(league_game_volumes)

    return player_rates, expected_dropbacks


def _qb_replacement_adjustment(
    team: str,
    week: int,
    season: int,
    kickoff: datetime,
    injury_reports: InjuryReports,
    context: QBContext,
    snap_history: SnapHistory,
    gsis_to_pfr: dict[str, str],
    adjustment_scale: float,
) -> tuple[float, set[str]] | None:
    if not isfinite(adjustment_scale) or not 0 <= adjustment_scale <= 1:
        return None

    timestamp = _as_utc_datetime(kickoff)
    if timestamp is None:
        return None

    eligible_charts = [
        entry
        for entry in context.depth_charts.get(team, [])
        if entry[0] <= timestamp
    ]
    if not eligible_charts:
        return None
    latest_snapshot = max(entry[0] for entry in eligible_charts)
    depth_order = sorted(
        {
            player_id: rank
            for entry_time, player_id, rank in eligible_charts
            if entry_time == latest_snapshot
        }.items(),
        key=lambda item: (item[1], item[0]),
    )

    roster = context.roster_players.get((season, week, team))
    if not roster:
        return None
    depth_order = [
        (player_id, rank)
        for player_id, rank in depth_order
        if player_id in roster
    ]
    if not depth_order:
        return None

    team_injuries = injury_reports.get((season, week, team), {})
    starter_id = depth_order[0][0]
    starter_injury = team_injuries.get(starter_id)
    if starter_injury is None or starter_injury[0] != "QB":
        return None
    status = starter_injury[1]
    status_multiplier = STATUS_WEIGHTS.get(status)
    if status_multiplier is None:
        return None

    smoothed_rates = _smoothed_qb_rates(context, team, timestamp)
    if smoothed_rates is None:
        return None
    player_rates, expected_dropbacks = smoothed_rates
    starter_rate = player_rates.get(starter_id)
    if starter_rate is None:
        return None

    remaining_probability = 1.0
    replacement_weighted_rate = 0.0
    replacement_probability = 0.0
    used_players = {starter_id}
    for player_id, _ in depth_order[1:]:
        used_players.add(player_id)
        injury = team_injuries.get(player_id)
        unavailable_probability = (
            STATUS_WEIGHTS.get(injury[1], 0.0)
            if injury is not None and injury[0] == "QB"
            else 0.0
        )
        available_probability = 1.0 - unavailable_probability
        probability = remaining_probability * available_probability
        if probability > 0:
            rate = player_rates.get(player_id)
            if rate is None:
                return None
            replacement_weighted_rate += probability * rate
            replacement_probability += probability
        remaining_probability *= unavailable_probability
        if remaining_probability <= 1e-9:
            break

    if replacement_probability <= 0 or remaining_probability > 1e-9:
        return None

    replacement_rate = replacement_weighted_rate / replacement_probability
    pfr_id = gsis_to_pfr.get(starter_id)
    if pfr_id is None:
        return None
    starter_snaps = [
        offense_pct
        for snap_week, offense_pct, _ in snap_history.get((team, pfr_id), [])
        if snap_week < week and offense_pct is not None
    ][-SNAP_LOOKBACK_GAMES:]
    if not starter_snaps:
        return None
    snap_share = min(max(sum(starter_snaps) / len(starter_snaps), 0.0), 1.0)

    raw_points = (replacement_rate - starter_rate) * expected_dropbacks
    applied_points = (
        min(raw_points, 0.0)
        * status_multiplier
        * snap_share
        * adjustment_scale
    )
    existing_qb_penalty = abs(
        INJURY_WEIGHTS["QB"] * status_multiplier * snap_share
    )
    applied_points = max(applied_points, -existing_qb_penalty)
    return applied_points, used_players

def load_snap_history(season: int) -> tuple[SnapHistory, dict[str, str]]:
    try:
        snap_counts = nfl.load_snap_counts(season)
        player_ids = nfl.load_players()
    except Exception as error:
        warnings.warn(
            f"Unable to load {season} NFL snap counts/player IDs; "
            f"existing injury weights will be retained: {error}",
            RuntimeWarning,
        )
        return {}, {}

    required_columns = {
        "season",
        "game_type",
        "week",
        "team",
        "pfr_player_id",
        "offense_pct",
        "defense_pct",
    }
    missing_columns = required_columns.difference(snap_counts.columns)
    if snap_counts.is_empty() or missing_columns:
        warnings.warn(
            f"Snap counts unavailable or missing columns {sorted(missing_columns)}; "
            "existing injury weights will be retained.",
            RuntimeWarning,
        )
        return {}, {}

    gsis_to_pfr = {
        row["gsis_id"]: row["pfr_id"]
        for row in player_ids.iter_rows(named=True)
        if isinstance(row["gsis_id"], str)
        and row["gsis_id"]
        and isinstance(row["pfr_id"], str)
        and row["pfr_id"]
    }

    weekly = {}
    for row in snap_counts.iter_rows(named=True):
        if (
            row["season"] != season
            or row["game_type"] != "REG"
            or row["week"] is None
            or not isinstance(row["team"], str)
            or not isinstance(row["pfr_player_id"], str)
        ):
            continue

        key = (row["team"], row["pfr_player_id"])
        weekly.setdefault(key, {})[int(row["week"])] = (
            row["offense_pct"],
            row["defense_pct"],
        )

    history = {
        key: [(week, *usage) for week, usage in sorted(weeks.items())]
        for key, weeks in weekly.items()
    }
    return history, gsis_to_pfr


def _recent_snap_share(
    player_id: str,
    team: str,
    week: int,
    position: str,
    snap_history: SnapHistory,
    gsis_to_pfr: dict[str, str],
) -> float | None:
    pfr_id = gsis_to_pfr.get(player_id)
    if pfr_id is None:
        return None

    usage_index = 1 if position in OFFENSIVE_POSITIONS else 2
    previous_usage = [
        row[usage_index]
        for row in snap_history.get((team, pfr_id), [])
        if row[0] < week and row[usage_index] is not None
    ][-SNAP_LOOKBACK_GAMES:]
    if not previous_usage:
        return None

    share = sum(previous_usage) / len(previous_usage)
    return min(max(float(share), 0.0), 1.0)


def load_injury_reports(season: int) -> InjuryReports:
    reports, _, _ = load_injury_report_data(season)
    return reports


def get_injury_adjustment(
    team: str,
    week: int,
    injury_reports: InjuryReports,
    season: int = 2026,
    snap_history: SnapHistory | None = None,
    gsis_to_pfr: dict[str, str] | None = None,
    qb_context: QBContext | None = None,
    kickoff: datetime | None = None,
    qb_adjustment_scale: float | None = None,
) -> float:
    players = injury_reports.get((season, week, team), {})
    adjustment = 0.0
    qb_players_handled: set[str] = set()

    scale = qb_adjustment_scale
    if scale is not None and qb_context is not None and kickoff is not None:
        if snap_history is not None and gsis_to_pfr is not None:
            replacement = _qb_replacement_adjustment(
                team,
                week,
                season,
                kickoff,
                injury_reports,
                qb_context,
                snap_history,
                gsis_to_pfr,
                scale,
            )
            if replacement is not None:
                qb_adjustment, qb_players_handled = replacement
                adjustment += qb_adjustment

    for player_id, (position, status) in players.items():
        if position == "QB" and player_id in qb_players_handled:
            continue

        player_adjustment = INJURY_WEIGHTS[position] * STATUS_WEIGHTS[status]
        if snap_history is not None and gsis_to_pfr is not None:
            snap_share = _recent_snap_share(
                player_id,
                team,
                week,
                position,
                snap_history,
                gsis_to_pfr,
            )
            if snap_share is not None:
                player_adjustment *= snap_share
        adjustment += player_adjustment

    return adjustment


def get_point_in_time_injury_adjustment(
    team: str,
    week: int,
    season: int,
    kickoff: datetime,
    report_history: InjuryReportHistory,
    snap_history: SnapHistory | None = None,
    gsis_to_pfr: dict[str, str] | None = None,
    qb_context: QBContext | None = None,
    qb_adjustment_scale: float | None = None,
) -> PointInTimeInjuryAdjustment:
    """Resolve feed rows as of kickoff without using status severity for ordering."""
    timestamp = _as_utc_datetime_if_aware(kickoff)
    if timestamp is None:
        return PointInTimeInjuryAdjustment(
            0.0,
            "unknown",
            ["Kickoff timestamp is missing or timezone-naive; no injury "
             "adjustment was applied."],
        )

    key = (season, week, team)
    players = report_history.get(key)
    if not players:
        return PointInTimeInjuryAdjustment(
            0.0,
            "unknown",
            [
                f"No injury-report records are available for {team} Week "
                f"{week}; injury coverage is unknown."
            ],
        )

    warnings_list: list[str] = []
    selected_reports: dict[str, tuple[str, str]] = {}
    unknown_players = 0

    for player_id, report_rows in players.items():
        timestamped_rows: list[tuple[str, str | None, datetime]] = []
        has_unverifiable_timestamp = False
        for position, status, modified in report_rows:
            modified_utc = _as_utc_datetime_if_aware(modified)
            if modified_utc is None:
                warnings_list.append(
                    f"{player_id}: injury-report timestamp is unavailable or "
                    "timezone-naive; status coverage is unknown."
                )
                has_unverifiable_timestamp = True
                continue
            if modified_utc > timestamp:
                warnings_list.append(
                    f"{player_id}: excluded post-kickoff injury update "
                    f"({modified_utc.isoformat()})."
                )
                continue
            normalized_position = position.strip().upper()
            normalized_position = POSITION_GROUPS.get(
                normalized_position,
                normalized_position,
            )
            normalized_status = (
                status.strip().upper()
                if isinstance(status, str) and status.strip()
                else None
            )
            timestamped_rows.append(
                (normalized_position, normalized_status, modified_utc)
            )

        if has_unverifiable_timestamp:
            unknown_players += 1
            continue

        if not timestamped_rows:
            warnings_list.append(
                f"{player_id}: no verifiable pre-kickoff injury status; "
                "status coverage is unknown."
            )
            unknown_players += 1
            continue

        latest_timestamp = max(row[2] for row in timestamped_rows)
        latest_rows = [
            row for row in timestamped_rows if row[2] == latest_timestamp
        ]
        latest_statuses = {(row[0], row[1]) for row in latest_rows}
        if len(latest_statuses) != 1:
            warnings_list.append(
                f"{player_id}: conflicting statuses share the latest "
                "timestamp; status coverage is unknown."
            )
            unknown_players += 1
            continue

        position, status = next(iter(latest_statuses))
        if position not in INJURY_WEIGHTS:
            warnings_list.append(
                f"{player_id}: unsupported injury position {position!r}; "
                "status coverage is unknown."
            )
            unknown_players += 1
            continue
        if status not in STATUS_WEIGHTS:
            warnings_list.append(
                f"{player_id}: latest pre-kickoff status is not a supported "
                f"injury status ({status or 'missing'}); coverage is unknown."
            )
            unknown_players += 1
            continue

        selected_reports[player_id] = (position, status)

    selected_report_map: InjuryReports = {
        key: selected_reports,
    }
    adjustment = get_injury_adjustment(
        team,
        week,
        selected_report_map,
        season,
        snap_history,
        gsis_to_pfr,
        qb_context,
        timestamp,
        qb_adjustment_scale,
    )
    coverage = (
        "partial"
        if unknown_players and selected_reports
        else "unknown"
        if unknown_players or not selected_reports
        else "verified"
    )
    return PointInTimeInjuryAdjustment(
        adjustment,
        coverage,
        warnings_list,
    )
