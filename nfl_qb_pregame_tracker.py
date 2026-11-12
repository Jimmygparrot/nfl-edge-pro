"""Local, timestamp-aware tracker for pregame NFL quarterback observations.

No web requests are made. Sources and evidence are entered manually.

Examples (PowerShell):
    .\\.venv\\Scripts\\python.exe .\\nfl_qb_pregame_tracker.py init
    .\\.venv\\Scripts\\python.exe .\\nfl_qb_pregame_tracker.py add --season 2026 --week 10 --game-id 2026_10_SEA_LAR --kickoff 2026-11-08T18:00:00-05:00 --team SEA --player-id 00-0030565 --source-type official_team_announcement --source-url https://example.com/team-announcement --published-at 2026-11-08T15:00:00-05:00 --evidence "Team statement: QB will start." --confirmed-role starter
    .\\.venv\\Scripts\\python.exe .\\nfl_qb_pregame_tracker.py list --season 2026 --week 10
"""

import argparse
import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

SourceType = Literal[
    "official_team_announcement",
    "projected_depth_chart",
    "official_injury_report",
    "other",
]

SOURCE_TYPES = {
    "official_team_announcement",
    "projected_depth_chart",
    "official_injury_report",
    "other",
}
PROJECTED_ROLES = {"QB1", "QB2", "QB3", "QB4"}
CONFIRMED_ROLES = {"starter", "backup"}
DEFAULT_DB = Path("nfl_qb_observations.sqlite3")

SCHEMA = """
CREATE TABLE IF NOT EXISTS qb_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id TEXT NOT NULL,
    kickoff_utc TEXT NOT NULL,
    team TEXT NOT NULL,
    player_id TEXT NOT NULL,
    projected_role TEXT,
    confirmed_role TEXT,
    injury_status TEXT,
    source_url TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_published_utc TEXT NOT NULL,
    retrieved_utc TEXT NOT NULL,
    source_evidence TEXT NOT NULL,
    verification_status TEXT NOT NULL,
    observation_fingerprint TEXT NOT NULL UNIQUE,
    CHECK (week BETWEEN 1 AND 22),
    CHECK (
        projected_role IS NULL OR
        projected_role IN ('QB1', 'QB2', 'QB3', 'QB4')
    ),
    CHECK (
        confirmed_role IS NULL OR
        confirmed_role IN ('starter', 'backup')
    ),
    CHECK (
        source_type IN (
            'official_team_announcement',
            'projected_depth_chart',
            'official_injury_report',
            'other'
        )
    )
);
CREATE INDEX IF NOT EXISTS idx_qb_observations_game
ON qb_observations(season, week, game_id, team);
"""


@dataclass(frozen=True)
class Observation:
    season: int
    week: int
    game_id: str
    kickoff: datetime
    team: str
    player_id: str
    source_type: str
    source_url: str
    source_published_at: datetime
    source_evidence: str
    projected_role: str | None = None
    confirmed_role: str | None = None
    injury_status: str | None = None


class DuplicateObservationError(ValueError):
    """Raised when an exact observation has already been recorded."""


def _utc_iso(value: datetime, field_name: str) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone offset")
    return value.astimezone(timezone.utc).isoformat()


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _validated_record(
    observation: Observation,
    retrieved_at: datetime,
) -> dict[str, object]:
    kickoff_utc = _utc_iso(observation.kickoff, "kickoff")
    published_utc = _utc_iso(
        observation.source_published_at,
        "source_published_at",
    )
    retrieved_utc = _utc_iso(retrieved_at, "retrieved_at")
    kickoff = datetime.fromisoformat(kickoff_utc)
    published = datetime.fromisoformat(published_utc)

    if published >= kickoff:
        raise ValueError("source publication timestamp must be before kickoff")
    if not 2000 <= observation.season <= 2100:
        raise ValueError("season must be between 2000 and 2100")
    if not 1 <= observation.week <= 22:
        raise ValueError("week must be between 1 and 22")

    game_id = observation.game_id.strip()
    team = observation.team.strip().upper()
    player_id = observation.player_id.strip()
    source_url = observation.source_url.strip()
    evidence = observation.source_evidence.strip()
    source_type = observation.source_type.strip().lower()
    projected_role = _normalize_optional(observation.projected_role)
    confirmed_role = _normalize_optional(observation.confirmed_role)
    injury_status = _normalize_optional(observation.injury_status)

    if not all((game_id, team, player_id, source_url, evidence)):
        raise ValueError(
            "game_id, team, player_id, source_url, and source_evidence are required"
        )
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"unsupported source_type: {source_type}")
    if projected_role is not None:
        projected_role = projected_role.upper()
        if projected_role not in PROJECTED_ROLES:
            raise ValueError("projected_role must be QB1, QB2, QB3, or QB4")
        if source_type != "projected_depth_chart":
            raise ValueError(
                "projected_role requires source_type projected_depth_chart"
            )
    if confirmed_role is not None:
        confirmed_role = confirmed_role.lower()
        if confirmed_role not in CONFIRMED_ROLES:
            raise ValueError("confirmed_role must be starter or backup")
        if source_type != "official_team_announcement":
            raise ValueError(
                "confirmed_role requires source_type official_team_announcement"
            )
    if projected_role is None and confirmed_role is None and injury_status is None:
        raise ValueError(
            "observation must include a projected role, confirmed role, "
            "or injury status"
        )

    if confirmed_role is not None:
        verification_status = "role_timestamp_validated"
    elif projected_role is not None:
        verification_status = "projected_only"
    elif source_type == "official_injury_report":
        verification_status = "pregame_timestamp_validated"
    else:
        verification_status = "timestamped_source_unverified"

    canonical = {
        "season": observation.season,
        "week": observation.week,
        "game_id": game_id,
        "kickoff_utc": kickoff_utc,
        "team": team,
        "player_id": player_id,
        "projected_role": projected_role,
        "confirmed_role": confirmed_role,
        "injury_status": injury_status,
        "source_url": source_url,
        "source_type": source_type,
        "source_published_utc": published_utc,
        "source_evidence": evidence,
    }
    fingerprint = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        **canonical,
        "retrieved_utc": retrieved_utc,
        "verification_status": verification_status,
        "observation_fingerprint": fingerprint,
    }


def initialize_database(database_path: str | Path = DEFAULT_DB) -> None:
    path = Path(database_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(SCHEMA)


def add_observation(
    observation: Observation,
    database_path: str | Path = DEFAULT_DB,
    *,
    retrieved_at: datetime | None = None,
) -> int:
    """Validate and append an observation; exact duplicates are rejected."""
    initialize_database(database_path)
    record = _validated_record(
        observation,
        retrieved_at or datetime.now(timezone.utc),
    )
    columns = tuple(record)
    placeholders = ", ".join("?" for _ in columns)
    sql = (
        f"INSERT INTO qb_observations ({', '.join(columns)}) "
        f"VALUES ({placeholders})"
    )
    try:
        with closing(sqlite3.connect(database_path)) as connection:
            with connection:
                cursor = connection.execute(
                    sql,
                    tuple(record[column] for column in columns),
                )
                observation_id = int(cursor.lastrowid)
            return observation_id
    except sqlite3.IntegrityError as error:
        if "observation_fingerprint" in str(error):
            raise DuplicateObservationError(
                "exact observation already exists; prior history was preserved"
            ) from error
        raise


def list_observations(
    database_path: str | Path = DEFAULT_DB,
    *,
    season: int | None = None,
    week: int | None = None,
    game_id: str | None = None,
    team: str | None = None,
) -> list[dict[str, object]]:
    initialize_database(database_path)
    clauses = []
    parameters: list[object] = []
    for column, value in (
        ("season", season),
        ("week", week),
        ("game_id", game_id),
        ("team", team.upper() if team else None),
    ):
        if value is not None:
            clauses.append(f"{column} = ?")
            parameters.append(value)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    query = (
        "SELECT observation_id, season, week, game_id, kickoff_utc, team, "
        "player_id, projected_role, confirmed_role, injury_status, source_url, "
        "source_type, source_published_utc, retrieved_utc, source_evidence, "
        "verification_status FROM qb_observations"
        f"{where} ORDER BY source_published_utc, observation_id"
    )
    with closing(sqlite3.connect(database_path)) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(query, parameters)]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manually track timestamped pregame QB observations in local SQLite. "
            "No websites are scraped."
        )
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="create the local database and table")

    add = commands.add_parser("add", help="append one manually verified observation")
    add.add_argument("--season", type=int, required=True)
    add.add_argument("--week", type=int, required=True)
    add.add_argument("--game-id", required=True)
    add.add_argument("--kickoff", required=True, help="ISO timestamp with timezone")
    add.add_argument("--team", required=True)
    add.add_argument("--player-id", required=True)
    add.add_argument(
        "--source-type",
        required=True,
        choices=sorted(SOURCE_TYPES),
    )
    add.add_argument("--source-url", required=True)
    add.add_argument(
        "--published-at",
        required=True,
        help="source publication ISO timestamp with timezone",
    )
    add.add_argument("--evidence", required=True, help="explicit source excerpt")
    add.add_argument(
        "--projected-role",
        choices=sorted(PROJECTED_ROLES),
        help="depth-chart projection only, never a confirmed starter",
    )
    add.add_argument("--confirmed-role", choices=sorted(CONFIRMED_ROLES))
    add.add_argument("--injury-status")

    listing = commands.add_parser("list", help="inspect stored observations")
    listing.add_argument("--season", type=int)
    listing.add_argument("--week", type=int)
    listing.add_argument("--game-id")
    listing.add_argument("--team")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            initialize_database(args.db)
            print(f"Initialized {args.db}")
            return 0
        if args.command == "add":
            observation = Observation(
                season=args.season,
                week=args.week,
                game_id=args.game_id,
                kickoff=datetime.fromisoformat(args.kickoff.replace("Z", "+00:00")),
                team=args.team,
                player_id=args.player_id,
                source_type=args.source_type,
                source_url=args.source_url,
                source_published_at=datetime.fromisoformat(
                    args.published_at.replace("Z", "+00:00")
                ),
                source_evidence=args.evidence,
                projected_role=args.projected_role,
                confirmed_role=args.confirmed_role,
                injury_status=args.injury_status,
            )
            observation_id = add_observation(observation, args.db)
            print(f"Stored observation {observation_id} in {args.db}")
            return 0
        rows = list_observations(
            args.db,
            season=args.season,
            week=args.week,
            game_id=args.game_id,
            team=args.team,
        )
        if not rows:
            print("No observations found.")
            return 0
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        return 0
    except (ValueError, DuplicateObservationError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
