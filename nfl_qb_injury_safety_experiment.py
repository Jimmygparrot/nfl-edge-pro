"""Isolated comparison of fixed and starter-aware QB injury penalties."""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Literal

QB_INJURY_WEIGHT = -5.0
STATUS_MULTIPLIERS = {
    "OUT": 1.0,
    "DOUBTFUL": 0.75,
    "QUESTIONABLE": 0.5,
}

QBRole = Literal["starter", "backup", "unknown"]
RoleClassification = Literal[
    "verified_starter",
    "verified_backup",
    "unknown",
]


@dataclass(frozen=True)
class QBInjuryInput:
    player_id: str
    status: str | None
    recent_snap_share: float | None
    pregame_role: str
    role_source_timestamp: datetime | None


@dataclass(frozen=True)
class QBInjuryComparison:
    player_id: str
    status: str | None
    role_classification: RoleClassification
    role_evidence_note: str
    fixed_penalty: float
    experimental_penalty: float


def _snap_factor(recent_snap_share: float | None) -> float:
    if recent_snap_share is None or not isfinite(recent_snap_share):
        return 1.0
    return min(max(float(recent_snap_share), 0.0), 1.0)


def _role_classification(
    pregame_role: str,
    role_source_timestamp: datetime | None,
    kickoff: datetime,
) -> tuple[RoleClassification, str]:
    role = pregame_role.strip().lower()
    if role not in {"starter", "backup"}:
        return "unknown", "role is unknown or ambiguous"
    if role_source_timestamp is None:
        return "unknown", "role source timestamp is missing"
    if (
        role_source_timestamp.tzinfo is None
        or role_source_timestamp.utcoffset() is None
    ):
        return "unknown", "role source timestamp has no timezone"
    if role_source_timestamp >= kickoff:
        return "unknown", "role source timestamp is not before kickoff"
    return f"verified_{role}", "role source timestamp is before kickoff"


def compare_qb_injury_penalties(
    injuries: list[QBInjuryInput],
    kickoff: datetime,
) -> list[QBInjuryComparison]:
    """Compare current fixed penalties with a role-aware experimental policy.

    Unknown or unverifiable roles retain the fixed penalty; only a verified
    backup has its starter-sized experimental penalty suppressed.
    """
    if kickoff.tzinfo is None or kickoff.utcoffset() is None:
        raise ValueError("kickoff must be timezone-aware")

    player_ids = [injury.player_id for injury in injuries]
    if any(not player_id.strip() for player_id in player_ids):
        raise ValueError("player_id must not be empty")
    if len(player_ids) != len(set(player_ids)):
        raise ValueError("injuries must contain at most one row per player")

    results = []
    for injury in injuries:
        status = (
            injury.status.strip().upper()
            if isinstance(injury.status, str)
            else None
        )
        multiplier = STATUS_MULTIPLIERS.get(status, 0.0)
        fixed_penalty = (
            QB_INJURY_WEIGHT * multiplier * _snap_factor(injury.recent_snap_share)
        )
        role_classification, evidence_note = _role_classification(
            injury.pregame_role,
            injury.role_source_timestamp,
            kickoff,
        )
        if role_classification == "verified_backup":
            experimental_penalty = 0.0
        else:
            experimental_penalty = fixed_penalty

        results.append(
            QBInjuryComparison(
                player_id=injury.player_id,
                status=status,
                role_classification=role_classification,
                role_evidence_note=evidence_note,
                fixed_penalty=fixed_penalty,
                experimental_penalty=experimental_penalty,
            )
        )
    return results


def illustrative_comparisons() -> list[QBInjuryComparison]:
    """Return hypothetical examples; these are not historical evaluations."""
    kickoff = datetime.fromisoformat("2026-11-08T18:00:00+00:00")
    return compare_qb_injury_penalties(
        [
            QBInjuryInput(
                "example-starter",
                "Out",
                1.0,
                "starter",
                datetime.fromisoformat("2026-11-08T16:00:00+00:00"),
            ),
            QBInjuryInput(
                "example-backup",
                "Questionable",
                0.4,
                "backup",
                datetime.fromisoformat("2026-11-08T15:00:00+00:00"),
            ),
            QBInjuryInput(
                "example-unknown",
                "Doubtful",
                None,
                "unknown",
                None,
            ),
        ],
        kickoff,
    )


if __name__ == "__main__":
    print("Illustrative QB injury penalty comparisons (not historical results)")
    print("player_id,role_classification,fixed_penalty,experimental_penalty")
    for comparison in illustrative_comparisons():
        print(
            f"{comparison.player_id},{comparison.role_classification},"
            f"{comparison.fixed_penalty:.2f},{comparison.experimental_penalty:.2f}"
        )
