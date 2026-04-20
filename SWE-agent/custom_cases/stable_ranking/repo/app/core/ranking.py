"""Leaderboard ranking with tie-break rules.

Primary sort: score descending.
Tie-break 1: submission_number ascending (earlier submissions rank higher).
Tie-break 2: team_id ascending (lexicographic on id string).

Entries are dicts with keys: team_id, score, submission_number.
"""

from __future__ import annotations

from typing import Any


def rank(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Sort by (-score, submission_number, team_id).
    # submission_number is stored as a string in the input (legacy import quirk).
    return sorted(
        entries,
        key=lambda e: (-e["score"], e["submission_number"], e["team_id"]),
    )
