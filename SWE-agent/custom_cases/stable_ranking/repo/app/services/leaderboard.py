from __future__ import annotations

from typing import Any

from app.core.ranking import rank


def build_leaderboard(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rank(entries)
