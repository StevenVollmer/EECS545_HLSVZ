"""Pipeline stage: keep only records with year >= cutoff."""

from __future__ import annotations


def filter_recent(records: list[dict], cutoff_year: int) -> list[dict]:
    return [r for r in records if r["ymd"][0] >= cutoff_year]
