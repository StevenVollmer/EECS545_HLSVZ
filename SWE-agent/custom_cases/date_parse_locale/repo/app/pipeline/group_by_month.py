"""Pipeline stage: group records by (year, month)."""

from __future__ import annotations
from collections import defaultdict


def group_by_month(records: list[dict]) -> dict[tuple[int, int], list[dict]]:
    out: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in records:
        y, m, _ = r["ymd"]
        out[(y, m)].append(r)
    return dict(out)
