"""Pipeline stage: normalize incoming date strings to (y, m, d) tuples."""

from __future__ import annotations

from app.parsing.date_parser import parse_date


def normalize_records(records: list[dict]) -> list[dict]:
    out = []
    for rec in records:
        rec = dict(rec)
        rec["ymd"] = parse_date(rec["date"])
        out.append(rec)
    return out
