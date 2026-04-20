"""Demo: a US-format date with asymmetric day/month values.

Operator feeds `03-15-2024` (March 15). The pipeline groups records by
(year, month). Correct output puts this record in group (2024, 3) because
March is month 3. The buggy parser swaps month and day on US format, so it
groups as (2024, 15) — an invalid month.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.group_by_month import group_by_month
from app.pipeline.normalize import normalize_records


def main() -> int:
    records = [{"date": "03-15-2024", "note": "march"}]
    grouped = group_by_month(normalize_records(records))
    keys = sorted(grouped.keys())
    print(f"group keys: {keys}")
    if keys == [(2024, 3)]:
        print("OK: US date grouped into March")
    else:
        print(f"WARNING: US date grouped incorrectly (expected [(2024, 3)])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
