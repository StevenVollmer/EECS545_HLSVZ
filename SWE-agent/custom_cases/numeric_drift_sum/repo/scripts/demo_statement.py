"""Demo: total 100,000 line items of 0.10 each.

Exact answer: 10000.00. Naive float summation drifts and, after rounding to
two decimals, produces a value that is NOT exactly 10000.00.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.reports.statement import statement_total


def main() -> int:
    amounts = [0.10] * 100_000
    displayed = statement_total(amounts)
    expected = 10000.0
    print(f"displayed total: {displayed!r}")
    print(f"expected:        {expected!r}")
    if displayed == expected:
        print("OK: reported total matches the exact sum")
    else:
        print("WARNING: reported total disagrees with the exact sum")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
