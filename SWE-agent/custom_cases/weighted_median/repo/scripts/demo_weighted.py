"""Demo: run weighted_median against several operator-realistic inputs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.stats.weighted import weighted_median


CASES = [
    # (pairs, expected)
    ([(1.0, 0.1), (5.0, 0.5), (10.0, 0.4)], 5.0),
    ([(2.0, 0.2), (4.0, 0.2), (6.0, 0.2), (8.0, 0.2), (10.0, 0.2)], 6.0),
    ([(1.0, 3.0), (2.0, 5.0), (3.0, 2.0)], 2.0),
    ([(100.0, 0.9), (200.0, 0.05), (300.0, 0.05)], 100.0),
]


def main() -> int:
    failed = []
    for pairs, expected in CASES:
        got = weighted_median(pairs)
        status = "ok" if got == expected else "BAD"
        print(f"  {status}: weighted_median({pairs}) = {got}  expected {expected}")
        if got != expected:
            failed.append((pairs, expected, got))
    if not failed:
        print("OK: all inputs match expected values")
    else:
        print(f"WARNING: {len(failed)} input(s) disagree with expected values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
