"""Demo: rank a leaderboard where submission numbers include 2-digit entries.

The intended order when scores tie is by submission_number ASCENDING NUMERICALLY
(1, 2, 10), which is the operator-facing contract documented in the spec.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.leaderboard import build_leaderboard


def main() -> int:
    entries = [
        {"team_id": "alpha", "score": 100, "submission_number": "10"},
        {"team_id": "bravo", "score": 100, "submission_number": "2"},
        {"team_id": "charlie", "score": 100, "submission_number": "1"},
    ]
    ranked = build_leaderboard(entries)
    order = [e["team_id"] for e in ranked]
    print("Leaderboard order:", order)
    expected = ["charlie", "bravo", "alpha"]
    if order == expected:
        print("OK: tie-break order matches the numeric submission order")
    else:
        print("WARNING: tie-break order is wrong; expected", expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
