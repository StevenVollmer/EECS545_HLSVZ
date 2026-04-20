"""Demo: run a search and print the rendered result lines for operator review.

Per spec, the first line must start with `1.` (1-indexed ranking).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.formatters.hit_renderer import render
from app.services.highlight_service import add_highlights
from app.services.search_service import search


def main() -> int:
    query = "widgets"
    hits = search(query)
    hits = add_highlights(hits, query)
    lines = render(hits)
    for line in lines:
        print(line)
    first = lines[0] if lines else ""
    if first.startswith("1."):
        print("OK: first result is rank 1")
    else:
        print(f"WARNING: first result does not start with rank 1 (got: {first[:4]!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
