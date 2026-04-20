"""Demo: diamond-shaped DAG (no cycle) that the current detector misclassifies.

  a -> b -> d
  a -> c -> d

A correct detector returns False. The buggy one returns True because d is
re-entered from c after already being marked from the b branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph.cycle import has_cycle


def main() -> int:
    graph = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
    result = has_cycle(graph)
    print(f"Diamond DAG has_cycle = {result}")
    if result is False:
        print("OK: diamond DAG correctly reported as acyclic")
    else:
        print("WARNING: diamond DAG incorrectly reported as cyclic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
