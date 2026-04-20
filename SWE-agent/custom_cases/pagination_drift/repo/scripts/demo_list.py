"""Demo: caller requests page_size=100 against a store with 200 rows.

The store silently caps each page at STORE_MAX_PAGE=50. The list endpoint's
termination condition (short-read -> end) fires on the first page because
it requested 100 but received 50. Result: caller sees only 50 of 200 rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.list_endpoint import list_all
from app.store.row_store import RowStore


def main() -> int:
    rows = [{"id": i} for i in range(200)]
    store = RowStore(rows)
    got = list_all(store, page_size=100)
    print(f"expected total rows: 200")
    print(f"got total rows:      {len(got)}")
    if len(got) == 200:
        print("OK: list endpoint returned all rows")
    else:
        print("WARNING: list endpoint did not return all rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
