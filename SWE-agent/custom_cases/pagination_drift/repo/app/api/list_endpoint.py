"""List endpoint. Pages through RowStore to return results.

The endpoint accepts a caller-specified ``page_size`` and walks until it has
no more rows. Termination condition: the store returns fewer rows than
requested, which the endpoint treats as the final page.
"""

from __future__ import annotations

from app.store.row_store import RowStore


def list_all(store: RowStore, page_size: int) -> list[dict]:
    collected: list[dict] = []
    offset = 0
    while True:
        page = store.fetch_page(offset, page_size)
        collected.extend(page)
        # BUG: treats "fewer than requested" as end-of-data. The store
        # silently caps pages at STORE_MAX_PAGE, so any caller requesting
        # page_size > STORE_MAX_PAGE gets STORE_MAX_PAGE rows on every call
        # and the endpoint halts after the first page, missing all remaining
        # rows. Fix requires aligning with the store's capping invariant —
        # either request within the cap, or detect end-of-data via an
        # explicit empty page rather than short-read.
        if len(page) < page_size:
            break
        offset += len(page)
    return collected
