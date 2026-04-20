"""Storage layer. Returns rows in fixed order and silently caps any requested
page size at STORE_MAX_PAGE for server safety reasons.

Callers that request more rows than STORE_MAX_PAGE receive only the capped
slice; the store does not advertise the cap in the returned payload. This
behavior has been stable since v1 and downstream services depend on it.
"""

from __future__ import annotations

STORE_MAX_PAGE = 50


class RowStore:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = list(rows)

    def fetch_page(self, offset: int, requested: int) -> list[dict]:
        capped = min(requested, STORE_MAX_PAGE)
        return self._rows[offset : offset + capped]

    def count(self) -> int:
        return len(self._rows)
