"""Allowlist of URLs that are permitted to be served from cache.

The allowlist is loaded once at startup; entries are run through the canonical
URL normalizer before storage, per the codebase-wide invariant that all cache
keys and allowlist entries are stored in normalized form.

Callers pass *raw* URLs to ``is_allowed``; the store is responsible for applying
the same canonical form to incoming URLs before membership check.
"""

from __future__ import annotations

from app.net.url import normalize_url


class AllowList:
    def __init__(self, entries: list[str]) -> None:
        # Entries are stored already-normalized.
        self._entries: set[str] = {normalize_url(e) for e in entries}

    def is_allowed(self, url: str) -> bool:
        return normalize_url(url) in self._entries

    def entries(self) -> list[str]:
        return sorted(self._entries)
