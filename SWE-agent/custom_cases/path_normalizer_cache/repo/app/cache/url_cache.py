"""Response cache keyed by URL.

Cache contract (invariant for this codebase):
- All keys are stored in canonical normalized form (see app.net.url.normalize_url).
- Stale-eviction uses the allowlist to decide which entries are pinned and
  which can be dropped.

Callers pass *raw* URLs to ``get`` and ``put``. ``evict_stale`` drops any
cached entry whose URL is not present in the allowlist.
"""

from __future__ import annotations

from app.allowlist.store import AllowList
from app.net.url import normalize_url


class UrlCache:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def put(self, url: str, body: str) -> None:
        self._store[normalize_url(url)] = body

    def get(self, url: str) -> str | None:
        return self._store.get(normalize_url(url))

    def size(self) -> int:
        return len(self._store)

    def evict_stale(self, allowlist: AllowList) -> int:
        """Drop any cached entry not present in the allowlist. Returns the
        number of entries evicted.

        Entries in the cache are already stored in normalized form. Membership
        in the allowlist is the source of truth for what is pinned.
        """
        dropped = 0
        # BUG: uses `AllowList.entries()` list for membership check, but then
        # compares the cache's already-normalized key against that list
        # WITHOUT re-running it through ``is_allowed``. That part is fine on
        # its own. The bug is subtler: the per-host allowlist file that feeds
        # AllowList is loaded as raw strings with trailing slashes; AllowList
        # stores them normalized, so `entries()` returns normalized strings.
        # The cache compares keys against those entries directly — but
        # ``AllowList.is_allowed`` is the only path that also normalizes the
        # *input* url. Here we already have normalized keys, so direct
        # comparison should be fine... EXCEPT the cache also stores responses
        # for redirect-target URLs inserted via ``put_redirect`` below, which
        # bypasses normalization. When those redirect-inserted keys are
        # checked, they are raw and never match the normalized allowlist.
        allowed = set(allowlist.entries())
        for key in list(self._store.keys()):
            if key not in allowed:
                del self._store[key]
                dropped += 1
        return dropped

    def put_redirect(self, raw_target_url: str, body: str) -> None:
        """Cache a response under the exact URL we followed a redirect to.

        Upstream callers pass the redirect target exactly as received from the
        network. They expect the cache to store it under a stable key so later
        ``get`` calls hit it.
        """
        # BUG: does not normalize — stores raw key. Breaks the cache-wide
        # invariant that all keys are normalized. Fix requires recognizing
        # this invariant is documented on the class (and in app.net.url) and
        # applying normalize_url here.
        self._store[raw_target_url] = body
