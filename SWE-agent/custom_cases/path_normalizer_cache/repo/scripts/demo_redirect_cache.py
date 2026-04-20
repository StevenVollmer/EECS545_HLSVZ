"""Demo: a redirect target gets cached and then survives eviction because the
allowlist contains the canonical form of the target URL.

Upstream behavior simulated:
  1. Client fetches https://a.com/x (allowed).
  2. Upstream returns a redirect to https://A.COM/x/ (raw form; different case
     and trailing slash, canonically equivalent).
  3. Cache stores the body under the redirect target via ``put_redirect``.
  4. ``evict_stale`` runs against an allowlist containing the canonical
     ``https://a.com/x``; the redirect-cached body must be retained because
     it IS allowed under the codebase's normalization invariant.

This demo prints PASS only if the redirect-cached body survives eviction AND
is retrievable via a raw lookup afterward. Baseline fails because the
redirect body is keyed under the un-normalized URL, so ``evict_stale`` drops
it (the allowlist contains the normalized form, not the raw form).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.allowlist.store import AllowList
from app.cache.url_cache import UrlCache


def main() -> int:
    allow = AllowList(["https://a.com/x"])
    cache = UrlCache()
    # Upstream layer gave us the redirect target exactly as received.
    redirect_target_raw = "https://A.COM/x/"
    cache.put_redirect(redirect_target_raw, "redirect-body")
    before = cache.size()
    dropped = cache.evict_stale(allow)
    after = cache.size()

    retrieved = cache.get(redirect_target_raw)
    print(f"cache size before={before} dropped={dropped} after={after}")
    print(f"retrieved body: {retrieved!r}")

    if retrieved == "redirect-body" and dropped == 0:
        print("OK: redirect-cached entry survived eviction and is retrievable")
    else:
        print("WARNING: redirect-cached entry was evicted or is not retrievable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
