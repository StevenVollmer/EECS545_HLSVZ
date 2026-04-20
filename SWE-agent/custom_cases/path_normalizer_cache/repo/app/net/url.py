"""URL normalizer used as the canonical-form function across the system.

Normalization rules (the canonical contract for this codebase):
- scheme and host are lowercased
- trailing slash on the path is stripped (except the root "/")
- fragment is dropped
- query string is preserved as-is but key-sorted

All cache keys and allowlist entries in this codebase are stored in the
normalized form produced here.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    query_pairs = sorted(parse_qsl(parts.query, keep_blank_values=True))
    query = urlencode(query_pairs)
    return urlunsplit((scheme, netloc, path, query, ""))
