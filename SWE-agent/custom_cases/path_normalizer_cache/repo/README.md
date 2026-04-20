# path_normalizer_cache fixture

Three cooperating modules:

- `app/net/url.py` — the canonical normalizer. Owns the contract that cache keys
  and allowlist entries are stored in normalized form.
- `app/allowlist/store.py` — stores allowlisted URLs already normalized; exposes
  `is_allowed(raw_url)` that normalizes the incoming URL before checking.
- `app/cache/url_cache.py` — response cache keyed by normalized URL. The
  redirect-cache path writes under the raw target URL.
