from app.allowlist.store import AllowList
from app.cache.url_cache import UrlCache


def test_put_and_get_round_trip():
    c = UrlCache()
    c.put("https://a.com/x/", "body")
    # Retrieval via raw variant should hit because both normalize.
    assert c.get("HTTPS://A.COM/x") == "body"


def test_evict_stale_keeps_allowed_entries():
    c = UrlCache()
    c.put("https://a.com/x/", "body-a")
    c.put("https://b.com/y/", "body-b")
    allow = AllowList(["https://a.com/x"])
    dropped = c.evict_stale(allow)
    assert dropped == 1
    assert c.get("https://a.com/x/") == "body-a"
    assert c.get("https://b.com/y/") is None


def test_evict_stale_on_empty_cache():
    c = UrlCache()
    allow = AllowList(["https://a.com/x"])
    assert c.evict_stale(allow) == 0
