from app.net.url import normalize_url


def test_lowercases_scheme_and_host():
    assert normalize_url("HTTPS://Example.COM/path") == "https://example.com/path"


def test_strips_trailing_slash_but_keeps_root():
    assert normalize_url("https://a.com/x/") == "https://a.com/x"
    assert normalize_url("https://a.com/") == "https://a.com/"


def test_sorts_query():
    assert normalize_url("https://a.com/?b=2&a=1") == "https://a.com/?a=1&b=2"
