from app.models.hit import SearchHit
from app.services.highlight_service import add_highlights


def test_wraps_match_in_mark_tags():
    hit = SearchHit("d1", "T", 1.0, "abc widget xyz")
    out = add_highlights([hit], "widget")
    assert "<mark>widget</mark>" in out[0].snippet


def test_no_match_leaves_snippet_unchanged():
    hit = SearchHit("d1", "T", 1.0, "nothing interesting here")
    out = add_highlights([hit], "widget")
    assert out[0].snippet == "nothing interesting here"
