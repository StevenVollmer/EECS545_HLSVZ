from app.services.search_service import search


def test_search_widgets_finds_intro():
    hits = search("widgets")
    ids = {h.doc_id for h in hits}
    assert "doc-1" in ids


def test_search_gadgets_matches_overview_and_assembly():
    hits = search("gadgets")
    ids = {h.doc_id for h in hits}
    assert ids == {"doc-2", "doc-3"}


def test_search_case_insensitive():
    hits = search("WIDGETS")
    assert any(h.doc_id == "doc-1" for h in hits)
