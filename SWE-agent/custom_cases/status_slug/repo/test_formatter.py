from formatter import slugify_status


def test_slugify_status_hyphenates_spaces() -> None:
    assert slugify_status("Needs Review") == "needs-review"


def test_slugify_status_normalizes_case() -> None:
    assert slugify_status("BLOCKED") == "blocked"
