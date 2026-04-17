from headline import format_headline


def test_format_headline_keeps_api_uppercase() -> None:
    assert format_headline("api_status") == "API Status"


def test_format_headline_basic_slug() -> None:
    assert format_headline("release_notes") == "Release Notes"
