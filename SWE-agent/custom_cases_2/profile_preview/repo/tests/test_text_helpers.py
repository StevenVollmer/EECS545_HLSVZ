from app.utils.text import format_display_name


def test_format_display_name_basic_hyphen_case() -> None:
    assert format_display_name("alice-jones") == "Alice-Jones"


def test_format_display_name_trims_segments() -> None:
    assert format_display_name("  ava -  west ") == "Ava-West"

