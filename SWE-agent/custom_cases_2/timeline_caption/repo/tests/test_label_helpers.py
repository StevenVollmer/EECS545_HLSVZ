from app.utils.labels import format_timeline_label


def test_format_timeline_label_basic_case() -> None:
    assert format_timeline_label("operations/platform-team") == "Operations/Platform-Team"


def test_format_timeline_label_trims_sections() -> None:
    assert format_timeline_label("  operations / platform-team  ") == "Operations/Platform-Team"
