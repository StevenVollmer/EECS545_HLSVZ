from labels import clean_label


def test_clean_label_normalizes_internal_spaces() -> None:
    assert clean_label("  High   Priority  ") == "High Priority"


def test_clean_label_handles_single_word() -> None:
    assert clean_label("Backlog") == "Backlog"

