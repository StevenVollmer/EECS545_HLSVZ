from title_case import to_title_case


def test_to_title_case_preserves_api() -> None:
    assert to_title_case("api_response") == "API Response"


def test_to_title_case_basic_words() -> None:
    assert to_title_case("release_notes") == "Release Notes"
