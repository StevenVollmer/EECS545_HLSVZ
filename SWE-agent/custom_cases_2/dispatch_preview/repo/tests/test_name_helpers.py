from app.utils.names import display_contact_name


def test_display_contact_name_simple_case() -> None:
    assert display_contact_name("ava") == "Ava"


def test_display_contact_name_hyphen_without_apostrophe() -> None:
    assert display_contact_name("mina-west") == "Mina-west"

