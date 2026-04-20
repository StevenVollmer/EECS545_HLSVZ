from initials import initials


def test_initials_are_uppercase() -> None:
    assert initials("Grace Hopper") == "GH"


def test_initials_single_name() -> None:
    assert initials("Ada") == "a"
