from owners import visible_owner_names


def test_visible_owner_names_sorts_case_insensitively() -> None:
    records = [
        {"owner": "zoe", "archived": False},
        {"owner": "ava", "archived": False},
        {"owner": "Noah", "archived": False},
    ]
    assert visible_owner_names(records) == ["ava", "Noah", "zoe"]


def test_visible_owner_names_ignores_archived_records() -> None:
    records = [
        {"owner": "Mina", "archived": False},
        {"owner": "Ava", "archived": True},
    ]
    assert visible_owner_names(records) == ["Mina"]
