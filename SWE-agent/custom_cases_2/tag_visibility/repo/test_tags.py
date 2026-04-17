from tags import visible_tags


def test_visible_tags_excludes_hidden_entries() -> None:
    tags = [
        {"name": "ops", "hidden": False},
        {"name": "urgent", "hidden": True},
    ]
    assert visible_tags(tags) == ["ops"]


def test_visible_tags_are_sorted() -> None:
    tags = [
        {"name": "beta", "hidden": False},
        {"name": "alpha", "hidden": False},
    ]
    assert visible_tags(tags) == ["alpha", "beta"]

