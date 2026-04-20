from app.main import preview_owner_recap


def test_preview_owner_recap_keeps_project_count() -> None:
    rendered = preview_owner_recap("acme-west", 3)
    assert rendered.endswith("Projects: 3")


def test_preview_owner_recap_renders_owner_prefix() -> None:
    rendered = preview_owner_recap("acme-west", 3)
    assert rendered.startswith("Owner:")
