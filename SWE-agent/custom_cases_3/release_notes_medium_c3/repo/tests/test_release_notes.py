from app.models.change import Change
from app.presenters.notes_presenter import render_notes
from app.services.release_service import public_change_titles


def test_public_change_titles_excludes_deprecated_changes() -> None:
    changes = [
        Change("Bulk edit", public=True, deprecated=False),
        Change("Legacy export", public=True, deprecated=True),
    ]
    assert public_change_titles(changes) == ["Bulk edit"]


def test_presenter_uses_service_output() -> None:
    assert render_notes([Change("Bulk edit", public=True, deprecated=False)]) == "public highlights: Bulk edit"


def test_non_matching_rows_do_not_count() -> None:
    assert public_change_titles([Change("Private cache", public=False, deprecated=False)]) == []
