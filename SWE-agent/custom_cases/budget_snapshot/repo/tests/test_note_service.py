from app.services.note_service import build_notes


def test_note_count_is_capped() -> None:
    assert len(build_notes("Dana", 9)) == 3
