from app.core.ranking import rank


def test_basic_descending_by_score():
    entries = [
        {"team_id": "a", "score": 10, "submission_number": "1"},
        {"team_id": "b", "score": 20, "submission_number": "1"},
    ]
    out = rank(entries)
    assert out[0]["team_id"] == "b"
    assert out[1]["team_id"] == "a"


def test_tie_by_team_id_when_submission_equal():
    # Same score, same submission number — sort by team_id
    entries = [
        {"team_id": "b", "score": 10, "submission_number": "1"},
        {"team_id": "a", "score": 10, "submission_number": "1"},
    ]
    out = rank(entries)
    assert out[0]["team_id"] == "a"


def test_single_digit_submissions_order():
    # Only single-digit submission numbers: lexicographic == numeric.
    entries = [
        {"team_id": "a", "score": 5, "submission_number": "3"},
        {"team_id": "b", "score": 5, "submission_number": "1"},
        {"team_id": "c", "score": 5, "submission_number": "2"},
    ]
    out = rank(entries)
    assert [e["team_id"] for e in out] == ["b", "c", "a"]
