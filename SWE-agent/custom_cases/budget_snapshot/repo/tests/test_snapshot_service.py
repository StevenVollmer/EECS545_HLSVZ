from app.main import render_budget_snapshot


def test_snapshot_contains_heading() -> None:
    preview = render_budget_snapshot("Dana", 12_400, -3.2, 2)
    assert "Weekly Budget Snapshot" in preview


def test_snapshot_contains_variance() -> None:
    preview = render_budget_snapshot("Dana", 12_400, -3.2, 2)
    assert "3.2% under plan" in preview
