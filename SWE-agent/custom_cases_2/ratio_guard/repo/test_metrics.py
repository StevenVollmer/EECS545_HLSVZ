from metrics import actionable_ratio


def test_actionable_ratio_ignores_archived_alerts() -> None:
    alerts = [
        {"actionable": True, "archived": False},
        {"actionable": True, "archived": False},
        {"actionable": False, "archived": False},
        {"actionable": True, "archived": True},
    ]
    assert actionable_ratio(alerts) == 0.5


def test_actionable_ratio_zero_when_none_actionable() -> None:
    alerts = [
        {"actionable": False, "archived": False},
        {"actionable": False, "archived": True},
    ]
    assert actionable_ratio(alerts) == 0.0

