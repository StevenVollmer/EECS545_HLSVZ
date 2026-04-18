from eta import eta_label


def test_eta_label_singular_day() -> None:
    assert eta_label(1) == "1 day"


def test_eta_label_plural_days() -> None:
    assert eta_label(3) == "3 days"
