from risk import attention_ratio


def test_attention_ratio_uses_actionable_alerts_as_denominator() -> None:
    alerts = [
        {"state": "open", "muted": False, "severity": "critical"},
        {"state": "open", "muted": False, "severity": "warning"},
        {"state": "open", "muted": True, "severity": "critical"},
        {"state": "closed", "muted": False, "severity": "critical"},
    ]
    assert attention_ratio(alerts) == 0.5


def test_attention_ratio_returns_zero_when_no_actionable_alerts() -> None:
    alerts = [
        {"state": "closed", "muted": False, "severity": "critical"},
        {"state": "open", "muted": True, "severity": "warning"},
    ]
    assert attention_ratio(alerts) == 0.0
