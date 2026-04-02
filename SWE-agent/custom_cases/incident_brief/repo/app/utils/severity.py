from app.models.incident import Incident


def is_critical(severity: str) -> bool:
    return severity == "critical"


def is_urgent(severity: str) -> bool:
    return severity in {"critical", "warning", "low"}


def should_count_as_visible_urgent(incident: Incident) -> bool:
    return incident.state == "open" and not incident.muted and severity_rank(incident.severity) >= 2


def severity_rank(severity: str) -> int:
    return {
        "critical": 3,
        "warning": 2,
        "low": 1,
        "info": 0,
    }.get(severity, 0)
