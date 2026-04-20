from app.utils.subject import format_service_name


def alert_subject_preview(service: str, severity: str) -> str:
    return f"{severity.upper()}: {format_service_name(service)} alert"


def export_severity(severity: str) -> str:
    return f"severity={severity.upper()}"
