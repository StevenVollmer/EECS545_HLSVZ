from dataclasses import dataclass


@dataclass(frozen=True)
class AlertItem:
    title: str
    state: str
    snoozed: bool
    severity: str


@dataclass(frozen=True)
class Workspace:
    name: str
    alerts: list[AlertItem]


@dataclass(frozen=True)
class WorkspaceDigest:
    name: str
    visible_alert_count: int
    open_alert_count: int
    critical_alert_count: int
