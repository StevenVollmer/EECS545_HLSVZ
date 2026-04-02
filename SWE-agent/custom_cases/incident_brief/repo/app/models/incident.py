from dataclasses import dataclass


@dataclass(frozen=True)
class Incident:
    title: str
    severity: str
    state: str
    muted: bool = False


@dataclass(frozen=True)
class IncidentBrief:
    team_name: str
    urgent_count: int
    open_count: int
    critical_count: int
