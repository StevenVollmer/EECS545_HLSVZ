from dataclasses import dataclass, field


@dataclass
class DigestRequest:
    owner_name: str
    total_value: float
    alert_count: int


@dataclass
class DigestMetrics:
    total_value: float
    alert_count: int
    movement_label: str


@dataclass
class DigestViewModel:
    greeting: str
    owner_name: str
    summary_line: str
    metrics: DigestMetrics
    notes: list[str] = field(default_factory=list)
