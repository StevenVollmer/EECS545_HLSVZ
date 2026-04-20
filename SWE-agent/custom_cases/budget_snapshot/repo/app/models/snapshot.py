from dataclasses import dataclass, field


@dataclass
class SnapshotRequest:
    owner: str
    total_spend: float
    variance: float
    note_count: int


@dataclass
class SnapshotMetrics:
    total_spend_label: str
    variance_label: str
    note_count: int


@dataclass
class SnapshotViewModel:
    heading: str
    owner: str
    summary_line: str
    metrics: SnapshotMetrics
    notes: list[str] = field(default_factory=list)
