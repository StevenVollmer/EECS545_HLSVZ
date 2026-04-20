from dataclasses import dataclass


@dataclass
class SnapshotItem:
    title: str
    severity: str
    archived: bool = False
