from dataclasses import dataclass


@dataclass
class BoardCard:
    title: str
    owner: str
    severity: str
    archived: bool = False
