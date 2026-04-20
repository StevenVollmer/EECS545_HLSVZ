from dataclasses import dataclass


@dataclass
class BoardSummary:
    active_cards: int
    alert_cards: int
    owners: list[str]
