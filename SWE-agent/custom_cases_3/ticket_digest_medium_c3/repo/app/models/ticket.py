from dataclasses import dataclass


@dataclass
class Ticket:
    title: str
    escalated: bool = True
    snoozed: bool = False
