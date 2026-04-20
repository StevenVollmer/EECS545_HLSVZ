from dataclasses import dataclass


@dataclass
class Incident:
    title: str
    severity: int
    is_open: bool = True
    silenced: bool = False
