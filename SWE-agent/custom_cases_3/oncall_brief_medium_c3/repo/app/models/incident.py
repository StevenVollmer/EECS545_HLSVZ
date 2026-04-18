from dataclasses import dataclass


@dataclass
class Incident:
    key: str
    active: bool = True
    acknowledged: bool = False
