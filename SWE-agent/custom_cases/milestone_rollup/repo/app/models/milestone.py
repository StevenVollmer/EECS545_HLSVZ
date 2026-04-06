from dataclasses import dataclass


@dataclass
class Milestone:
    name: str
    blocked: bool = False
    closed: bool = False
