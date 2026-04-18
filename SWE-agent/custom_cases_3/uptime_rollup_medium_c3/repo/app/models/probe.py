from dataclasses import dataclass


@dataclass
class Probe:
    name: str
    failing: bool = True
    suppressed: bool = False
