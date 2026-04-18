from dataclasses import dataclass


@dataclass
class Account:
    name: str
    used: int
    limit: int
    suspended: bool = False
