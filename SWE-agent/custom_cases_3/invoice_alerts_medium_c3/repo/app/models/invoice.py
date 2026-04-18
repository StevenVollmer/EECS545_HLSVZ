from dataclasses import dataclass


@dataclass
class Invoice:
    number: str
    overdue: bool = True
    waived: bool = False
