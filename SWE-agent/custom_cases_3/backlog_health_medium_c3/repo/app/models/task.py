from dataclasses import dataclass


@dataclass
class Task:
    title: str
    blocked: bool = True
    archived: bool = False
