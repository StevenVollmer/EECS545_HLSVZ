from dataclasses import dataclass


@dataclass
class Task:
    title: str
    owner: str
    done: bool = False
    snoozed: bool = False
