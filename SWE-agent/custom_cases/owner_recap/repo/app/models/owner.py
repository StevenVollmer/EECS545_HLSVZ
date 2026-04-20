from dataclasses import dataclass


@dataclass(frozen=True)
class OwnerRecap:
    owner_name: str
    total_projects: int
