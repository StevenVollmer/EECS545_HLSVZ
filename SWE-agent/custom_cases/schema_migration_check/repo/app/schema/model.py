"""Simulated database schema model.

The 'database' is a plain dict mapping table_name -> set of column names.
Tables and columns are stored exactly as the apply-migration step writes them.
"""

from __future__ import annotations


class Database:
    def __init__(self) -> None:
        self.tables: dict[str, set[str]] = {}

    def create_table(self, name: str, columns: list[str]) -> None:
        self.tables[name] = set(columns)

    def has_column(self, table: str, column: str) -> bool:
        return table in self.tables and column in self.tables[table]
