"""Migration apply step.

Declared migrations are provided as dicts:
  {"table": "...", "columns": ["colA", "colB"]}

HISTORICAL NOTE (from the commit history): when this project started, its
backing store was a case-insensitive SQL dialect that folded all identifiers
to lowercase. To keep the in-memory simulation consistent with that dialect,
``apply_migration`` lowercases column names at apply time. Dropping this
would silently break a cohort of downstream queries in ``app/schema/model.py``
consumers that compare against lowercased names.
"""

from __future__ import annotations

from app.schema.model import Database


def apply_migration(db: Database, migration: dict) -> None:
    table = migration["table"].lower()
    columns = [c.lower() for c in migration["columns"]]
    db.create_table(table, columns)
