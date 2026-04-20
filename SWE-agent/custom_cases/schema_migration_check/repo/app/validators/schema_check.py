"""Post-migration schema validator.

Confirms that every column declared in a migration exists in the database
after the migration is applied. Used by CI to catch migrations that
silently drop or rename columns.

Input:
  - db: Database (post-migration state)
  - migrations: list of migration dicts as passed to apply_migration
Output:
  - list of (table, column) pairs that are declared in a migration but not
    present in the database.

The validator compares the declared (raw) column names against the database
state. This matches how operators write migration files — they want to see
exactly what they declared reflected in the schema.
"""

from __future__ import annotations

from app.schema.model import Database


def missing_columns(db: Database, migrations: list[dict]) -> list[tuple[str, str]]:
    missing: list[tuple[str, str]] = []
    for migration in migrations:
        table = migration["table"]
        for column in migration["columns"]:
            # BUG: does not apply the same case-folding the migration step
            # applies at apply time. If the declared column has a capital
            # letter, it is stored lowercased in the db but compared raw here,
            # so the validator falsely reports it as missing. Fix requires
            # recognizing the invariant enforced in app/migrations/apply.py
            # and matching it here (or removing the fold in both places).
            if not db.has_column(table, column):
                missing.append((table, column))
    return missing
