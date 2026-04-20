"""Demo: a migration declares a column with a capital letter.

The migration apply step lowercases it (per the codebase's long-standing
case-insensitive invariant). The post-migration validator then compares the
*declared* name (with capital) against the *stored* name (lowercased) and
falsely reports it as missing.

A correct validator must align with the apply step's invariant.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.migrations.apply import apply_migration
from app.schema.model import Database
from app.validators.schema_check import missing_columns


def main() -> int:
    mig = {"table": "Users", "columns": ["UserId", "Email"]}
    db = Database()
    apply_migration(db, mig)
    report = missing_columns(db, [mig])
    print(f"missing_columns report: {report}")
    if report == []:
        print("OK: validator aligns with apply-step invariant")
    else:
        print("WARNING: validator reports declared columns as missing after apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
