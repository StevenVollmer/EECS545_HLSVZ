from app.migrations.apply import apply_migration
from app.schema.model import Database
from app.validators.schema_check import missing_columns


def test_all_lowercase_migration_no_missing():
    db = Database()
    mig = {"table": "orders", "columns": ["id", "total"]}
    apply_migration(db, mig)
    assert missing_columns(db, [mig]) == []


def test_reports_truly_missing_column():
    db = Database()
    # Apply a migration that only creates "id"; validator is asked about a
    # migration that declared "id" AND "total" — "total" should be reported.
    apply_migration(db, {"table": "orders", "columns": ["id"]})
    declared = {"table": "orders", "columns": ["id", "total"]}
    assert missing_columns(db, [declared]) == [("orders", "total")]
