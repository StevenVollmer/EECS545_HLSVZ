from app.migrations.apply import apply_migration
from app.schema.model import Database


def test_apply_creates_table():
    db = Database()
    apply_migration(db, {"table": "users", "columns": ["id", "email"]})
    assert db.has_column("users", "id")
    assert db.has_column("users", "email")


def test_apply_is_idempotent_on_overwrite():
    db = Database()
    apply_migration(db, {"table": "t", "columns": ["a"]})
    apply_migration(db, {"table": "t", "columns": ["a", "b"]})
    assert db.has_column("t", "a")
    assert db.has_column("t", "b")
