# schema_migration_check fixture

Three cooperating modules:

- `app/schema/model.py` — in-memory database
- `app/migrations/apply.py` — applies migration dicts, case-folds identifiers
- `app/validators/schema_check.py` — post-apply validator
