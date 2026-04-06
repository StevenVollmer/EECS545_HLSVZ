def visible_owner_names(records: list[dict[str, object]]) -> list[str]:
    names = [str(record["owner"]) for record in records if not bool(record.get("archived", False))]
    return sorted(names)
