from app.config import MAX_NOTES


def build_notes(owner: str, note_count: int) -> list[str]:
    notes = [
        f"Prepared for {owner}",
        "Variance reviewed",
        "Snapshot generated from staging data",
        "Unused note",
    ]
    return notes[: min(note_count, MAX_NOTES)]
