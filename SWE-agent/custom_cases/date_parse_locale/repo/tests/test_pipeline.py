from app.pipeline.filter_recent import filter_recent
from app.pipeline.group_by_month import group_by_month
from app.pipeline.normalize import normalize_records


def test_end_to_end_iso_only():
    records = [{"date": "2024-01-15", "v": 1}, {"date": "2022-12-01", "v": 2}]
    norm = normalize_records(records)
    recent = filter_recent(norm, 2023)
    grouped = group_by_month(recent)
    assert list(grouped.keys()) == [(2024, 1)]
    assert grouped[(2024, 1)][0]["v"] == 1
