from app.main import export_segment_slug, segment_badge_preview
from app.utils.badge import bucket_count


def test_export_segment_slug_uppercase() -> None:
    assert export_segment_slug('west-enterprise') == 'segment=WEST-ENTERPRISE'


def test_bucket_count_small_values_unchanged() -> None:
    assert bucket_count(42) == '42'


def test_segment_badge_high_values_collapsed() -> None:
    assert segment_badge_preview('northwest', 101) == 'Northwest [99+]'
