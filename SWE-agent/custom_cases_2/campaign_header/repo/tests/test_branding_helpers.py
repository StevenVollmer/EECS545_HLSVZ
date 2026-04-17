from app.utils.branding import format_brand_name


def test_format_brand_name_basic_words() -> None:
    assert format_brand_name("sunrise labs") == "Sunrise Labs"


def test_format_brand_name_trims_spacing() -> None:
    assert format_brand_name("  north   star  ") == "North Star"
