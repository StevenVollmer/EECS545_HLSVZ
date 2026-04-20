from app.utils.text import format_display_owner_name, normalize_owner_code


def test_normalize_owner_code_uppercases_segments() -> None:
    assert normalize_owner_code("mcallister-smith") == "MCALLISTER-SMITH"


def test_format_display_owner_name_handles_mc_prefix() -> None:
    assert format_display_owner_name("mcallister-smith") == "McAllister-Smith"
