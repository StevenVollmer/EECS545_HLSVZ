from app.main import export_owner


def test_export_owner_keeps_uppercase_owner_code() -> None:
    assert export_owner("mcallister-smith", 4) == "owner_code=MCALLISTER-SMITH,projects=4"


def test_export_owner_handles_spaces_around_segments() -> None:
    assert export_owner(" acme - west ", 2) == "owner_code=ACME-WEST,projects=2"
