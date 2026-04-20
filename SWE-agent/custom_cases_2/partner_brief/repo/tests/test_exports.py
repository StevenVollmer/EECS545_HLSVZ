from app.main import export_partner_codes
from app.models.partner import Partner


def test_export_partner_codes_remains_uppercase() -> None:
    partners = [Partner("north"), Partner("west")]
    assert export_partner_codes(partners) == "NORTH,WEST"

