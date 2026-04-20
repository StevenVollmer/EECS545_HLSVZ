from app.models.partner import Partner
from app.services.brief_service import active_partner_count


def preview_brief(partners: list[Partner]) -> str:
    return f"Active partners: {active_partner_count(partners)}"


def export_partner_codes(partners: list[Partner]) -> str:
    codes = [partner.name.strip().upper() for partner in partners if partner.name.strip()]
    return ",".join(codes)

