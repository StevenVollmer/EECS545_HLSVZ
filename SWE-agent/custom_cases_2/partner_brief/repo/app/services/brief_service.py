from app.models.partner import Partner


def active_partner_count(partners: list[Partner]) -> int:
    return len([partner for partner in partners if partner.name])

