from app.models.partner import Partner
from app.services.brief_service import active_partner_count


def test_active_partner_count_empty_list() -> None:
    assert active_partner_count([]) == 0


def test_active_partner_count_simple_active_case() -> None:
    assert active_partner_count([Partner("north", paused=False)]) == 1

