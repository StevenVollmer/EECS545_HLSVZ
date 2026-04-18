from app.models.ticket import Ticket
from app.presenters.digest_presenter import render_digest
from app.services.ticket_service import escalated_open_count


def test_escalated_open_count_excludes_snoozed_tickets() -> None:
    tickets = [
        Ticket("API", escalated=True, snoozed=False),
        Ticket("Billing", escalated=True, snoozed=True),
    ]
    assert escalated_open_count(tickets) == 1


def test_presenter_uses_service_output() -> None:
    assert render_digest([Ticket("API", escalated=True, snoozed=False)]) == "open escalations: 1 ticket"


def test_non_matching_rows_do_not_count() -> None:
    assert escalated_open_count([Ticket("Docs", escalated=False, snoozed=False)]) == 0
