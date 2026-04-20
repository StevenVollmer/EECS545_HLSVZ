from app.models.ticket import Ticket
from app.services.ticket_service import escalated_open_count
from app.utils.labels import render_count_label


def render_digest(tickets: list[Ticket]) -> str:
    return render_count_label("open escalations", escalated_open_count(tickets), "ticket")
