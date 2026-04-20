from app.models.ticket import Ticket


def escalated_open_count(tickets: list[Ticket]) -> int:
    return len([ticket for ticket in tickets if ticket.escalated])
