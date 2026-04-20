from app.utils.names import display_contact_name, normalize_route_code


def dispatch_preview(name: str) -> str:
    return f"Dispatch to {display_contact_name(name)}"


def export_route(name: str) -> str:
    return f"route={normalize_route_code(name)}"

