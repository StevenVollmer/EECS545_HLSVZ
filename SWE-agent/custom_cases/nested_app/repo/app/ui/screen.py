from app.services.profile_service import build_user
from app.ui.components import greeting_prefix
from app.ui.presenter import present_name


def render_screen(name: str) -> str:
    user = build_user(name)
    return f"{greeting_prefix()}, {present_name(user.display_name)}!"
