from app.utils.rates import format_transfer_rate


def transfer_digest_preview(owner: str, rate: float) -> str:
    titled = owner[:1].upper() + owner[1:]
    return f"Transfer for {titled} at {format_transfer_rate(rate)}"


def export_owner_code(owner: str) -> str:
    return f"owner={owner.upper()}"
