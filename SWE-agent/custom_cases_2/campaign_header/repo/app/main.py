from app.utils.branding import format_brand_name, normalize_campaign_code


def render_campaign_header(name: str, channel: str) -> str:
    return f"Campaign for {format_brand_name(name)} [{channel}]"


def export_campaign(name: str, batch: int) -> str:
    return f"code={normalize_campaign_code(name)},batch={batch}"
