from app.models.account import Account
from app.services.quota_service import over_quota_count
from app.utils.labels import render_count_label


def render_snapshot(accounts: list[Account]) -> str:
    return render_count_label("over quota", over_quota_count(accounts), "account")
