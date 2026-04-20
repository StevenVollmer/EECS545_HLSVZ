from app.models.account import Account


def over_quota_count(accounts: list[Account]) -> int:
    return len([account for account in accounts if account.used > account.limit])
