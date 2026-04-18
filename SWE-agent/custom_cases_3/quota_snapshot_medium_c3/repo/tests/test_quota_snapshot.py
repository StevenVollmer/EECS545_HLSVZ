from app.models.account import Account
from app.presenters.snapshot_presenter import render_snapshot
from app.services.quota_service import over_quota_count


def test_over_quota_count_excludes_suspended_accounts() -> None:
    accounts = [
        Account("Aster", used=14, limit=10, suspended=False),
        Account("Boreal", used=20, limit=10, suspended=True),
    ]
    assert over_quota_count(accounts) == 1


def test_presenter_uses_service_output() -> None:
    assert render_snapshot([Account("Aster", used=14, limit=10, suspended=False)]) == "over quota: 1 account"


def test_non_matching_rows_do_not_count() -> None:
    assert over_quota_count([Account("Cedar", used=5, limit=10, suspended=False)]) == 0
