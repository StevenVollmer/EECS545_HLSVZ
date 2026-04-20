from app.main import export_profile


def test_export_profile_keeps_uppercase_account_code() -> None:
    assert export_profile("o'connor-smith", 4) == "account=O'CONNOR-SMITH,tickets=4"

