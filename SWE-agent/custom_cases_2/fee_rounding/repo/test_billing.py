from billing import service_fee


def test_service_fee_rounds_half_up() -> None:
    assert service_fee(50) == 2


def test_service_fee_for_full_dollar() -> None:
    assert service_fee(100) == 3
