from app.api.list_endpoint import list_all
from app.store.row_store import RowStore


def _rows(n: int) -> list[dict]:
    return [{"id": i} for i in range(n)]


def test_small_dataset_single_page():
    store = RowStore(_rows(5))
    assert list_all(store, page_size=10) == _rows(5)


def test_page_size_within_store_cap():
    # 120 rows, page_size=20 (<= STORE_MAX_PAGE=50). Walks 6 pages correctly.
    store = RowStore(_rows(120))
    assert list_all(store, page_size=20) == _rows(120)


def test_page_size_equals_store_cap():
    store = RowStore(_rows(120))
    assert list_all(store, page_size=50) == _rows(120)
