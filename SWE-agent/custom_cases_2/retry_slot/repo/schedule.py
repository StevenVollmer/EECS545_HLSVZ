def next_retry_minutes(attempt: int) -> int:
    if attempt <= 0:
        return 0
    return min(attempt * 4, 20)
