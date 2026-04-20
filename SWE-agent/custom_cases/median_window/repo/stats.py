def median_window(values: list[int]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint + 1]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2
