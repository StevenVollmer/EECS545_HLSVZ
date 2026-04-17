def center_value(values):
    if not values:
        raise ValueError("values must not be empty")
    return values[len(values) // 2 - 1]

