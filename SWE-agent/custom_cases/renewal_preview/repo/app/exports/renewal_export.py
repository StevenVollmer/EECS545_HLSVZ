def render_renewal_export(account_name: str, route_code: str) -> str:
    return f"account={account_name};route_code={route_code.upper()}"
