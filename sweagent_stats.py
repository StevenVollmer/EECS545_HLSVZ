#!/usr/bin/env python3
import re
import sys
from pathlib import Path

PATTERN = re.compile(
    r"total_tokens_sent=([\d,]+),\s*"
    r"total_tokens_received=([\d,]+),\s*"
    r"total_cost=([0-9.]+),\s*"
    r"total_api_calls=(\d+)"
)


def parse_int(value: str) -> int:
    return int(value.replace(",", ""))


def parse_log(log_path: Path) -> tuple[int, int, float, int]:
    matches = []

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                tokens_sent = parse_int(m.group(1))
                tokens_received = parse_int(m.group(2))
                cost = float(m.group(3))
                calls = int(m.group(4))
                matches.append((tokens_sent, tokens_received, cost, calls, line.strip()))

    if not matches:
        raise ValueError(f"Could not find token stats in {log_path}")

    # Usually the final summary is the best one to use.
    tokens_sent, tokens_received, cost, calls, matched_line = matches[-1]
    return tokens_sent, tokens_received, cost, calls


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 sweagent_stats.py <debug.log>")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"Error: file not found: {log_path}")
        sys.exit(1)

    try:
        tokens_sent, tokens_received, cost, calls = parse_log(log_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    total_tokens = tokens_sent + tokens_received
    tokens_per_call = total_tokens / calls if calls else 0.0

    print("SWE-agent run stats")
    print("-------------------")
    print(f"log file: {log_path}")
    print(f"api calls: {calls}")
    print(f"prompt tokens: {tokens_sent}")
    print(f"completion tokens: {tokens_received}")
    print(f"total tokens: {total_tokens}")
    print(f"tokens per call: {tokens_per_call:.2f}")
    print(f"cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
