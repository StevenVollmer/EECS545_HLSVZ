"""Demo: preview backoff delays for a 10-attempt retry schedule with a 30s cap.

The cap MUST bound the scheduled wait. We print the maximum scheduled delay so
operators can confirm retries never wait longer than the configured cap.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.retry_runner import plan_schedule


def main() -> int:
    rng = random.Random(42)
    schedule = plan_schedule(
        attempts=10,
        base_seconds=1.0,
        factor=2.0,
        max_delay_seconds=30.0,
        jitter_ratio=0.25,
        rng=rng,
    )
    print("Retry schedule preview")
    print("======================")
    print(f"Max scheduled delay: {schedule.max_delay:.2f}s")
    print(f"Cap configured:      30.00s")
    if schedule.max_delay > 30.0:
        print("WARNING: scheduled delay exceeds configured cap")
    else:
        print("OK: all scheduled delays are within the cap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
