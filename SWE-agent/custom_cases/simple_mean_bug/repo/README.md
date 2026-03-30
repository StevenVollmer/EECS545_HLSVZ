# Simple Mean Bug Repo

This tiny repository exists to smoke-test the custom SWE-bench runner.

The bug is in `calculator.mean()`: it divides by `len(values) - 1` instead of `len(values)`.
