# Custom Cases 2

`custom_cases_2` is an extension set for lightweight software-engineering fixtures.

Each case follows the same layout as `custom_cases`:

```text
custom_cases_2/
  <case_name>/
    case.json
    repo/
      ...
```

Current target mix:

- easy: 7
- medium: 7
- hard: 6

Use the same runner/judger flow as existing cases, for example:

```bash
./env/bin/python SWE-agent/scripts/custom/judge_custom_case.py \
  --case SWE-agent/custom_cases_2/ratio_guard \
  --mode baseline \
  --run-install
```
