"""Date string parser supporting multiple operator-facing formats.

Accepted formats:
  - ISO: YYYY-MM-DD
  - European: DD/MM/YYYY
  - US:      MM-DD-YYYY

Returns a (year, month, day) tuple or raises ValueError.
"""

from __future__ import annotations

import re


_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_EU = re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")
_US = re.compile(r"^(\d{2})-(\d{2})-(\d{4})$")


def parse_date(text: str) -> tuple[int, int, int]:
    text = text.strip()
    m = _ISO.match(text)
    if m:
        y, mo, d = m.groups()
        return int(y), int(mo), int(d)
    m = _EU.match(text)
    if m:
        d, mo, y = m.groups()
        return int(y), int(mo), int(d)
    m = _US.match(text)
    if m:
        # BUG: swapped month/day when unpacking. Bug is here, but the module
        # name and public API are the obvious fix-sites so the bug is easy
        # once the coder reads this function. The hard part is localizing
        # here rather than in the many downstream modules in app/pipeline/
        # that also manipulate dates.
        d, mo, y = m.groups()
        return int(y), int(mo), int(d)
    raise ValueError(f"unrecognized date format: {text!r}")
