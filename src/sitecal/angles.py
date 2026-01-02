from __future__ import annotations

import re


_DMS_RE = re.compile(
    r"""^\s*
    (?P<hem>[NSEW])\s*
    (?P<deg>\d{1,3})\s*°\s*
    (?P<min>\d{1,2})\s*'\s*
    (?P<sec>\d+(?:\.\d+)?)\s*"\s*
    $""",
    re.VERBOSE,
)


def dms_to_decimal(dms: str) -> float:
    """
    Convert strings like:
      S24°17'00.52919"  -> -24.28348033...
      W69°05'18.44412"  -> -69.08845670...
    """
    m = _DMS_RE.match(dms)
    if not m:
        raise ValueError(f"Bad DMS format: {dms!r}")

    hem = m.group("hem").upper()
    deg = float(m.group("deg"))
    minute = float(m.group("min"))
    sec = float(m.group("sec"))

    dec = deg + minute / 60.0 + sec / 3600.0
    if hem in ("S", "W"):
        dec = -dec
    return dec
