"""
ts_dsp.units — time and voltage unit vocabulary.

The user may type "usec", "microseconds" or "µs"; the EDF header may declare
"uV" or "mV". These helpers turn any accepted spelling into a numeric factor
(to seconds / to volts) plus one canonical display symbol, so labels read the
same regardless of how the unit was spelled. Also the H:M:S clock helpers.

Used by the template (parsing the user's inputs), the readers (interpreting
the file header) and the plot stage (tick labels). Standard library only.
"""

from __future__ import annotations


############# UNIT TABLES ##################
# Time units -> seconds. Keys are lowercased and stripped of a trailing
# "s"-plural at lookup time, so "microseconds"/"microsecond" both hit.
_TIME_TO_S: dict[str, float] = {
    "us": 1e-6, "usec": 1e-6, "µs": 1e-6, "microsecond": 1e-6,
    "ms": 1e-3, "msec": 1e-3, "millisecond": 1e-3,
    "s": 1.0, "sec": 1.0, "second": 1.0,
    "min": 60.0, "minute": 60.0, "m": 60.0,
    "h": 3600.0, "hr": 3600.0, "hour": 3600.0,
}
# Canonical display symbol per resolved factor (so labels read cleanly
# regardless of which alias the user typed).
_TIME_SYMBOL: dict[float, str] = {
    1e-6: "µs", 1e-3: "ms", 1.0: "s", 60.0: "min", 3600.0: "h",
}

# Voltage units -> volts (SI).
_VOLT_TO_V: dict[str, float] = {
    "v": 1.0, "volt": 1.0,
    "mv": 1e-3, "millivolt": 1e-3,
    "uv": 1e-6, "µv": 1e-6, "microvolt": 1e-6,
    "nv": 1e-9, "nanovolt": 1e-9,
}
_VOLT_SYMBOL: dict[float, str] = {1.0: "V", 1e-3: "mV", 1e-6: "µV", 1e-9: "nV"}


def _normalise_unit_key(unit: str) -> str:
    """Lowercase, strip whitespace, and drop a trailing plural 's'."""
    key = str(unit).strip().lower()
    # "seconds" -> "second", "microvolts" -> "microvolt", but leave a bare
    # "s"/"ms"/"us"/"mv"/"uv" alone (they are already the short symbols).
    if len(key) > 2 and key.endswith("s"):
        key = key[:-1]
    return key


############# UNIT RESOLUTION ##################
def resolve_time_unit(unit: str | None, *, required: bool) -> tuple[float, str | None]:
    """(factor-to-seconds, display symbol) for a time unit.

    We do not default the unit. When `required` (a numeric time was given)
    and none is supplied, raise and ask the user for one. A supplied unit
    that is not a time unit always raises.
    """
    if unit is None or str(unit).strip() == "":
        if required:
            raise RuntimeError(
                "A time unit is required for numeric start/end/duration "
                "values — please specify one (us/usec, ms, s/sec, min, h), "
                "or give the times as H:M:S clock values instead."
            )
        return 1.0, None
    factor = _TIME_TO_S.get(_normalise_unit_key(unit))
    if factor is None:
        raise RuntimeError(
            f"Time unit {unit!r} is not a recognised unit of time. "
            "Use one of: us/usec, ms, s/sec, min, h."
        )
    return factor, _TIME_SYMBOL[factor]


def resolve_volt_unit(unit: str | None, *, required: bool = True) -> tuple[float, str]:
    """(factor-to-volts, display symbol) for a voltage unit.

    We do not default the unit. When `required` and none is supplied, raise
    and ask the user for one. A supplied unit that is not a voltage unit
    always raises. (`required=False` is used by readers for a unit declared
    inside the file, where a blank declaration may fall back to µV.)
    """
    if unit is None or str(unit).strip() == "":
        if required:
            raise RuntimeError(
                "A y-axis unit is required — please specify the voltage unit "
                "the range is in (V, mV, uV, or nV)."
            )
        return 1e-6, "µV"
    factor = _VOLT_TO_V.get(_normalise_unit_key(unit))
    if factor is None:
        raise RuntimeError(
            f"y-axis unit {unit!r} is not a recognised unit of voltage. "
            "Use one of: V, mV, uV, nV."
        )
    return factor, _VOLT_SYMBOL[factor]


############# CLOCK TIME ##################
def is_clock_string(value: "object") -> bool:
    """True for an 'H:M:S'-style string (as opposed to a plain number)."""
    return isinstance(value, str) and ":" in value


def clock_to_seconds(value: str) -> float:
    """'H:M:S' or 'H:M:S.ms' -> seconds since midnight."""
    parts = str(value).strip().split(":")
    if len(parts) != 3:
        raise RuntimeError(
            f"Clock time {value!r} is not in H:M:S format (e.g. '14:07:00')."
        )
    try:
        h, m, s = (float(p) for p in parts)
    except ValueError as exc:
        raise RuntimeError(f"Clock time {value!r} has non-numeric fields.") from exc
    return h * 3600.0 + m * 60.0 + s


def seconds_to_clock(seconds: float) -> str:
    """Seconds-since-midnight -> 'H:MM:SS' (or 'H:MM:SS.mmm' with a fraction)."""
    s = float(seconds) % (24 * 3600.0)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s - h * 3600 - m * 60
    if abs(sec - round(sec)) < 1e-6:
        return f"{h:d}:{m:02d}:{int(round(sec)):02d}"
    return f"{h:d}:{m:02d}:{sec:06.3f}"
