"""
ts_dsp.units — time and voltage unit vocabulary, and the time-point rules.

The user may type "usec", "microseconds" or "µs"; the EDF header may declare
"uV" or "mV". These helpers turn any accepted spelling into a numeric factor
(to seconds / to volts) plus one canonical display symbol, so labels read the
same regardless of how the unit was spelled.

A time input (start_time, end_time) can be written three ways; the rules for
reading them and anchoring them to a recording live here too, so every
reader resolves them the same way (see TIME POINTS below).

Used by the template (parsing the user's inputs), the readers (interpreting
the file header) and the plot stage (tick labels). Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from processor.errors import (
    PlotDataUnavailableError,
    PlotErrorCode,
    PlotErrorStage,
    PlotInvalidInputError,
)


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
            raise PlotInvalidInputError(
                PlotErrorCode.MISSING_ARGUMENT,
                "A time unit is required for numeric start/end/duration "
                "values — please specify one (us/usec, ms, s/sec, min, h), "
                "or give the times as H:M:S clock values instead."
            )
        return 1.0, None
    factor = _TIME_TO_S.get(_normalise_unit_key(unit))
    if factor is None:
        raise PlotInvalidInputError(
            PlotErrorCode.UNSUPPORTED_UNIT,
            f"Time unit {unit!r} is not a recognised unit of time. "
            "Use one of: us/usec, ms, s/sec, min, h.",
            error_facts={"unit": str(unit), "supported": ["us", "ms", "s", "min", "h"]},
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
            raise PlotInvalidInputError(
                PlotErrorCode.MISSING_ARGUMENT,
                "A y-axis unit is required — please specify the voltage unit "
                "the range is in (V, mV, uV, or nV)."
            )
        return 1e-6, "µV"
    factor = _VOLT_TO_V.get(_normalise_unit_key(unit))
    if factor is None:
        raise PlotInvalidInputError(
            PlotErrorCode.UNSUPPORTED_UNIT,
            f"y-axis unit {unit!r} is not a recognised unit of voltage. "
            "Use one of: V, mV, uV, nV.",
            error_facts={"unit": str(unit), "supported": ["V", "mV", "uV", "nV"]},
        )
    return factor, _VOLT_SYMBOL[factor]


############# CLOCK TIME ##################
def is_clock_string(value: "object") -> bool:
    """True for an 'H:M:S' or 'YYYY-MM-DD H:M:S' string (as opposed to a
    plain number). Such a value carries its own unit, so no time_unit is
    needed for it."""
    return isinstance(value, str) and ":" in value


def clock_to_seconds(value: str) -> float:
    """'H:M:S' or 'H:M:S.ms' -> seconds since midnight."""
    parts = str(value).strip().split(":")
    if len(parts) != 3:
        raise PlotInvalidInputError(
            PlotErrorCode.INVALID_ARGUMENT,
            f"Clock time {value!r} is not in H:M:S format (e.g. '14:07:00')."
        )
    try:
        h, m, s = (float(p) for p in parts)
    except ValueError as exc:
        raise PlotInvalidInputError(
            PlotErrorCode.INVALID_ARGUMENT,
            f"Clock time {value!r} has non-numeric fields.") from exc
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


def seconds_since_midnight(moment: datetime) -> float:
    """A datetime -> its time of day in seconds (what the H:M:S helpers use)."""
    return moment.hour * 3600.0 + moment.minute * 60.0 + moment.second + moment.microsecond / 1e6


def clock_name(recording_start: "datetime | None") -> str:
    """Which clock a recording's timestamps are on, for an axis label.

    A naive start (EDF) is the recorder's own clock, so "local"; a
    timezone-aware start (NWB stores UTC) names the zone.
    """
    if recording_start is None or recording_start.tzinfo is None:
        return "local"
    return recording_start.tzname() or "UTC"


############# TIME POINTS ##################
# A time input can be given three ways:
#
#   a number                   seconds from the recording start, once scaled
#                              by time_unit (500 with time_unit="ms" -> 0.5)
#   "HH:MM:SS"                 a time of day on the recording's own clock, on
#                              the day the recording started — or the next
#                              day, when it is earlier than the start clock
#                              (a recording that began at 23:50 and a request
#                              for 00:10 means twenty minutes in)
#   "YYYY-MM-DD HH:MM:SS"      a full date and time on the recording's clock,
#                              for recordings that run for days (fractional
#                              seconds allowed; no timezone suffix)
#
# parse_time_point() reads the text without knowing the recording;
# resolve_time_point() turns the result into seconds from the recording
# start once a reader has the header. Both readers call the same two
# functions, so a time means the same thing whatever the file format.

RELATIVE = "relative"
CLOCK = "clock"
DATETIME = "datetime"


@dataclass(frozen=True)
class TimePoint:
    """One time input after parsing, before it is anchored to a recording.

    seconds   RELATIVE: seconds from the recording start (already scaled);
              CLOCK:    seconds since midnight; DATETIME: unused
    at        DATETIME: the moment itself (naive, on the recording's clock)
    """
    kind: str
    seconds: float = 0.0
    at: "datetime | None" = None

    @property
    def is_clock(self) -> bool:
        """True when the value was typed as a clock or a date (it then
        needs the recording's start timestamp to mean anything)."""
        return self.kind != RELATIVE


def _looks_like_date(text: str) -> bool:
    # "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DDTHH:MM:SS": a 4-digit year, two
    # dashes, then a separator before the clock part.
    head = text[:11]
    return (len(text) > 11 and head[4:5] == "-" and head[7:8] == "-"
            and head[10:11] in (" ", "T") and head[:4].isdigit())


def _parse_datetime(value: str, *, field: str) -> datetime:
    text = str(value).strip().replace("T", " ", 1)
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise PlotInvalidInputError(
            PlotErrorCode.INVALID_ARGUMENT,
            f"{field} {value!r} is not a date and time in "
            "YYYY-MM-DD HH:MM:SS format (e.g. '2020-09-24 13:12:36')."
        ) from exc
    if moment.tzinfo is not None:
        raise PlotInvalidInputError(
            PlotErrorCode.INVALID_ARGUMENT,
            f"{field} {value!r} carries a timezone; give the time without one. "
            "Times are read on the recording's own clock."
        )
    return moment


def parse_time_point(value: "object", time_factor: float, *, field: str) -> TimePoint:
    """A raw start_time / end_time value -> TimePoint.

    `time_factor` scales a numeric value to seconds (from resolve_time_unit);
    clock and date strings carry their own unit and ignore it.
    """
    if is_clock_string(value):
        text = str(value).strip()
        if _looks_like_date(text):
            return TimePoint(DATETIME, at=_parse_datetime(text, field=field))
        return TimePoint(CLOCK, seconds=clock_to_seconds(text))
    try:
        return TimePoint(RELATIVE, seconds=float(value) * time_factor)
    except (TypeError, ValueError) as exc:
        raise PlotInvalidInputError(
            PlotErrorCode.INVALID_ARGUMENT, f"{field} {value!r} is not a number.") from exc


def resolve_time_point(point: TimePoint, recording_start: "datetime | None", *, field: str) -> float:
    """A TimePoint -> seconds from the recording start.

    A relative value is already that. A clock or date value needs the
    recording's start timestamp; a file without one is a data problem (the
    user can still give the time in seconds). The result may be negative or
    past the end of the recording — the caller checks the window as a whole.
    """
    if point.kind == RELATIVE:
        return float(point.seconds)
    if recording_start is None:
        raise PlotDataUnavailableError(
            PlotErrorCode.MISSING_REQUIRED_METADATA,
            f"{field} was given as a clock time but the recording has no "
            "start timestamp to anchor it to. Give the time in seconds instead.",
            error_facts={"field": field},
            source_stage=PlotErrorStage.READER,
        )
    if point.kind == CLOCK:
        offset = float(point.seconds) - seconds_since_midnight(recording_start)
        if offset < 0:
            offset += 24 * 3600.0
        return offset
    # DATETIME: the user's naive moment is read on the recording's clock, so
    # compare it with the start as a naive datetime of that same clock.
    start = recording_start.replace(tzinfo=None)
    return (point.at - start).total_seconds()
