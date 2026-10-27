"""
Unit tests for ts_dsp/units.py — the time-point rules.

A start/end time can be typed three ways (a number in `time_unit`, an
"HH:MM:SS" clock time, a full "YYYY-MM-DD HH:MM:SS"). These tests pin how
each is read (`parse_time_point`) and how it becomes seconds from the
recording start once a reader knows when the recording began
(`resolve_time_point`). Both readers use these two functions, so the rules
are tested once here, not per format.

Layout:
  1. PARSING     text -> TimePoint, and what is rejected
  2. RESOLVING   TimePoint + recording start -> seconds from start
  3. CLOCK NAME  which clock an axis label should name

To run:
pytest processor/test_tools/test_ts_dsp_units.py
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from processor.errors import PlotDataUnavailableError, PlotErrorCode, PlotInvalidInputError
from processor.tools.ts_dsp import (
    TimePoint,
    clock_name,
    parse_time_point,
    resolve_time_point,
    seconds_since_midnight,
)

_START = datetime(2020, 9, 24, 13, 12, 36)          # naive: the recorder's clock
_START_CLOCK_S = 13 * 3600 + 12 * 60 + 36


# ---------------------------------------------------------------------------
# 1. PARSING
# ---------------------------------------------------------------------------
def test_number_is_relative_and_scaled_by_the_unit():
    assert parse_time_point(500, 1e-3, field="start time") == TimePoint("relative", 0.5)
    assert parse_time_point("2", 60.0, field="start time") == TimePoint("relative", 120.0)
    assert not parse_time_point(0, 1.0, field="start time").is_clock


def test_clock_string_is_seconds_since_midnight():
    p = parse_time_point("13:12:36", 1.0, field="start time")
    assert p == TimePoint("clock", _START_CLOCK_S) and p.is_clock
    assert parse_time_point("0:00:01.25", 1.0, field="end time").seconds == pytest.approx(1.25)


def test_date_string_keeps_the_whole_moment():
    p = parse_time_point("2020-09-24 13:12:36", 1.0, field="start time")
    assert p == TimePoint("datetime", at=_START) and p.is_clock
    # ISO "T" separator and fractional seconds are accepted too.
    q = parse_time_point("2020-09-25T00:00:02.5", 1.0, field="end time")
    assert q.at == datetime(2020, 9, 25, 0, 0, 2, 500000)


def test_unit_does_not_apply_to_clock_or_date_strings():
    # The string carries its own unit; a time_factor for numeric inputs
    # must not scale it.
    assert parse_time_point("0:00:10", 1e-3, field="start time").seconds == 10.0
    assert parse_time_point("2020-09-24 13:12:36", 60.0, field="start time").at == _START


@pytest.mark.parametrize("bad", [
    "abc",                    # not a number
    "13:12",                  # clock needs three fields
    "13:12:xx",               # non-numeric clock field
    "2020-13-40 13:12:36",    # impossible date
    "2020-09-24 13:12:36+00:00",   # a timezone: times are on the recording's clock
])
def test_uninterpretable_values_are_invalid_input(bad):
    with pytest.raises(PlotInvalidInputError) as ei:
        parse_time_point(bad, 1.0, field="start time")
    assert ei.value.error_code is PlotErrorCode.INVALID_ARGUMENT
    assert "start time" in str(ei.value) or "Clock time" in str(ei.value)


# ---------------------------------------------------------------------------
# 2. RESOLVING
# ---------------------------------------------------------------------------
def test_relative_needs_no_recording_start():
    p = TimePoint("relative", 7.5)
    assert resolve_time_point(p, None, field="start time") == 7.5
    assert resolve_time_point(p, _START, field="start time") == 7.5


def test_clock_is_anchored_to_the_start_clock():
    # 13:12:38 is two seconds after a recording that began at 13:12:36.
    p = TimePoint("clock", _START_CLOCK_S + 2)
    assert resolve_time_point(p, _START, field="start time") == pytest.approx(2.0)


def test_clock_earlier_than_the_start_clock_is_the_next_day():
    # Began at 23:59:50; "00:00:02" can only mean twelve seconds in.
    late = datetime(2020, 9, 24, 23, 59, 50)
    assert resolve_time_point(TimePoint("clock", 2.0), late, field="end time") == pytest.approx(12.0)


def test_date_is_the_difference_from_the_start_moment():
    late = datetime(2020, 9, 24, 23, 59, 50)
    p = TimePoint("datetime", at=datetime(2020, 9, 26, 0, 0, 2))     # a day and 12 s on
    assert resolve_time_point(p, late, field="end time") == pytest.approx(86400 + 12)
    # Earlier than the start comes back negative; the caller decides.
    before = TimePoint("datetime", at=datetime(2020, 9, 24, 23, 59, 0))
    assert resolve_time_point(before, late, field="start time") == pytest.approx(-50.0)


def test_date_is_read_on_the_recording_clock_when_the_start_is_aware():
    # NWB stores the start in UTC; the user's naive date is read as UTC too.
    aware = _START.replace(tzinfo=timezone.utc)
    p = TimePoint("datetime", at=datetime(2020, 9, 24, 13, 12, 40))
    assert resolve_time_point(p, aware, field="start time") == pytest.approx(4.0)
    assert resolve_time_point(TimePoint("clock", _START_CLOCK_S + 4), aware, field="start time") == pytest.approx(4.0)


def test_clock_or_date_without_a_recording_start_is_a_data_problem():
    # The file lacks what the request needs: data_unavailable, not a typo,
    # and the message says seconds would still work.
    for p in (TimePoint("clock", 10.0), TimePoint("datetime", at=_START)):
        with pytest.raises(PlotDataUnavailableError) as ei:
            resolve_time_point(p, None, field="start time")
        assert ei.value.error_code is PlotErrorCode.MISSING_REQUIRED_METADATA
        assert "Give the time in seconds" in str(ei.value)


# ---------------------------------------------------------------------------
# 3. CLOCK NAME
# ---------------------------------------------------------------------------
def test_clock_name_is_local_for_naive_and_the_zone_for_aware():
    assert clock_name(_START) == "local"
    assert clock_name(None) == "local"
    assert clock_name(_START.replace(tzinfo=timezone.utc)) == "UTC"


def test_seconds_since_midnight():
    assert seconds_since_midnight(_START) == _START_CLOCK_S
    assert seconds_since_midnight(datetime(2020, 1, 1, 0, 0, 0, 500000)) == 0.5
