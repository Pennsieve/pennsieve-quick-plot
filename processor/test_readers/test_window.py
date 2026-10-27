"""
Tests for processor.readers.window — the window rules every reader shares.

These need no file: resolve_window() takes what a header would supply
(recording start, length, sampling rate) and the template's RenderParams.
The per-format reader tests (test_edf_to_signal.py, ...) only check that
their reader hands those values over; the rules themselves are pinned here
once.

To run:
pytest processor/test_readers/test_window.py
"""

from __future__ import annotations

from datetime import datetime

import pytest

from processor.errors import (
    PlotDataUnavailableError,
    PlotErrorCode,
    PlotInvalidInputError,
    PlotResourceLimitError,
)
from processor.readers import MAX_DURATION_S, MIN_SAMPLES
from processor.readers.window import resolve_window
from processor.templates.ieeg_processed_timeseries import RenderParams
from processor.tools.ts_dsp import TimePoint

_START = datetime(2020, 9, 24, 13, 12, 36)
_START_CLOCK_S = 13 * 3600 + 12 * 60 + 36


def _params(start=TimePoint("relative", 0.0), end=None, duration_s=2.0) -> RenderParams:
    return RenderParams(
        channel="F7", channel2=None, montage=False,
        start=start, end=end, duration_s=duration_s,
        time_factor=1.0, time_symbol="s", clock_mode=start.is_clock,
        y_limits=None, volt=None, pipeline=None,
    )


def _window(params, *, recording_start=_START, recording_s=20.0, fs=256.0):
    return resolve_window(params, recording_start=recording_start,
                          recording_s=recording_s, fs=fs)


def test_relative_start_plus_duration():
    assert _window(_params(TimePoint("relative", 3.0), duration_s=2.0)) == (3.0, 5.0)


def test_relative_start_and_end():
    p = _params(TimePoint("relative", 3.0), end=TimePoint("relative", 5.0), duration_s=None)
    assert _window(p) == (3.0, 5.0)


def test_clock_and_date_inputs_are_anchored_to_the_recording_start():
    clock = _params(TimePoint("clock", _START_CLOCK_S + 2), duration_s=1.0)
    assert _window(clock) == (pytest.approx(2.0), pytest.approx(3.0))
    date = _params(TimePoint("datetime", at=datetime(2020, 9, 24, 13, 12, 40)), duration_s=1.0)
    assert _window(date) == (pytest.approx(4.0), pytest.approx(5.0))


def test_clock_input_needs_a_recording_start():
    p = _params(TimePoint("clock", _START_CLOCK_S), duration_s=1.0)
    with pytest.raises(PlotDataUnavailableError) as ei:
        _window(p, recording_start=None)
    assert ei.value.error_code is PlotErrorCode.MISSING_REQUIRED_METADATA


def test_end_and_duration_must_agree():
    agree = _params(TimePoint("relative", 1.0), end=TimePoint("relative", 3.0), duration_s=2.0)
    assert _window(agree) == (1.0, 3.0)
    disagree = _params(TimePoint("relative", 1.0), end=TimePoint("relative", 5.0), duration_s=2.0)
    with pytest.raises(PlotInvalidInputError, match="disagree"):
        _window(disagree)


@pytest.mark.parametrize("start, end, duration_s, why", [
    (-1.0, None, 2.0, "before the recording start"),
    (25.0, None, 1.0, "at or past the end"),
    (5.0, 5.0, None, "not after the start"),
    (10.0, 25.0, None, "past the end of the recording"),
])
def test_window_must_lie_inside_the_recording(start, end, duration_s, why):
    p = _params(TimePoint("relative", start),
                end=None if end is None else TimePoint("relative", end),
                duration_s=duration_s)
    with pytest.raises(PlotInvalidInputError, match=why) as ei:
        _window(p, recording_s=20.0)
    assert ei.value.error_code is PlotErrorCode.INVALID_TIME_WINDOW
    assert ei.value.error_facts["recording_s"] == 20.0


def test_window_longer_than_the_cap_is_a_resource_limit():
    p = _params(duration_s=MAX_DURATION_S + 5)
    with pytest.raises(PlotResourceLimitError) as ei:
        _window(p, recording_s=1000.0)
    assert ei.value.error_code is PlotErrorCode.WINDOW_TOO_LARGE


def test_window_with_too_few_samples_is_data_unavailable():
    p = _params(duration_s=0.5)          # 2 samples at 4 Hz < MIN_SAMPLES
    with pytest.raises(PlotDataUnavailableError, match="Widen the window") as ei:
        _window(p, fs=4.0)
    assert ei.value.error_code is PlotErrorCode.INSUFFICIENT_SAMPLES
    assert ei.value.error_facts["min_samples"] == MIN_SAMPLES
