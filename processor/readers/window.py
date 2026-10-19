"""
readers.window — the window-vs-recording rules every reader applies.

A reader knows its file format; the rules about the *window* do not depend
on it: the start and end resolve to seconds from the recording start, end
and duration must agree, the window must lie inside the recording, and it
must be neither longer than one plot can show nor shorter than is worth
drawing. They live here so each reader applies exactly the same rules and
raises exactly the same errors, and a new format adds none of its own.

    resolve_window(params, recording_start=..., recording_s=..., fs=...)
        -> (start_s, end_s)

The caller passes what its header knows: when the recording started (for
clock and date inputs), how long it is, and the sampling rate of the
channel being read (for the sample-count floor). Standard library only;
the heavy imports stay in the readers.

Clip limits shared by every reader:
  MAX_DURATION_S  longest window we will read (a longer raw trace is both
                  slow to draw and unreadable at screen resolution)
  MIN_SAMPLES     fewest samples worth plotting (a handful of samples is a
                  dot cloud, not a signal)
"""

from __future__ import annotations

from datetime import datetime

from processor.errors import (
    PlotDataUnavailableError,
    PlotErrorCode,
    PlotErrorStage,
    PlotInvalidInputError,
    PlotResourceLimitError,
)
from processor.tools.ts_dsp import resolve_time_point

MAX_DURATION_S = 600.0
MIN_SAMPLES = 16


def resolve_window(params, *, recording_start: "datetime | None",
                   recording_s: float, fs: float) -> tuple[float, float]:
    """The request's window as (start_s, end_s), seconds from the recording
    start, checked against the recording's length and the clip limits.

    `params` is the template's RenderParams (start, end, duration_s).
    Clock and date inputs are anchored to `recording_start` by the shared
    ts_dsp rule; `recording_s` is the recording's length for the channel(s)
    being read; `fs` their sampling rate.
    """
    start_s = resolve_time_point(params.start, recording_start, field="start time")
    end_s = resolve_time_point(params.end, recording_start, field="end time") if params.end else None
    if end_s is not None and params.duration_s is not None:
        implied_end = start_s + params.duration_s
        if abs(implied_end - end_s) > 1e-6:
            raise _bad_window(
                f"end time and duration disagree: end implies {end_s:g}s but "
                f"start+duration implies {implied_end:g}s. Provide one.",
                start_s=start_s, end_s=end_s, recording_s=recording_s)
    if end_s is None:
        end_s = start_s + params.duration_s
    if start_s < 0:
        raise _bad_window(f"start time ({start_s:g}s) is before the recording start.",
                          start_s=start_s, end_s=end_s, recording_s=recording_s)
    if start_s >= recording_s:
        raise _bad_window(
            f"start time ({start_s:g}s) is at or past the end of the recording "
            f"({recording_s:g}s).",
            start_s=start_s, end_s=end_s, recording_s=recording_s)
    if end_s <= start_s:
        raise _bad_window(
            f"end time ({end_s:g}s) is not after the start time ({start_s:g}s).",
            start_s=start_s, end_s=end_s, recording_s=recording_s)
    if end_s > recording_s:
        raise _bad_window(
            f"end time ({end_s:g}s) is past the end of the recording ({recording_s:g}s).",
            start_s=start_s, end_s=end_s, recording_s=recording_s)
    if (end_s - start_s) > MAX_DURATION_S:
        # The request is well-formed and the data exists; it is just more
        # than one plot is allowed to pull. Ask for less.
        raise PlotResourceLimitError(
            PlotErrorCode.WINDOW_TOO_LARGE,
            f"window is {end_s - start_s:g}s long, over the {MAX_DURATION_S:g}s maximum.",
            error_facts={"window_s": end_s - start_s, "max_window_s": MAX_DURATION_S},
            source_stage=PlotErrorStage.READER,
        )
    expected_samples = int(round((end_s - start_s) * fs))
    if expected_samples < MIN_SAMPLES:
        raise PlotDataUnavailableError(
            PlotErrorCode.INSUFFICIENT_SAMPLES,
            f"window spans only {expected_samples} sample(s) at {fs:g} Hz — "
            f"need at least {MIN_SAMPLES}. Widen the window.",
            error_facts={"samples": expected_samples, "min_samples": MIN_SAMPLES, "fs": fs},
            source_stage=PlotErrorStage.READER,
        )
    return start_s, end_s


def _bad_window(message: str, *, start_s: float, end_s, recording_s: float) -> PlotInvalidInputError:
    """All window-vs-recording complaints share one code and the same facts,
    so the chat agent can propose a window that fits."""
    return PlotInvalidInputError(
        PlotErrorCode.INVALID_TIME_WINDOW, message,
        error_facts={"start_s": start_s, "end_s": end_s, "recording_s": recording_s},
        source_stage=PlotErrorStage.READER,
    )


__all__ = ["resolve_window", "MAX_DURATION_S", "MIN_SAMPLES"]
