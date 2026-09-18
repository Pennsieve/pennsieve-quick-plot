"""
edf_processed_timeseries — voltage trace for one channel (or a bipolar
montage) over a time window, rendered from an EDF (iEEG/EEG) recording,
with optional DSP processing.

render() is five lines, one per stage; see processor/README.md for the
call chain with an example request:

    params = _parse(...)                   # user's inputs -> typed RenderParams
    _validate(params)                      # rules that need only those inputs
    signal = load_signal(path, params)     # file -> raw Signal   (processor/readers)
    signal = apply_dsp_pipeline(signal, params.pipeline)   # optional DSP (tools/ts_dsp)
    _plot(signal, params, output_path)     # figure.png

Where a check lives: anything checkable without the file is in _parse /
_validate; anything that needs the file's header (channels exist, window
fits the recording, clock times anchored to the recording start, header
voltage unit) is in the reader.

--------------------------------------------------------------------------
INPUTS (keyword args to render(), i.e. the MCP `template_args`)
--------------------------------------------------------------------------
Required:
  channel                 channel/label name, e.g. "F8", "EKG2".
  start_time              window start. A number (interpreted in
                          `time_unit`) OR a "HH:MM:SS" clock string.
  end_time OR duration    at least one. `end_time` is a number (in
                          `time_unit`) or "HH:MM:SS"; `duration` is a number
                          in `time_unit`. Both may be given if they agree.
  time_unit               unit for the NUMERIC time inputs (us/ms/s/min/h).
                          Required whenever any of start/end/duration is a
                          plain number; not needed when the times are given
                          as "HH:MM:SS" clock strings.
Optional:
  channel2                second channel; the trace is then the bipolar
                          derivation channel - channel2 (e.g. "F7-F8").
  y_range                 y-axis extent: a single positive number N -> [-N, +N],
                          or an explicit [min, max] pair. Omitted -> auto-scale.
                          Applied only while the y-axis is still voltage; if a
                          DSP step changes the y-domain (e.g. energy, psd),
                          the y-axis is auto-scaled instead.
  y_unit                  voltage unit the y-axis (and y_range) are displayed
                          in (V/mV/uV/nV). Omitted -> the unit the file itself
                          declares for the channel (blank -> µV).
  pipeline                ordered list of DSP steps, each {"tool": <name>,
                          "params": {...}}, run via ts_dsp.apply_dsp_pipeline.

The x-axis mirrors the FORMAT the user typed: clock-string inputs render as
H:M:S wall-clock ticks; numeric inputs render as plain numbers in the given
unit (no scientific-notation offset).

--------------------------------------------------------------------------
ERRORS (RuntimeError / ts_dsp.ToolInputError; the processor then falls back
to the agent loop — see processor/main.py::try_canned_template)
--------------------------------------------------------------------------
_parse / _validate:  wrong extension; channel or start missing; neither end
                     nor duration; both montage channels the same; time_unit
                     missing for numeric times or not a time unit; y_unit not a
                     voltage unit; malformed y_range or clock string.
reader:              file unreadable; channel not present; montage channels
                     with different sampling rates; header declares an
                     unrecognizable voltage unit; window outside the
                     recording, over the duration cap, or too few samples;
                     end and duration both given but inconsistent.
pipeline:            unknown tool; missing/invalid tool parameter; a step
                     whose required input domain doesn't match the signal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import NamedTuple

from processor.readers import load_signal, MAX_DURATION_S, MIN_SAMPLES  # noqa: F401 (limits re-exported for docs/tests)
from processor.templates.contract import TemplateArg
from processor.tools.ts_dsp import (
    Signal,
    apply_dsp_pipeline,
    clock_to_seconds,
    is_clock_string,
    resolve_time_unit,
    resolve_volt_unit,
    seconds_to_clock,
)


############# TEMPLATE CONTRACT ##################
NAME = "edf_processed_timeseries"
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".edf",)

# Declarative contract — the generated schema (templates.json) and therefore
# the MCP plot_file tool description are built from these. Keep them in sync
# with render()'s keyword arguments and the INPUTS section above.
SUMMARY = (
    "voltage trace for one channel, or a bipolar montage of two channels, "
    "over a chosen time window, with optional DSP processing (scalp EEG / "
    "intracranial iEEG); \"plot channel F8 from 10s to 15s\", \"show the "
    "F7-F8 montage for the first minute\""
)
ARGS_SPEC: tuple[TemplateArg, ...] = (
    TemplateArg("channel", "string",
                description="channel/label name, e.g. \"F8\", \"EKG2\""),
    TemplateArg("channel2", "string", required=False,
                description="second channel; when given, the trace is the "
                            "bipolar montage channel-channel2 (e.g. \"F7\"-\"F8\"). "
                            "Both channels must share a sampling rate"),
    TemplateArg("start_time", "number or \"HH:MM:SS\" string",
                description="window start: a number (in `time_unit`) OR a "
                            "wall-clock \"HH:MM:SS\" string"),
    TemplateArg("end_time", "number or \"HH:MM:SS\" string", required=False,
                description="window end; provide EXACTLY ONE of end_time / duration"),
    TemplateArg("duration", "number", required=False,
                description="window length in `time_unit`; provide EXACTLY ONE "
                            "of end_time / duration"),
    TemplateArg("time_unit", "string", required=False,
                description="unit for NUMERIC start/end/duration values (us, ms, "
                            "s, min, h). REQUIRED whenever any of those is a plain "
                            "number; omit only when all times are \"HH:MM:SS\" "
                            "clock strings"),
    TemplateArg("y_range", "positive number or [min, max] pair", required=False,
                description="y-axis extent: a single positive number N for a "
                            "symmetric [-N, +N], or an explicit [min, max] pair; "
                            "omit to auto-scale the y-axis to the data. "
                            "Recommended for raw voltage traces (EEG is "
                            "conventionally read at a fixed scale); ignored when "
                            "the pipeline moves the y-axis off voltage (e.g. "
                            "fft, psd, energy)"),
    TemplateArg("y_unit", "string", required=False,
                description="voltage unit the y-axis / y_range are displayed in "
                            "(V, mV, uV, nV); omit to use the unit the file "
                            "itself declares for the channel"),
)
EXAMPLE_ARGS = {
    "channel": "F8", "start_time": 10, "duration": 5,
    "time_unit": "s", "y_range": 200, "y_unit": "uV",
}
ARGS_NOTES = (
    "There are NO silent defaults for the data selection: a missing channel, "
    "time, or time_unit (for numeric times) makes the canned render fail (it "
    "then falls back to the agent). y_range and y_unit are display settings "
    "with truthful defaults: omitted y_range auto-scales the y-axis, and "
    "omitted y_unit displays in the unit the file itself declares. The window "
    "may not exceed 600 s."
)
# `pipeline` (optional render() kwarg) accepts ordered steps drawn from this
# tool-registry family; the MCP side renders the family's tool list from the
# generated <family> tools JSON.
PIPELINE_TOOLS = "ts_dsp"

# The y_domain a reader produces. While the Signal is still in this domain
# the user's y_range applies; once a DSP step changes it, we auto-scale.
_RAW_Y_DOMAIN = "amplitude"


############# PARAMETERS ##################
class TimePoint(NamedTuple):
    """One time input after parsing.

    seconds   numeric input -> seconds (already scaled by time_unit);
              clock input   -> seconds since midnight.
    is_clock  which of the two `seconds` means. The reader anchors a clock
              value to the recording's own start time; a numeric value is
              already relative to the recording start.
    """
    seconds: float
    is_clock: bool


@dataclass(frozen=True)
class RenderParams:
    """render()'s keyword arguments, parsed into typed values.

    Built by _parse(), checked by _validate(), then read by the reader
    (channel/window/volt) and by _plot() (display choices).
    """
    channel: "str | None"
    channel2: "str | None"
    montage: bool
    start: "TimePoint | None"
    end: "TimePoint | None"
    duration_s: "float | None"
    time_factor: float            # numeric time input unit -> seconds
    time_symbol: "str | None"     # canonical display symbol for that unit
    clock_mode: bool              # start was a clock string -> H:M:S x ticks
    y_limits: "tuple[float, float] | None"
    volt: "tuple[float, str] | None"   # (factor-to-volts, symbol) if y_unit given
    pipeline: "list | None"


def _blank(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _parse_time(value, time_factor: float, *, field: str) -> "TimePoint | None":
    if _blank(value):
        return None
    if is_clock_string(value):
        return TimePoint(clock_to_seconds(value), is_clock=True)
    try:
        return TimePoint(float(value) * time_factor, is_clock=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} {value!r} is not a number.") from exc


def _parse_y_range(y_range) -> "tuple[float, float] | None":
    """(ymin, ymax) from the ± shorthand or an explicit pair; None = auto-scale."""
    if _blank(y_range):
        return None
    if isinstance(y_range, (int, float)) and not isinstance(y_range, bool):
        n = float(y_range)
        if n <= 0:
            raise RuntimeError(
                f"y-axis range shorthand must be a positive number; got {n}."
            )
        return -n, n
    if isinstance(y_range, (list, tuple)) and len(y_range) == 2:
        try:
            ymin, ymax = float(y_range[0]), float(y_range[1])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"y-axis range {y_range!r} has non-numeric bounds.") from exc
        if not ymin < ymax:
            raise RuntimeError(
                f"y-axis range min ({ymin}) must be strictly less than max ({ymax})."
            )
        return ymin, ymax
    raise RuntimeError(
        f"y-axis range {y_range!r} must be a single positive number "
        "(± shorthand) or a [min, max] pair."
    )


def _parse(*, channel, channel2, start_time, end_time, duration, time_unit,
           y_range, y_unit, pipeline) -> RenderParams:
    """Raw template_args -> RenderParams. Turns strings and numbers into
    typed values; opens no file. Raises only where a value cannot be
    interpreted at all (bad unit spelling, malformed clock string or y_range)."""
    # A time unit is only needed when a time is a plain number; clock
    # strings carry their own unit.
    has_numeric_time = any(
        not _blank(v) and not is_clock_string(v) for v in (start_time, end_time, duration)
    )
    time_factor, time_symbol = resolve_time_unit(time_unit, required=has_numeric_time)

    duration_s = None
    if not _blank(duration):
        try:
            duration_s = float(duration) * time_factor
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"duration {duration!r} is not a number.") from exc

    return RenderParams(
        channel=None if _blank(channel) else str(channel),
        channel2=None if _blank(channel2) else str(channel2),
        montage=not _blank(channel2),
        start=_parse_time(start_time, time_factor, field="start time"),
        end=_parse_time(end_time, time_factor, field="end time"),
        duration_s=duration_s,
        time_factor=time_factor,
        time_symbol=time_symbol,
        clock_mode=is_clock_string(start_time),
        y_limits=_parse_y_range(y_range),
        volt=None if _blank(y_unit) else resolve_volt_unit(y_unit, required=True),
        pipeline=pipeline,
    )


def _validate(params: RenderParams) -> None:
    """Rules that need only the user's inputs. Anything that needs the
    file's header (channel exists, window fits, ...) is the reader's job."""
    if params.channel is None:
        raise RuntimeError("A channel name is required (e.g. 'F8').")
    if params.start is None:
        raise RuntimeError("A start time is required.")
    if params.end is None and params.duration_s is None:
        raise RuntimeError(
            "Provide an end time or a duration for the window (exactly one)."
        )
    if params.montage and params.channel.strip().lower() == params.channel2.strip().lower():
        raise RuntimeError(
            f"A montage needs two different channels, but both are {params.channel!r}."
        )


############# PLOT ##################
# How each channel_kind reads in the figure title ("... for <noun> <name>").
_KIND_NOUN: dict[str, str] = {"single_channel": "channel", "montage": "montage"}


def _x_axis(signal: Signal, params: RenderParams):
    """(x values to plot, axis label, tick formatter or None).

    The reader always gives time in seconds from the recording start. For a
    time axis we show it the way the user typed it: numeric input -> plain
    numbers in the user's unit; clock input -> H:M:S wall clock anchored at
    the recording start. A non-time axis (a spectrum) is shown as-is.
    """
    from matplotlib.ticker import FuncFormatter, ScalarFormatter

    if signal.x_domain != "time":
        return signal.t, signal.x_label(), None
    if params.clock_mode:
        t0 = signal.recording_start_clock_s or 0.0
        return (signal.t, "time (h:m:s)",
                FuncFormatter(lambda v, _pos: seconds_to_clock(v + t0)))
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    return signal.t / params.time_factor, f"time ({params.time_symbol})", fmt


def _plot(signal: Signal, params: RenderParams, output_path: str) -> None:
    """Draw the Signal in the academic house style; labels from its metadata."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # The user's y_range is a voltage range: apply it only while the y-axis
    # is still voltage; a DSP step that changed the domain -> auto-scale.
    y_limits = params.y_limits if signal.y_domain == _RAW_Y_DOMAIN else None
    x, x_label, x_formatter = _x_axis(signal, params)

    # Auto physical size. Width tracks the window length; height tracks the
    # y-extent, both clamped so extreme inputs stay printable.
    span_x = float(x[-1] - x[0]) if x.size > 1 else 1.0
    fig_w = max(6.0, min(18.0, 6.0 + span_x / (span_x + 1.0) * 8.0))
    span_y = abs(y_limits[1] - y_limits[0]) if y_limits is not None else (
        float(np.ptp(signal.y)) if signal.y.size else 1.0)
    fig_h = max(3.0, min(10.0, 3.0 + span_y / (span_y + 1.0) * 3.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.plot(x, signal.y, color="#2a6c97", linewidth=0.8)
    ax.set_xlim(x[0], x[-1])
    if y_limits is not None:
        ax.set_ylim(y_limits)
    ax.set_xlabel(x_label)
    ax.set_ylabel(signal.y_label())
    noun = _KIND_NOUN.get(signal.channel_kind, "channel")
    ax.set_title(f"Processed timeseries for {noun} {signal.channel}")
    if x_formatter is not None:
        ax.xaxis.set_major_formatter(x_formatter)

    # Academic styling: black left/bottom axes, no top/right frame, ticks
    # pointing out, no gridlines, white background.
    ax.grid(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(axis="both", direction="out", color="black", labelsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


############# ENTRY POINT ##################
def render(
    target_file_path: str,
    output_path: str,
    *,
    channel: str | None = None,
    channel2: str | None = None,
    start_time: "object" = None,
    end_time: "object" = None,
    duration: "object" = None,
    time_unit: str | None = None,
    y_range: "object" = None,
    y_unit: str | None = None,
    pipeline: "object" = None,
) -> None:
    """Render an (optionally processed) timeseries clip from an EDF file."""
    ext = os.path.splitext(target_file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise RuntimeError(
            f"{target_file_path!r} is not an EDF file (extension {ext!r}). "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    params = _parse(channel=channel, channel2=channel2, start_time=start_time,
                    end_time=end_time, duration=duration, time_unit=time_unit,
                    y_range=y_range, y_unit=y_unit, pipeline=pipeline)
    _validate(params)
    signal = load_signal(target_file_path, params)
    signal = apply_dsp_pipeline(signal, params.pipeline)
    _plot(signal, params, output_path)
