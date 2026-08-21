"""
edf_processed_timeseries — voltage trace for one channel (or a bipolar
montage) over a time window, rendered from an EDF (iEEG/EEG) recording,
with optional DSP processing.

The template is a thin orchestrator: it validates the request, reads the
clip into a `Signal` (the shared contract from `processor.tools.ts_dsp`),
optionally runs a DSP pipeline over it, and plots it in the academic house
style. The signal-processing itself lives in the reusable `ts_dsp` family,
not here — so other EDF templates can share the same tools.

--------------------------------------------------------------------------
INPUTS (passed as keyword args to render())
--------------------------------------------------------------------------
Required:
  channel                 channel/label name, e.g. "F8", "EKG2".
  start_time              window start. A number (interpreted in
                          `time_unit`) OR a "HH:MM:SS" clock string.
  end_time OR duration    exactly one of the two. `end_time` is a number
                          (in `time_unit`) or "HH:MM:SS"; `duration` is a
                          number in `time_unit`.
  time_unit               unit for the NUMERIC time inputs (us/ms/s/min/h).
                          Required whenever any of start/end/duration is a
                          plain number; not needed when the times are given
                          as "HH:MM:SS" clock strings (which carry their own
                          unit). A nonsensical unit raises.
  y_range                 y-axis extent, two accepted forms:
                            * a single positive number N  -> [-N, +N]
                              (the "± shorthand").
                            * an explicit [min, max] pair -> used exactly.
                          Applied only while the y-axis stays voltage; if a
                          DSP step changes the y-domain (e.g. energy), the
                          y-axis is auto-scaled instead.
  y_unit                  voltage unit the y-axis (and y_range) are in
                          (V/mV/uV/nV). A nonsensical unit raises.

Montage (optional):
  channel2                a second channel name. When given, the trace is the
                          bipolar derivation channel - channel2 (e.g.
                          "F7-F8"). Both channels are validated (exist,
                          identical sampling rate, window fits both) before
                          any samples are read.

Processing (optional):
  pipeline                an ordered list of DSP steps, each a dict
                          {"tool": <name>, "params": {...}}. Steps are
                          resolved + validated (known tool, required params,
                          domain compatibility) and then run in order via
                          ts_dsp.apply_dsp_pipeline. Absent -> a raw trace.

NOTE on units. We do not default units; a missing unit raises (looking ahead
to processing stages whose domains — frequency, power, energy — have no one
sensible default). A supplied unit must be dimensionally valid or we raise.

NOTE on the x-axis display. The x-axis mirrors the FORMAT the user typed:
clock-string inputs ("14:07:00") render as H:M:S wall-clock ticks; numeric
inputs render as plain integers in the given unit (e.g. 973000000 for usec)
with no scientific-notation offset. See `_apply_x_axis_format`.

--------------------------------------------------------------------------
ERRORS (raised as RuntimeError / ts_dsp.ToolInputError so the processor
falls back to the agent loop; see processor/main.py::try_canned_template)
--------------------------------------------------------------------------
File:      not an EDF, unreadable, or channel not present.
Required:  channel / start / (end|duration) missing; both end & duration
           given but inconsistent.
Units:     time_unit not a time unit; y_unit not a voltage unit.
Window:    start past the end of the recording; computed end past the end
           of the recording; start after end; computed duration > 600 s;
           window so short it yields fewer than MIN_SAMPLES samples.
Montage:   the two channels are the same; the two channels have different
           sampling rates.
Pipeline:  unknown tool; missing/invalid tool parameter; a step whose
           required input domain doesn't match the running signal.
"""

from __future__ import annotations

from processor.templates.contract import TemplateArg
from processor.tools.ts_dsp import (
    Signal,
    apply_dsp_pipeline,
    probe_edf,
    read_signal,
    resolve_time_unit,
    resolve_volt_unit,
    seconds_to_clock,
)


############# TEMPLATE CONTRACT ##################
NAME = "edf_processed_timeseries"
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".edf",)

# Declarative contract — the generated schema (templates.json) and therefore
# the MCP plot_file tool description are built from these. Keep them in sync
# with render()'s keyword arguments and the INPUTS section of the module
# docstring above.
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
    TemplateArg("y_range", "positive number or [min, max] pair",
                description="y-axis extent: a single positive number N for a "
                            "symmetric [-N, +N], or an explicit [min, max] pair"),
    TemplateArg("y_unit", "string",
                description="voltage unit for the y-axis / y_range (V, mV, uV, nV)"),
)
EXAMPLE_ARGS = {
    "channel": "F8", "start_time": 10, "duration": 5,
    "time_unit": "s", "y_range": 200, "y_unit": "uV",
}
ARGS_NOTES = (
    "There are NO silent defaults: a missing channel, time, unit, or y_range "
    "makes the canned render fail (it then falls back to the agent). The "
    "window may not exceed 600 s."
)
# `pipeline` (optional render() kwarg) accepts ordered steps drawn from this
# tool-registry family; the MCP side renders the family's tool list from the
# generated <family> tools JSON.
PIPELINE_TOOLS = "ts_dsp"


############# LIMITS ##################
# Longest window we will render. A raw trace longer than this is both slow to
# draw and useless to read at screen resolution.
MAX_DURATION_S = 600.0
# Fewest samples worth plotting. A window a few samples wide (relative to the
# sampling rate) is not a signal, it is a dot cloud.
MIN_SAMPLES = 16


############# INPUT PARSING / DETERMINING X AND Y RANGE ##################
# detects whether user's input is an HH:MM:SS clock string
def _is_clock_time(value: "object") -> bool:
    return isinstance(value, str) and ":" in value

# detects whether user's input is a plain-number time
def _is_numeric_time(value: "object") -> bool:
    return (value is not None and not _is_clock_time(value)
            and not (isinstance(value, str) and value.strip() == ""))

def _parse_montage(channel: "object", channel2: "object") -> bool:
    """True when a second channel is given -> bipolar A-B derivation (montage).

    The two channels must differ (compared case-insensitively); an absent or
    blank `channel2` simply means single-channel mode.
    """
    montage = channel2 is not None and str(channel2).strip() != ""
    if montage and str(channel).strip().lower() == str(channel2).strip().lower():
        raise RuntimeError(
            f"A montage needs two different channels, but both are {channel!r}."
        )
    return montage

# parse a clock string into seconds since midnight 
def _parse_clock_time(value: str) -> float:
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

# dispatch on input kind: clock vs numeric time 
def _time_input_to_seconds(value: "object", unit_factor: float, recording_start_clock_s: float | None,
                           *, field: str) -> float:
    """Convert one time input to seconds *from the start of the recording*.

    Numeric inputs are scaled by `unit_factor`. Clock time ("HH:MM:SS")
    are interpreted as a wall-clock time-of-day and offset against the
    recording's own start-of-day clock (rolling to the next day if the clock
    time is earlier than the recording start).
    """
    # if the user provided the value as a clock time (HH:MM:SS)
    if _is_clock_time(value):
        # if the recording start timestamp is invalid, fallback 
        if recording_start_clock_s is None:
            raise RuntimeError(
                f"{field} was given as a clock time ({value!r}) but the "
                "recording has no start timestamp to anchor it to."
            )
        # convert the clock string to seconds since midnight, then offset it against the recording start
        offset = _parse_clock_time(value) - recording_start_clock_s
        if offset < 0:
            offset += 24 * 3600.0  # window is on the day after the start
        return offset
    # if the user provided the value as a numeric time, converts its unit to seconds
    try:
        return float(value) * unit_factor
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} {value!r} is not a number.") from exc


def _parse_y_range(y_range: "object") -> tuple[float, float]:
    """Return (ymin, ymax) from the ± shorthand or an explicit pair."""
    if y_range is None or (isinstance(y_range, str) and y_range.strip() == ""):
        raise RuntimeError(
            "A y-axis range is required — give either a single positive number "
            "N for a symmetric ±N range, or an explicit [min, max] pair."
        )
    # Single number -> symmetric ± range.
    if isinstance(y_range, (int, float)) and not isinstance(y_range, bool):
        n = float(y_range)
        if n <= 0:
            raise RuntimeError(
                f"y-axis range shorthand must be a positive number; got {n}."
            )
        return -n, n
    # Explicit [min, max] pair.
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



def _resolve_x_range(start_s: float, end_time_s: float | None, duration_s: float | None,
                    channel_duration_s: float) -> tuple[float, float]:
    """Validate + return (start_s, end_s), all in seconds from recording start."""
    if end_time_s is None and duration_s is None:
        raise RuntimeError(
            "Provide an end time or a duration for the window (exactly one)."
        )
    if end_time_s is not None and duration_s is not None:
        # Both given: allowed only if they agree (small float tolerance).
        implied_end = start_s + duration_s
        if abs(implied_end - end_time_s) > 1e-6:
            raise RuntimeError(
                f"end time and duration disagree: end implies {end_time_s:g}s "
                f"but start+duration implies {implied_end:g}s. Provide one."
            )
        end_s = end_time_s
    elif end_time_s is not None:
        end_s = end_time_s
    else:
        end_s = start_s + duration_s

    if start_s < 0:
        raise RuntimeError(f"start time ({start_s:g}s) is before the recording start.")
    if start_s >= channel_duration_s:
        raise RuntimeError(
            f"start time ({start_s:g}s) is at or past the end of the recording "
            f"({channel_duration_s:g}s)."
        )
    if end_s <= start_s:
        raise RuntimeError(
            f"end time ({end_s:g}s) is not after the start time ({start_s:g}s)."
        )
    if end_s > channel_duration_s:
        raise RuntimeError(
            f"end time ({end_s:g}s) is past the end of the recording "
            f"({channel_duration_s:g}s)."
        )
    if (end_s - start_s) > MAX_DURATION_S:
        raise RuntimeError(
            f"window is {end_s - start_s:g}s long, over the {MAX_DURATION_S:g}s maximum."
        )
    return start_s, end_s


############# PLOTTING ##################
# How each channel_kind reads in the figure title ("... for <noun> <name>").
_KIND_NOUN: dict[str, str] = {"single_channel": "channel", "montage": "montage"}


def _apply_x_axis_format(ax, signal: Signal) -> None:
    """Format the x tick LABELS to match how the user gave the times.

    Two styles, chosen upstream and carried on `signal.x_tick_style`:
      * "clock"   -> ticks read as H:M:S wall clock (signal.t is
                     seconds-since-midnight).
      * "numeric" -> ticks read as plain integers/decimals in signal.x_unit.
                     We force a non-scientific, no-offset formatter so a
                     value like 973000000 shows in full rather than as "9.73"
                     with a shared "1e8" offset in the corner.
    """
    from matplotlib.ticker import FuncFormatter, ScalarFormatter

    if signal.x_tick_style == "clock":
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: seconds_to_clock(v)))
        return
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(fmt)


def _plot_timeseries(signal: Signal, y_limits: "tuple[float, float] | None",
                     output_path: str) -> None:
    """Plot the timeseries in the academic style, labels from signal metadata.

    `y_limits` is the user's requested voltage range, or None to auto-scale
    (used when a DSP step has moved the y-axis off voltage).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Auto physical size. Width tracks the window length; height tracks the
    # y-extent (from the requested range, or from the data when auto-scaling),
    # both clamped so extreme inputs stay printable.
    span_x = float(signal.t[-1] - signal.t[0]) if signal.t.size > 1 else 1.0
    fig_w = max(6.0, min(18.0, 6.0 + span_x / (span_x + 1.0) * 8.0))
    if y_limits is not None:
        span_y = abs(y_limits[1] - y_limits[0])
    else:
        span_y = float(np.ptp(signal.y)) if signal.y.size else 1.0
    fig_h = max(3.0, min(10.0, 3.0 + span_y / (span_y + 1.0) * 3.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.plot(signal.t, signal.y, color="#2a6c97", linewidth=0.8)

    ax.set_xlim(signal.t[0], signal.t[-1])
    if y_limits is not None:
        ax.set_ylim(y_limits)
    ax.set_xlabel(signal.x_label())
    ax.set_ylabel(signal.y_label())
    noun = _KIND_NOUN.get(signal.channel_kind, "channel")
    ax.set_title(f"Processed timeseries for {noun} {signal.channel}")

    # Render the x tick labels in the user's own format/unit.
    _apply_x_axis_format(ax, signal)

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
    """Render a (optionally processed) timeseries clip from an EDF file.

    With `channel` alone this is a single-channel trace; `channel2` makes it a
    bipolar montage. A `pipeline` runs DSP tools over the clip before plotting.

    Written as a thin pipeline (probe -> validate -> read -> build Signal ->
    process -> plot) so the DSP stage slots in between read and plot without
    changing the plotting code. See the module docstring for the full
    input / error contract.
    """
    import os

    ####### 1. VALIDATE EDF FILE + PATH
    ext = os.path.splitext(target_file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise RuntimeError(
            f"{target_file_path!r} is not an EDF file (extension {ext!r}). "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    if not os.path.isfile(target_file_path):
        raise RuntimeError(f"Input file does not exist: {target_file_path!r}")

    ####### 2. VALIDATE REQUIRED SCALAR INPUTS (end_time/duration checked later in _resolve_x_range)
    if channel is None or str(channel).strip() == "":
        raise RuntimeError("A channel name is required (e.g. 'F8').")
    if start_time is None or (isinstance(start_time, str) and start_time.strip() == ""):
        raise RuntimeError("A start time is required.")

    ####### 3. VALIDATE X AND Y UNITS + PARSE Y-AXIS RANGE
    # The x-axis mirrors the user's input format: clock-string times render as
    # H:M:S; numeric times render in their own unit. `clock_mode` keys off the
    # start time (what anchors the window).
    clock_mode = _is_clock_time(start_time)
    # A time unit is only required when a time is given as a plain number;
    # clock strings ("HH:MM:SS") carry their own unit and don't need one.
    has_numeric_time = any(_is_numeric_time(v) for v in (start_time, end_time, duration))
    time_factor, time_symbol = resolve_time_unit(time_unit, required=has_numeric_time)
    volt_factor, volt_symbol = resolve_volt_unit(y_unit, required=True)
    y_limits = _parse_y_range(y_range)

    ####### 4. VALIDATE FS, DURATION FOR EACH REQUESTED CHANNEL
    try:
        _channels, fs_of, duration_of, start_clock_s = probe_edf(target_file_path)
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 — any reader failure = unreadable file
        raise RuntimeError(f"Could not read {target_file_path!r} as {ext} data: {exc}") from exc

    # Montage = a bipolar A-B derivation, active when a second channel is given.
    montage = _parse_montage(channel, channel2)

    # Validate every requested channel BEFORE reading any samples. Check if both channel exists/equal-rates/duration match 
    wanted = [str(channel)] + ([str(channel2)] if montage else [])
    rates = {c: fs_of(c) for c in wanted}
    durations = {c: duration_of(c) for c in wanted}
    if montage:
        fs_a, fs_b = rates[str(channel)], rates[str(channel2)]
        if abs(fs_a - fs_b) > 1e-6:
            raise RuntimeError(
                f"Montage {channel}-{channel2} needs both channels at the same "
                f"sampling rate, but {channel} is {fs_a:g} Hz and {channel2} is "
                f"{fs_b:g} Hz."
            )
    fs = rates[str(channel)]
    # The window must fit within BOTH channels -> validate against the shorter.
    channel_duration_s = min(durations.values())

    ####### 5. VALIDATE THE X RANGE (seconds from recording start)
    start_s = _time_input_to_seconds(start_time, time_factor, start_clock_s, field="start time")
    end_time_s = (
        _time_input_to_seconds(end_time, time_factor, start_clock_s, field="end time")
        if end_time is not None and not (isinstance(end_time, str) and end_time.strip() == "")
        else None
    )
    duration_s = (
        _time_input_to_seconds(duration, time_factor, None, field="duration")
        if duration is not None and not (isinstance(duration, str) and duration.strip() == "")
        else None
    )
    start_s, end_s = _resolve_x_range(start_s, end_time_s, duration_s, channel_duration_s)

    # Reject windows too short to be a signal (a handful of samples).
    expected_samples = int(round((end_s - start_s) * fs))
    if expected_samples < MIN_SAMPLES:
        raise RuntimeError(
            f"window spans only {expected_samples} sample(s) at {fs:g} Hz — "
            f"need at least {MIN_SAMPLES}. Widen the window."
        )

    ####### READ + BUILD THE RAW-STAGE SIGNAL 
    # read_signal owns the always-on plumbing every EDF template shares:
    # read + crop to the window (montage -> channel - channel2), convert SI
    # to the requested display units, and set the raw time/amplitude metadata
    # (x display follows the input format: clock -> H:M:S, numeric -> unit).
    signal = read_signal(
        target_file_path, channel, start_s, end_s,
        channel2=channel2, montage=montage,
        time_factor=time_factor, time_symbol=time_symbol,
        volt_factor=volt_factor, volt_symbol=volt_symbol,
        clock_mode=clock_mode, start_clock_s=start_clock_s,
    )

    ####### PROCESS (optional DSP pipeline)
    raw_y_domain = signal.y_domain
    signal = apply_dsp_pipeline(signal, pipeline)
    # The requested y_range is a voltage range; it only makes sense while the
    # y-axis is still voltage. If a step moved it to another domain (e.g.
    # energy), auto-scale instead of clipping to the voltage range.
    y_limits_final = y_limits if signal.y_domain == raw_y_domain else None

    ####### PLOT
    _plot_timeseries(signal, y_limits_final, output_path)