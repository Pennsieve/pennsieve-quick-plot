"""
edf_processed_timeseries — raw voltage trace for one channel over a time
window, rendered from an EDF (iEEG/EEG) recording.

This is the FIRST step of the ieeg template family: it draws the *raw*
signal only. No filtering / transformation / feature-extraction tools are
wired in yet. The module is, however, structured so those tools can be
added later without touching the plotting code (see "PIPELINE DESIGN"
below).

--------------------------------------------------------------------------
WHAT IT DOES
--------------------------------------------------------------------------
Given a channel name and a time window, it reads just that slice of the
recording and plots amplitude vs. time in the academic house style (black
left/bottom axes, no top/right frame, out-pointing ticks, no gridlines,
white background) — the same styling as tsv_sessions_lineplot.

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

Montage (optional):
  channel2                a second channel name. When given, the trace is the
                          bipolar derivation channel - channel2 (e.g.
                          "F7-F8") instead of a single channel. Both channels
                          are validated (exist, identical sampling rate,
                          window fits both) before any samples are read.
  time_unit               unit for the NUMERIC time inputs (us/ms/s/min/h).
                          Required whenever any of start/end/duration is a
                          plain number; not needed when the times are given
                          as "HH:MM:SS" clock strings (which carry their own
                          unit). A nonsensical unit raises.
  y_range                 y-axis extent, two accepted forms:
                            * a single positive number N  -> [-N, +N]
                              (the "± shorthand").
                            * an explicit [min, max] pair -> used exactly.
  y_unit                  voltage unit the y-axis (and y_range) are in
                          (V/mV/uV/nV). A nonsensical unit raises.

NOTE on units. We deliberately do NOT default units. Units are required so
the label always reflects the user's intent, and — looking ahead to the
processing pipeline — because later stages move the signal into entirely
different domains (frequency, power, z-score) where no single default is
meaningful. A missing unit therefore raises and asks the user to supply
one; a supplied unit must be dimensionally valid (time for x, voltage for
y) or we raise. Required *values* (channel, start, one of end/duration,
y_range) likewise have no default and raise when missing.

NOTE on the x-axis display. The x-axis mirrors the FORMAT the user typed:
clock-string inputs ("14:07:00") render as H:M:S wall-clock ticks; numeric
inputs render as plain integers in the given unit (e.g. 973000000 for
usec) with no scientific-notation offset. See `_apply_x_axis_format`.

--------------------------------------------------------------------------
ERRORS (all raised as RuntimeError so the processor falls back to the
agent loop; see processor/main.py::try_canned_template)
--------------------------------------------------------------------------
File:      not an EDF, unreadable, or channel not present.
Required:  channel / start / (end|duration) missing; both end & duration
           given but inconsistent.
Units:     time_unit not a time unit; y_unit not a voltage unit.
Window:    start past the end of the recording; computed end past the end
           of the recording; start after end; computed duration > 600 s;
           window so short it yields fewer than MIN_SAMPLES samples.
Montage:   the two channels are the same; the two channels have different
           sampling rates. (Window fit is enforced against the shorter of
           the two, via the Window checks above.)

--------------------------------------------------------------------------
PIPELINE DESIGN (forward-looking; only the raw stage exists today)
--------------------------------------------------------------------------
The signal moves through the module as a `Signal` bundle that carries its
own axis metadata: (x_domain, x_unit, y_domain, y_unit). The raw stage
initialises this to time / <time unit> and amplitude / <voltage unit>. The
plotting function reads ONLY that metadata to build its axis labels — it
never hard-codes "time (…)" or "amplitude (…)". So when later stages are
added (a band-pass filter carries the domains through unchanged; an FFT
flips x to frequency/Hz and y to magnitude; a band-power step sets y to
power/uV²·Hⁿ), the plotter needs no change: it just renders whatever the
final `Signal.meta` says. `render()` is the pipeline seam where those
stages will be chained.
"""

from __future__ import annotations

from dataclasses import dataclass


############# TEMPLATE CONTRACT ##################
NAME = "edf_processed_timeseries"
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".edf",)


############# LIMITS ##################
# Longest window we will render. A raw trace longer than this is both
# slow to draw and useless to read at screen resolution.
MAX_DURATION_S = 600.0
# Fewest samples worth plotting. A window a few samples wide (relative to
# the sampling rate) is not a signal, it is a dot cloud.
MIN_SAMPLES = 16


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


############# SIGNAL BUNDLE ##################
@dataclass
class Signal:
    """A signal clip plus the axis metadata the plotter renders from.

    `t` and `y` are already expressed in `x_unit` / `y_unit`. Processing
    stages (added later) take a Signal and return a Signal, updating the
    four *_domain / *_unit fields as the physical meaning changes.
    """

    t: "object"          # np.ndarray of x values, in x_unit
    y: "object"          # np.ndarray of y values, in y_unit
    fs: float            # sampling rate (Hz) of the source recording
    channel: str
    # What `channel` represents: "single_channel" for one trace, "montage"
    # for a bipolar A-B derivation (then `channel` holds "A-B"). The plot
    # maps this to a title noun via _KIND_NOUN.
    channel_kind: str = "single_channel"
    x_domain: str = "time"
    x_unit: str = "s"
    y_domain: str = "amplitude"
    y_unit: str = "µV"
    # How the x tick LABELS are rendered (the axis data is always numeric):
    #   "numeric" -> plain integers/decimals in x_unit, no 1e9-style offset.
    #   "clock"   -> H:M:S wall clock; `t` then holds seconds-since-midnight.
    x_tick_style: str = "numeric"

    def x_label(self) -> str:
        return f"{self.x_domain} ({self.x_unit})"

    def y_label(self) -> str:
        return f"{self.y_domain} ({self.y_unit})"


############# UNIT / INPUT PARSING HELPERS ##################
def _normalise_unit_key(unit: str) -> str:
    """Lowercase, strip whitespace, and drop a trailing plural 's'."""
    key = str(unit).strip().lower()
    # "seconds" -> "second", "microvolts" -> "microvolt", but leave a bare
    # "s"/"ms"/"us"/"mv"/"uv" alone (they are already the short symbols).
    if len(key) > 2 and key.endswith("s"):
        key = key[:-1]
    return key


def _resolve_time_unit(unit: str | None, *, required: bool) -> tuple[float, str | None]:
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


def _resolve_volt_unit(unit: str | None, *, required: bool = True) -> tuple[float, str]:
    """(factor-to-volts, display symbol) for a voltage unit.

    We do not default the unit. When `required` and none is supplied, raise
    and ask the user for one. A supplied unit that is not a voltage unit
    always raises. (`required=False` is used internally when reading a unit
    declared inside the file, where a sensible fallback is acceptable.)
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


def _seconds_to_clock(seconds: float) -> str:
    """Seconds-since-midnight -> 'H:MM:SS' (or 'H:MM:SS.mmm' with a fraction)."""
    s = float(seconds) % (24 * 3600.0)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s - h * 3600 - m * 60
    if abs(sec - round(sec)) < 1e-6:
        return f"{h:d}:{m:02d}:{int(round(sec)):02d}"
    return f"{h:d}:{m:02d}:{sec:06.3f}"


def _is_clock_string(value: "object") -> bool:
    return isinstance(value, str) and ":" in value


def _clock_to_seconds(value: str) -> float:
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


def _to_seconds(value: "object", unit_factor: float, recording_start_clock_s: float | None,
                *, field: str) -> float:
    """Convert one time input to seconds *from the start of the recording*.

    Numeric inputs are scaled by `unit_factor`. Clock strings ("HH:MM:SS")
    are interpreted as a wall-clock time-of-day and offset against the
    recording's own start-of-day clock (rolling to the next day if the
    clock time is earlier than the recording start).
    """
    if _is_clock_string(value):
        if recording_start_clock_s is None:
            raise RuntimeError(
                f"{field} was given as a clock time ({value!r}) but the "
                "recording has no start timestamp to anchor it to."
            )
        offset = _clock_to_seconds(value) - recording_start_clock_s
        if offset < 0:
            offset += 24 * 3600.0  # window is on the day after the start
        return offset
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
        v = float(y_range)
        if v <= 0:
            raise RuntimeError(
                f"y-axis range shorthand must be a positive number; got {v}."
            )
        return -v, v
    # Explicit [min, max] pair.
    if isinstance(y_range, (list, tuple)) and len(y_range) == 2:
        try:
            lo, hi = float(y_range[0]), float(y_range[1])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"y-axis range {y_range!r} has non-numeric bounds.") from exc
        if not lo < hi:
            raise RuntimeError(
                f"y-axis range min ({lo}) must be strictly less than max ({hi})."
            )
        return lo, hi
    raise RuntimeError(
        f"y-axis range {y_range!r} must be a single positive number "
        "(± shorthand) or a [min, max] pair."
    )


def _resolve_window(start_s: float, end_time_s: float | None, duration_s: float | None,
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


############# READER (EDF) ##################
# The reader exposes a tiny interface to render():
#   _probe_edf(path) -> (channels, fs_of(channel), duration_s_of(channel),
#                        recording_start_clock_s or None)
#   _read_edf(path, channel, start_s, end_s) -> (t_seconds, y_volts, fs)
# Data is returned in SI volts; render() converts to the requested y_unit.

def _probe_edf(path: str):
    import pyedflib

    reader = pyedflib.EdfReader(path)
    try:
        labels = list(reader.getSignalLabels())

        def fs_of(channel: str) -> float:
            return float(reader.getSampleFrequency(_edf_index(labels, channel)))

        def duration_of(channel: str) -> float:
            idx = _edf_index(labels, channel)
            return reader.getNSamples()[idx] / float(reader.getSampleFrequency(idx))

        start_clock_s = None
        try:
            dt = reader.getStartdatetime()
            start_clock_s = dt.hour * 3600.0 + dt.minute * 60.0 + dt.second + dt.microsecond / 1e6
        except Exception:  # noqa: BLE001 — a missing/parse-broken start date is non-fatal
            start_clock_s = None

        return labels, fs_of, duration_of, start_clock_s
    finally:
        reader._close()


def _edf_index(labels: list[str], channel: str) -> int:
    """Case-insensitive exact match of a channel label; raise if absent."""
    wanted = str(channel).strip().lower()
    for i, lab in enumerate(labels):
        if lab.strip().lower() == wanted:
            return i
    raise RuntimeError(
        f"Channel {channel!r} is not in the file. Available channels: "
        + ", ".join(labels)
    )


def _read_edf(path: str, channel: str, start_s: float, end_s: float,
              channel2: str | None = None):
    """Read one channel's window in SI volts.

    For a montage (`channel2` given) return the bipolar derivation
    channel - channel2, sample-by-sample. Both channels are read over the
    same [start_sample, n) span; render() has already verified they share a
    sampling rate, so the two slices line up. Lengths are clipped to the
    shorter of the two as a defensive measure.
    """
    import numpy as np
    import pyedflib

    reader = pyedflib.EdfReader(path)
    try:
        labels = list(reader.getSignalLabels())

        def read_volts(ch: str):
            idx = _edf_index(labels, ch)
            fs = float(reader.getSampleFrequency(idx))
            start_sample = int(round(start_s * fs))
            n = int(round((end_s - start_s) * fs))
            # pyedflib returns physical values already scaled to the signal's
            # declared physical dimension (e.g. 'uV'); convert that to volts.
            y_phys = reader.readSignal(idx, start=start_sample, n=n)
            unit_factor, _ = _resolve_volt_unit(reader.getPhysicalDimension(idx), required=False)
            return fs, start_sample, np.asarray(y_phys, dtype=float) * unit_factor

        fs, start_sample, y_volts = read_volts(channel)
        if channel2 is not None:
            _, _, y_ref = read_volts(channel2)
            m = min(y_volts.size, y_ref.size)
            y_volts = y_volts[:m] - y_ref[:m]

        t_s = (start_sample + np.arange(y_volts.size)) / fs
        return t_s, y_volts, fs
    finally:
        reader._close()


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
                     value like 973000000 shows in full rather than as
                     "9.73" with a shared "1e8" offset in the corner.
    """
    from matplotlib.ticker import FuncFormatter, ScalarFormatter

    if signal.x_tick_style == "clock":
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: _seconds_to_clock(v)))
        return
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(fmt)


def _plot_timeseries(signal: Signal, y_limits: tuple[float, float], output_path: str) -> None:
    """Draw the trace in the academic house style, labels from signal.meta."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Auto physical size. Width tracks the window length (longer clip ->
    # wider axes) and height tracks the y-range magnitude, both clamped so
    # extreme inputs stay printable.
    span_x = float(signal.t[-1] - signal.t[0]) if signal.t.size > 1 else 1.0
    fig_w = max(6.0, min(18.0, 6.0 + span_x / (span_x + 1.0) * 8.0))
    span_y = abs(y_limits[1] - y_limits[0])
    fig_h = max(3.0, min(10.0, 3.0 + span_y / (span_y + 1.0) * 3.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.plot(signal.t, signal.y, color="#2a6c97", linewidth=0.8)

    ax.set_xlim(signal.t[0], signal.t[-1])
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
) -> None:
    """Render a raw timeseries clip from an EDF file.

    With `channel` alone this is a single-channel trace. Supplying `channel2`
    turns it into a bipolar montage — the trace becomes channel - channel2,
    titled "... for montage <channel>-<channel2>".

    See the module docstring for the full input / error contract. Written
    as a thin pipeline (probe -> validate -> read -> build Signal -> plot)
    so later processing stages slot in between the read and the plot
    without changing the plotting code.
    """
    import os

    ####### VALIDATE FILE + FORMAT
    ext = os.path.splitext(target_file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise RuntimeError(
            f"{target_file_path!r} is not an EDF file (extension {ext!r}). "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    if not os.path.isfile(target_file_path):
        raise RuntimeError(f"Input file does not exist: {target_file_path!r}")

    ####### VALIDATE REQUIRED SCALAR INPUTS
    if channel is None or str(channel).strip() == "":
        raise RuntimeError("A channel name is required (e.g. 'F8').")
    if start_time is None or (isinstance(start_time, str) and start_time.strip() == ""):
        raise RuntimeError("A start time is required.")

    ####### DECIDE X-AXIS DISPLAY MODE + RESOLVE UNITS (no silent defaults)
    # The x-axis mirrors the user's input format: clock-string times render
    # as H:M:S; numeric times render in their own unit. `clock_mode` keys off
    # the start time (what anchors the window).
    def _is_numeric_time(v: "object") -> bool:
        return (v is not None and not _is_clock_string(v)
                and not (isinstance(v, str) and v.strip() == ""))

    clock_mode = _is_clock_string(start_time)
    # A time unit is only required when a time is given as a plain number;
    # clock strings ("HH:MM:SS") carry their own unit and need none.
    has_numeric_time = any(_is_numeric_time(v) for v in (start_time, end_time, duration))
    time_factor, time_symbol = _resolve_time_unit(time_unit, required=has_numeric_time)
    volt_factor, volt_symbol = _resolve_volt_unit(y_unit, required=True)
    y_limits = _parse_y_range(y_range)

    ####### PROBE THE RECORDING (channels, rate, duration, start clock)
    try:
        _channels, fs_of, duration_of, start_clock_s = _probe_edf(target_file_path)
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 — any reader failure = unreadable file
        raise RuntimeError(f"Could not read {target_file_path!r} as {ext} data: {exc}") from exc

    # Montage = a bipolar A-B derivation, active when a second channel is given.
    montage = channel2 is not None and str(channel2).strip() != ""
    if montage and str(channel).strip().lower() == str(channel2).strip().lower():
        raise RuntimeError(
            f"A montage needs two different channels, but both are {channel!r}."
        )

    # Validate EVERY channel BEFORE reading any samples. fs_of / duration_of
    # resolve each name (case-insensitive) and raise — listing the available
    # channels — if a name is absent. For a montage we additionally require a
    # matching sampling rate and a window that fits BOTH channels, so a bad
    # request fails fast with a readable error, just like the single-channel path.
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

    ####### RESOLVE + VALIDATE THE TIME WINDOW (seconds from recording start)
    start_s = _to_seconds(start_time, time_factor, start_clock_s, field="start time")
    end_time_s = (
        _to_seconds(end_time, time_factor, start_clock_s, field="end time")
        if end_time is not None and not (isinstance(end_time, str) and end_time.strip() == "")
        else None
    )
    duration_s = (
        _to_seconds(duration, time_factor, None, field="duration")
        if duration is not None and not (isinstance(duration, str) and duration.strip() == "")
        else None
    )
    start_s, end_s = _resolve_window(start_s, end_time_s, duration_s, channel_duration_s)

    # Reject windows too short to be a signal (a handful of samples).
    expected_samples = int(round((end_s - start_s) * fs))
    if expected_samples < MIN_SAMPLES:
        raise RuntimeError(
            f"window spans only {expected_samples} sample(s) at {fs:g} Hz — "
            f"need at least {MIN_SAMPLES}. Widen the window."
        )

    ####### READ THE WINDOW (returns SI volts; montage -> channel - channel2)
    t_s, y_volts, fs = _read_edf(
        target_file_path, channel, start_s, end_s, channel2 if montage else None
    )
    if y_volts.size == 0:
        raise RuntimeError("The requested window contained no samples.")

    ####### BUILD THE RAW-STAGE SIGNAL (convert to requested display units)
    # X values + label follow the input format. Clock input -> keep time as
    # seconds-since-midnight so ticks render as wall-clock H:M:S; numeric
    # input -> scale seconds into the requested unit and label with it.
    if clock_mode:
        t_display = t_s + start_clock_s     # seconds from start -> since midnight
        x_unit = "h:m:s"
        x_tick_style = "clock"
    else:
        t_display = t_s / time_factor       # seconds -> requested time unit
        x_unit = time_symbol
        x_tick_style = "numeric"

    signal = Signal(
        t=t_display,
        y=y_volts / volt_factor,           # volts   -> requested voltage unit
        fs=fs,
        channel=f"{channel}-{channel2}" if montage else str(channel),
        channel_kind="montage" if montage else "single_channel",
        x_domain="time",
        x_unit=x_unit,
        y_domain="amplitude",
        y_unit=volt_symbol,
        x_tick_style=x_tick_style,
    )
    # (Future processing stages would transform `signal` here, updating its
    #  *_domain / *_unit metadata, before it reaches the plotter.)

    ####### PLOT
    _plot_timeseries(signal, y_limits, output_path)
