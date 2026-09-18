"""
readers.edf_to_signal — EDF / EDF+ recording -> raw Signal.

    load_signal(path, params) -> Signal

in four steps, each its own function:

  _read_header            open the file once, metadata only (no samples)
  _check_against_header   everything about the request that can only be
                          checked with the header in hand: channels exist,
                          montage channels share a sampling rate, clock-string
                          times resolved against the recording start, window
                          inside the recording and within the clip limits,
                          voltage unit (user's or the header's)
  _read_channels          read the window for the channel (and channel2),
                          montage = channel - channel2, in volts
  _build_signal           convert to display voltage, package as a Signal

Time is returned in seconds from the start of the recording; the plot
stage converts to the user's display unit or to H:M:S ticks. Voltage is
converted here (to the user's unit, or the header's) so DSP tools can label
derived units correctly (µV -> µV²).

Rules that need only the user's inputs live in the template's _validate();
this module never repeats them. Heavy imports (numpy, pyedflib) are lazy
inside the functions so importing at startup stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

from processor.tools.ts_dsp import Signal, resolve_volt_unit


############# HEADER ##################
@dataclass(frozen=True)
class _Header:
    labels: tuple            # channel labels, in file order
    fs: dict                 # label -> sampling rate (Hz)
    duration_s: dict         # label -> recording length for that channel (s)
    unit: dict               # label -> declared physical dimension, verbatim
    start_clock_s: "float | None"   # recording start as seconds since midnight

    def index(self, channel: str) -> str:
        """Resolve a user-typed channel name to the file's label
        (case-insensitive exact match); raise, listing the channels, if absent."""
        wanted = str(channel).strip().lower()
        for lab in self.labels:
            if lab.strip().lower() == wanted:
                return lab
        raise RuntimeError(
            f"Channel {channel!r} is not in the file. Available channels: "
            + ", ".join(self.labels)
        )


def _read_header(path: str) -> _Header:
    """Metadata only — no sample data is read."""
    import pyedflib

    reader = pyedflib.EdfReader(path)
    try:
        labels = tuple(reader.getSignalLabels())
        n_samples = reader.getNSamples()
        fs = {lab: float(reader.getSampleFrequency(i)) for i, lab in enumerate(labels)}
        duration_s = {lab: n_samples[i] / fs[lab] for i, lab in enumerate(labels)}
        unit = {lab: str(reader.getPhysicalDimension(i)) for i, lab in enumerate(labels)}
        try:
            dt = reader.getStartdatetime()
            start_clock_s = dt.hour * 3600.0 + dt.minute * 60.0 + dt.second + dt.microsecond / 1e6
        except Exception:  # noqa: BLE001 — a missing/parse-broken start date is non-fatal
            start_clock_s = None
    finally:
        reader._close()
    return _Header(labels, fs, duration_s, unit, start_clock_s)


############# REQUEST vs HEADER ##################
@dataclass(frozen=True)
class _Window:
    channel: str             # file label for the primary channel
    channel2: "str | None"   # file label for the reference channel (montage)
    start_s: float           # seconds from recording start
    end_s: float
    fs: float
    volt_factor: float       # display unit -> volts
    volt_symbol: str


def _to_seconds_from_start(point, header: _Header, *, field: str) -> float:
    """A TimePoint from the template -> seconds from the recording start.

    Numeric inputs were already converted to seconds by _parse. Clock inputs
    are seconds-since-midnight and are anchored to the recording's own
    start-of-day clock (rolling to the next day if earlier than the start).
    """
    if not point.is_clock:
        return float(point.seconds)
    if header.start_clock_s is None:
        raise RuntimeError(
            f"{field} was given as a clock time but the recording has no "
            "start timestamp to anchor it to."
        )
    offset = float(point.seconds) - header.start_clock_s
    if offset < 0:
        offset += 24 * 3600.0
    return offset


def _check_against_header(params, header: _Header) -> _Window:
    from processor.readers import MAX_DURATION_S, MIN_SAMPLES

    # Channels exist (case-insensitive); montage channels share a rate.
    channel = header.index(params.channel)
    channel2 = header.index(params.channel2) if params.montage else None
    fs = header.fs[channel]
    if channel2 is not None and abs(fs - header.fs[channel2]) > 1e-6:
        raise RuntimeError(
            f"Montage {params.channel}-{params.channel2} needs both channels at "
            f"the same sampling rate, but {params.channel} is {fs:g} Hz and "
            f"{params.channel2} is {header.fs[channel2]:g} Hz."
        )
    # The window must fit within BOTH channels -> validate against the shorter.
    recording_s = min(header.duration_s[c] for c in (channel, channel2) if c is not None)

    # Window, in seconds from the recording start.
    start_s = _to_seconds_from_start(params.start, header, field="start time")
    end_s = _to_seconds_from_start(params.end, header, field="end time") if params.end else None
    if end_s is not None and params.duration_s is not None:
        implied_end = start_s + params.duration_s
        if abs(implied_end - end_s) > 1e-6:
            raise RuntimeError(
                f"end time and duration disagree: end implies {end_s:g}s but "
                f"start+duration implies {implied_end:g}s. Provide one."
            )
    if end_s is None:
        end_s = start_s + params.duration_s
    if start_s < 0:
        raise RuntimeError(f"start time ({start_s:g}s) is before the recording start.")
    if start_s >= recording_s:
        raise RuntimeError(
            f"start time ({start_s:g}s) is at or past the end of the recording "
            f"({recording_s:g}s)."
        )
    if end_s <= start_s:
        raise RuntimeError(
            f"end time ({end_s:g}s) is not after the start time ({start_s:g}s)."
        )
    if end_s > recording_s:
        raise RuntimeError(
            f"end time ({end_s:g}s) is past the end of the recording ({recording_s:g}s)."
        )
    if (end_s - start_s) > MAX_DURATION_S:
        raise RuntimeError(
            f"window is {end_s - start_s:g}s long, over the {MAX_DURATION_S:g}s maximum."
        )
    expected_samples = int(round((end_s - start_s) * fs))
    if expected_samples < MIN_SAMPLES:
        raise RuntimeError(
            f"window spans only {expected_samples} sample(s) at {fs:g} Hz — "
            f"need at least {MIN_SAMPLES}. Widen the window."
        )

    # Display voltage unit: the user's if given, else the header's for the
    # primary channel (a blank declaration falls back to µV; an unrecognisable
    # one means the recording cannot be read reliably).
    if params.volt is not None:
        volt_factor, volt_symbol = params.volt
    else:
        volt_factor, volt_symbol = _header_volt_unit(channel, header)

    return _Window(channel, channel2, start_s, end_s, fs, volt_factor, volt_symbol)


def _header_volt_unit(channel: str, header: _Header) -> tuple[float, str]:
    declared = header.unit[channel]
    try:
        return resolve_volt_unit(declared, required=False)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Channel {channel!r} declares its voltage unit as {declared!r}, "
            "which is not a recognized voltage unit, so the recording cannot "
            "be read reliably."
        ) from exc


############# SAMPLES ##################
def _read_channels(path: str, window: _Window, header: _Header):
    """(t_seconds_from_start, y_volts) for the window.

    For a montage the result is channel - channel2, sample by sample (both
    read over the same span; lengths clipped to the shorter defensively).
    """
    import numpy as np
    import pyedflib

    reader = pyedflib.EdfReader(path)
    try:
        labels = list(reader.getSignalLabels())
        start_sample = int(round(window.start_s * window.fs))
        n = int(round((window.end_s - window.start_s) * window.fs))

        def read_volts(label: str):
            idx = labels.index(label)
            # pyedflib returns physical values already scaled to the channel's
            # declared dimension (e.g. 'uV'); convert that to volts.
            factor, _ = _header_volt_unit(label, header)
            return np.asarray(reader.readSignal(idx, start=start_sample, n=n), dtype=float) * factor

        y_volts = read_volts(window.channel)
        if window.channel2 is not None:
            y_ref = read_volts(window.channel2)
            m = min(y_volts.size, y_ref.size)
            y_volts = y_volts[:m] - y_ref[:m]
    finally:
        reader._close()

    if y_volts.size == 0:
        raise RuntimeError("The requested window contained no samples.")
    t_s = (start_sample + np.arange(y_volts.size)) / window.fs
    return t_s, y_volts


############# SIGNAL ##################
def _build_signal(t_s, y_volts, window: _Window, header: _Header, params) -> Signal:
    montage = window.channel2 is not None
    return Signal(
        t=t_s,                                   # seconds from recording start
        y=y_volts / window.volt_factor,          # volts -> display voltage unit
        fs=window.fs,
        channel=f"{params.channel}-{params.channel2}" if montage else str(params.channel),
        channel_kind="montage" if montage else "single_channel",
        x_domain="time",
        x_unit="s",
        y_domain="amplitude",
        y_unit=window.volt_symbol,
        recording_start_clock_s=header.start_clock_s,
    )


############# ENTRY POINT ##################
def load_signal(path: str, params) -> Signal:
    try:
        header = _read_header(path)
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 — any reader failure = unreadable file
        raise RuntimeError(f"Could not read {path!r} as EDF data: {exc}") from exc
    window = _check_against_header(params, header)
    t_s, y_volts = _read_channels(path, window, header)
    return _build_signal(t_s, y_volts, window, header, params)
