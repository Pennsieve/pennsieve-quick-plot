"""
readers.nwb_to_signal — NWB (ElectricalSeries) recording -> raw Signal.

    load_signal(path, params) -> Signal

in the same four steps as the EDF reader, each its own function:

  _read_header            open the file once, metadata only: channel names,
                          rate, length, start time, the voltage scaling
  _check_against_header   channels exist; the window (readers/window.py);
                          for a recording with gaps, that the window does
                          not cross one; voltage unit (user's, else µV)
  _read_channels          read the window for the channel (and channel2),
                          montage = channel - channel2, in volts
  _build_signal           convert to display voltage, package as a Signal

What the file looks like
------------------------
An NWB file is HDF5. This reader uses h5py directly rather than pynwb: the
compute node's shared layer carries h5py already, pynwb would add ~15 s of
import time to every cold start, and the handful of datasets we need have
fixed paths in the NWB schema:

  /acquisition/<series>/data              [samples x channels], raw values
      attrs conversion, offset             volts = data * conversion
  /acquisition/<series>/channel_conversion            * channel_conversion[ch] + offset
  /acquisition/<series>/starting_time     attr rate      continuous recording
  /acquisition/<series>/timestamps        per sample     recording with gaps
  /acquisition/<series>/electrodes        rows of the electrodes table, one
                                          per data column
  /general/extracellular_ephys/electrodes/channel_name   the channel names
  /session_start_time                     ISO 8601, with a UTC offset

processor-plot-prep writes exactly this from a MEF 2.0 recording: one
ElectricalSeries named "ElectricalSeries", a `channel_name` column, `rate`
when the recording is continuous and per-sample `timestamps` when it has
gaps. Any NWB file with one ElectricalSeries reads the same way.

Gaps
----
When the samples carry timestamps, the recording may have holes (the
amplifier was off, a segment is missing). A window that crosses one is
refused with WINDOW_SPANS_GAP, naming the hole, rather than drawn with the
two sides stitched together as if they were continuous. Timestamps are
searched, not loaded: a multi-week recording has billions of samples and
the array would not fit in memory, so the window's sample range is found
by bisection on the dataset and only that range is read.

Time is returned in seconds from the recording start; voltage is converted
here (to the user's unit, or µV, the iEEG convention — NWB stores volts and
declares no display unit). Heavy imports (numpy, h5py) are lazy inside the
functions so importing at startup stays cheap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

from processor.errors import (
    PlotDataUnavailableError,
    PlotErrorCode,
    PlotErrorStage,
    PlotInvalidInputError,
)
from processor.readers.window import resolve_window
from processor.tools.ts_dsp import (
    Signal,
    resolve_volt_unit,
    seconds_since_midnight,
    seconds_to_clock,
)

# NWB fixes ElectricalSeries.unit to volts; a file declares no display unit.
_DEFAULT_VOLT_UNIT = "uV"
# Electrode-table columns tried, in order, for a channel's name.
_NAME_COLUMNS = ("channel_name", "label")
# Two samples further apart than this (in sampling periods) are a gap.
_GAP_TOLERANCE_PERIODS = 1.5


############# HEADER ##################
@dataclass(frozen=True)
class _Header:
    series: str              # HDF5 path of the ElectricalSeries read
    labels: tuple            # channel names, in data-column order
    fs: float                # sampling rate (Hz); one time axis for every channel
    n_samples: int
    first_s: float           # time of the first sample, seconds from recording start
    last_s: float            # time of the last sample
    start: "datetime | None"  # recording start (session_start_time), timezone-aware
    gapped: bool             # samples carry explicit timestamps (may have holes)
    conversion: float        # data -> volts, with the two below
    offset: float
    channel_conversion: tuple   # per channel; empty when the file has none (=1)

    @property
    def duration_s(self) -> float:
        """Length of the recording from its start, as a window must fit in."""
        return self.last_s + 1.0 / self.fs

    def index(self, channel: str) -> int:
        """Resolve a user-typed channel name to its data column
        (case-insensitive exact match); raise, listing the channels, if absent."""
        wanted = str(channel).strip().lower()
        for i, lab in enumerate(self.labels):
            if lab.strip().lower() == wanted:
                return i
        raise PlotInvalidInputError(
            PlotErrorCode.UNKNOWN_CHANNEL,
            f"Channel {channel!r} is not in the file. Available channels: "
            + ", ".join(self.labels),
            error_facts={"channel": str(channel), "available_channels": list(self.labels)},
            source_stage=PlotErrorStage.READER,
        )


def _find_series(f) -> str:
    """The HDF5 path of the one ElectricalSeries to read."""
    acquisition = f.get("acquisition")
    candidates = []
    if acquisition is not None:
        for name, obj in acquisition.items():
            if _text(obj.attrs.get("neurodata_type", "")) == "ElectricalSeries" and "data" in obj:
                candidates.append(name)
    if not candidates:
        raise PlotDataUnavailableError(
            PlotErrorCode.MISSING_REQUIRED_METADATA,
            "The NWB file has no ElectricalSeries under /acquisition, so there "
            "is no voltage recording to plot.",
            source_stage=PlotErrorStage.READER,
        )
    if len(candidates) > 1:
        if "ElectricalSeries" in candidates:      # the converter's name
            candidates = ["ElectricalSeries"]
        else:
            raise PlotDataUnavailableError(
                PlotErrorCode.MISSING_REQUIRED_METADATA,
                "The NWB file holds several ElectricalSeries ("
                + ", ".join(sorted(candidates)) + ") and this reader plots one; "
                "convert the recording with one series per file.",
                error_facts={"series": sorted(candidates)},
                source_stage=PlotErrorStage.READER,
            )
    return f"/acquisition/{candidates[0]}"


def _channel_names(f, series) -> tuple:
    """Channel names in data-column order, from the electrodes table rows
    the series points at (a `channel_name` or `label` column, else the row id)."""
    region = series["electrodes"]
    rows = [int(r) for r in region[()]]
    table = f[region.attrs["table"]] if "table" in region.attrs else f["/general/extracellular_ephys/electrodes"]
    for column in _NAME_COLUMNS:
        if column in table:
            values = table[column][()]
            return tuple(_text(values[r]) for r in rows)
    ids = table["id"][()]
    return tuple(str(int(ids[r])) for r in rows)


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _session_start(f) -> "datetime | None":
    if "session_start_time" not in f:
        return None
    try:
        moment = datetime.fromisoformat(_text(f["session_start_time"][()]))
    except ValueError:
        return None
    # NWB writes the start with its UTC offset; show the clock as UTC so the
    # axis label can say which clock the ticks are on.
    return moment.astimezone(timezone.utc) if moment.tzinfo is not None else moment


def _rate_from_timestamps(timestamps) -> float:
    """Sampling rate of a gapped recording: the typical spacing of its first
    samples (a median, so a gap right at the start does not skew it)."""
    import numpy as np

    head = timestamps[: min(len(timestamps), 1024)]
    if len(head) < 2:
        raise PlotDataUnavailableError(
            PlotErrorCode.INSUFFICIENT_SAMPLES,
            "The recording has fewer than two samples.",
            source_stage=PlotErrorStage.READER,
        )
    period = float(np.median(np.diff(head)))
    if not period > 0:
        raise PlotDataUnavailableError(
            PlotErrorCode.MISSING_REQUIRED_METADATA,
            "The recording's timestamps do not increase, so its sampling rate "
            "cannot be determined.",
            source_stage=PlotErrorStage.READER,
        )
    return 1.0 / period


def _read_header(path: str) -> _Header:
    """Metadata only — no sample data is read (two timestamps and the first
    few for a gapped recording's rate)."""
    import h5py

    with h5py.File(path, "r") as f:
        series_path = _find_series(f)
        series = f[series_path]
        data = series["data"]
        n_samples = int(data.shape[0])
        if n_samples == 0:
            raise PlotDataUnavailableError(
                PlotErrorCode.EMPTY_DATA, "The recording contains no samples.",
                source_stage=PlotErrorStage.READER)
        labels = _channel_names(f, series)
        if "timestamps" in series:
            timestamps = series["timestamps"]
            fs = _rate_from_timestamps(timestamps)
            first_s, last_s = float(timestamps[0]), float(timestamps[-1])
            gapped = True
        elif "starting_time" in series and "rate" in series["starting_time"].attrs:
            fs = float(series["starting_time"].attrs["rate"])
            first_s = float(series["starting_time"][()])
            last_s = first_s + (n_samples - 1) / fs
            gapped = False
        else:
            raise PlotDataUnavailableError(
                PlotErrorCode.MISSING_REQUIRED_METADATA,
                "The ElectricalSeries has neither a sampling rate nor timestamps, "
                "so its samples cannot be placed in time.",
                source_stage=PlotErrorStage.READER,
            )
        channel_conversion = tuple(float(c) for c in series["channel_conversion"][()]) \
            if "channel_conversion" in series else ()
        return _Header(
            series=series_path, labels=labels, fs=fs, n_samples=n_samples,
            first_s=first_s, last_s=last_s, start=_session_start(f), gapped=gapped,
            conversion=float(data.attrs.get("conversion", 1.0)),
            offset=float(data.attrs.get("offset", 0.0)),
            channel_conversion=channel_conversion,
        )


############# REQUEST vs HEADER ##################
@dataclass(frozen=True)
class _Window:
    channel: int             # data column of the primary channel
    channel2: "int | None"   # data column of the reference channel (montage)
    start_s: float           # seconds from recording start
    end_s: float
    volt_factor: float       # display unit -> volts
    volt_symbol: str


def _check_against_header(params, header: _Header) -> _Window:
    channel = header.index(params.channel)
    channel2 = header.index(params.channel2) if params.montage else None
    # One time axis for every channel of an ElectricalSeries, so a montage
    # never has a rate mismatch here.
    start_s, end_s = resolve_window(params, recording_start=header.start,
                                    recording_s=header.duration_s, fs=header.fs)
    if start_s < header.first_s:
        # The series may begin after the session start (starting_time > 0).
        raise PlotInvalidInputError(
            PlotErrorCode.INVALID_TIME_WINDOW,
            f"start time ({start_s:g}s) is before the first sample, which is at "
            f"{header.first_s:g}s from the recording start.",
            error_facts={"start_s": start_s, "end_s": end_s, "recording_s": header.duration_s,
                         "first_sample_s": header.first_s},
            source_stage=PlotErrorStage.READER,
        )
    if params.volt is not None:
        volt_factor, volt_symbol = params.volt
    else:
        volt_factor, volt_symbol = resolve_volt_unit(_DEFAULT_VOLT_UNIT)
    return _Window(channel, channel2, start_s, end_s, volt_factor, volt_symbol)


############# SAMPLES ##################
def _first_at_or_after(timestamps, value: float) -> int:
    """Index of the first timestamp >= value, by bisection on the dataset
    (a few dozen scalar reads, however long the recording)."""
    lo, hi = 0, len(timestamps)
    while lo < hi:
        mid = (lo + hi) // 2
        if float(timestamps[mid]) < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _gapped_range(timestamps, window: _Window, header: _Header, params):
    """(i0, i1, t) for a recording with timestamps: the sample range inside
    the window and the samples' times. Raises if the window crosses a gap."""
    import numpy as np

    i0 = _first_at_or_after(timestamps, window.start_s)
    i1 = _first_at_or_after(timestamps, window.end_s)
    t = np.asarray(timestamps[i0:i1], dtype=float)
    tolerance = _GAP_TOLERANCE_PERIODS / header.fs

    # A hole can start before the window (no sample near its start), lie
    # inside it (two consecutive samples far apart), or run past its end.
    if t.size == 0 or t[0] - window.start_s > tolerance:
        before = float(timestamps[i0 - 1]) if i0 > 0 else window.start_s
        after = float(t[0]) if t.size else (float(timestamps[i1]) if i1 < len(timestamps) else window.end_s)
        raise _spans_gap(window, header, params, gap_start_s=before, gap_end_s=after)
    steps = np.diff(t)
    if steps.size and float(steps.max()) > tolerance:
        k = int(np.argmax(steps))
        raise _spans_gap(window, header, params, gap_start_s=float(t[k]), gap_end_s=float(t[k + 1]))
    if window.end_s - t[-1] > tolerance:
        after = float(timestamps[i1]) if i1 < len(timestamps) else window.end_s
        raise _spans_gap(window, header, params, gap_start_s=float(t[-1]), gap_end_s=after)
    return i0, i1, t


def _spans_gap(window: _Window, header: _Header, params, *, gap_start_s: float, gap_end_s: float):
    """The window is well-formed but the recording has no data for part of
    it. Says where the hole is, in the form the user typed the times in."""
    def describe(seconds: float) -> str:
        if params.clock_mode and header.start is not None:
            return seconds_to_clock(seconds_since_midnight(header.start) + seconds)
        return f"{seconds / params.time_factor:g} {params.time_symbol or 's'}"

    return PlotInvalidInputError(
        PlotErrorCode.WINDOW_SPANS_GAP,
        f"The requested window {describe(window.start_s)}–{describe(window.end_s)} "
        f"crosses a gap in the recording (no data from {describe(gap_start_s)} "
        f"to {describe(gap_end_s)}). Choose a different window.",
        user_next_step=f"Choose a window that ends before {describe(gap_start_s)} "
                       f"or starts after {describe(gap_end_s)}.",
        error_facts={"start_s": window.start_s, "end_s": window.end_s,
                     "gap_start_s": gap_start_s, "gap_end_s": gap_end_s},
        source_stage=PlotErrorStage.READER,
    )


def _read_channels(path: str, window: _Window, header: _Header, params):
    """(t_seconds_from_start, y_volts) for the window.

    For a montage the result is channel - channel2, sample by sample (both
    columns share the series' time axis, so the lengths always match).
    """
    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        series = f[header.series]
        if header.gapped:
            i0, i1, t_s = _gapped_range(series["timestamps"], window, header, params)
        else:
            i0 = int(round((window.start_s - header.first_s) * header.fs))
            i1 = i0 + int(round((window.end_s - window.start_s) * header.fs))
            i1 = min(i1, header.n_samples)
            t_s = header.first_s + (i0 + np.arange(i1 - i0)) / header.fs

        def read_volts(column: int):
            raw = np.asarray(series["data"][i0:i1, column], dtype=float)
            per_channel = header.channel_conversion[column] if header.channel_conversion else 1.0
            return raw * header.conversion * per_channel + header.offset

        y_volts = read_volts(window.channel)
        if window.channel2 is not None:
            y_volts = y_volts - read_volts(window.channel2)

    if y_volts.size == 0:
        raise PlotDataUnavailableError(
            PlotErrorCode.EMPTY_DATA, "The requested window contained no samples.",
            source_stage=PlotErrorStage.READER)
    return t_s, y_volts


############# SIGNAL ##################
def _build_signal(t_s, y_volts, window: _Window, header: _Header, params) -> Signal:
    montage = window.channel2 is not None
    return Signal(
        t=t_s,                                   # seconds from recording start
        y=y_volts / window.volt_factor,          # volts -> display voltage unit
        fs=header.fs,
        channel=f"{params.channel}-{params.channel2}" if montage else str(params.channel),
        channel_kind="montage" if montage else "single_channel",
        x_domain="time",
        x_unit="s",
        y_domain="amplitude",
        y_unit=window.volt_symbol,
        recording_start=header.start,
    )


############# ENTRY POINT ##################
def load_signal(path: str, params) -> Signal:
    try:
        header = _read_header(path)
    except (RuntimeError, ImportError, MemoryError):
        # Already a PlotError, or a platform problem (missing library, out of
        # memory) that main.py's wrap_unexpected_exception() knows how to
        # name. Not the file's fault, so not "unreadable".
        raise
    except Exception as exc:  # noqa: BLE001 — any other reader failure = unreadable file
        raise PlotDataUnavailableError(
            PlotErrorCode.UNREADABLE_FILE,
            f"Could not read {path!r} as NWB data: {exc}",
            error_facts={"file": os.path.basename(path), "cause": exc.__class__.__name__},
            source_stage=PlotErrorStage.READER,
        ) from exc
    window = _check_against_header(params, header)
    t_s, y_volts = _read_channels(path, window, header, params)
    return _build_signal(t_s, y_volts, window, header, params)
