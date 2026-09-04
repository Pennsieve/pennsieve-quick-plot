"""
ts_dsp.io — raw acquisition for voltage timeseries.

This is the always-on plumbing every EDF template shares to turn a file +
a requested window into a raw `Signal`: probe metadata, read + crop to the
window ("clip"), convert to the requested display units, and build the raw
Signal. None of this is a user-composable DSP step, so nothing here is
registered in the tool registry — the DSP pipeline (filters, features, …)
runs *after* a raw Signal exists.

Heavy imports (numpy, pyedflib) are lazy inside the functions so importing
this module at startup stays cheap.
"""

from __future__ import annotations

from .signal import Signal, resolve_volt_unit


############# CHANNEL LOOKUP ##################
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


############# PROBE (metadata only) ##################
def probe_edf(path: str):
    """(labels, fs_of(channel), duration_s_of(channel), unit_of(channel),
    start_clock_s | None).

    Cheap metadata pass: no sample data is read. `fs_of` / `duration_of` /
    `unit_of` resolve the channel name (case-insensitive) and raise — listing
    the available channels — if it is absent. `unit_of` is the channel's
    declared physical dimension, verbatim (may be blank). `start_clock_s` is
    the recording's start-of-day in seconds-since-midnight, or None if the
    file has no usable start timestamp.
    """
    import pyedflib

    reader = pyedflib.EdfReader(path)
    try:
        labels = list(reader.getSignalLabels())

        def fs_of(channel: str) -> float:
            return float(reader.getSampleFrequency(_edf_index(labels, channel)))

        def duration_of(channel: str) -> float:
            idx = _edf_index(labels, channel)
            return reader.getNSamples()[idx] / float(reader.getSampleFrequency(idx))

        def unit_of(channel: str) -> str:
            return str(reader.getPhysicalDimension(_edf_index(labels, channel)))

        start_clock_s = None
        try:
            # so start_clock_s is None if the file has no start timestamp, 
            # otherwise it's the seconds-since-midnight of the start timestamp.
            dt = reader.getStartdatetime()
            start_clock_s = dt.hour * 3600.0 + dt.minute * 60.0 + dt.second + dt.microsecond / 1e6
        except Exception:  # noqa: BLE001 — a missing/parse-broken start date is non-fatal
            start_clock_s = None

        return labels, fs_of, duration_of, unit_of, start_clock_s
    finally:
        reader._close()


############# READ + CROP (the "clip") ##################
def read_edf(path: str, channel: str, start_s: float, end_s: float,
             channel2: str | None = None):
    """Read one channel's window in SI volts, returning (t_seconds, y_volts, fs).

    The [start_s, end_s) crop *is* the time-window extraction ("clip"). For a
    montage (`channel2` given) the result is the bipolar derivation
    channel - channel2, sample-by-sample. Both channels are read over the same
    span; the caller has verified they share a sampling rate, so the two
    slices line up. Lengths are clipped to the shorter of the two defensively.
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
            dim = reader.getPhysicalDimension(idx)
            try:
                unit_factor, _ = resolve_volt_unit(dim, required=False)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Channel {ch!r} declares its voltage unit as {dim!r}, "
                    "which is not a recognized voltage unit, so the recording "
                    "cannot be read reliably."
                ) from exc
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


############# BUILD THE RAW SIGNAL (shared by every EDF template) ##################
def read_signal(path: str, channel: str, start_s: float, end_s: float, *,
                channel2: str | None = None, montage: bool = False,
                time_factor: float = 1.0, time_symbol: str | None = "s",
                volt_factor: float = 1e-6, volt_symbol: str = "µV",
                clock_mode: bool = False, start_clock_s: float | None = None) -> Signal:
    """Read + crop + unit-convert an EDF window into a raw `Signal`.

    This is the one place the raw-clip is constructed, so every EDF template
    builds it identically. Units are converted from SI (seconds, volts) into
    the requested display units. The x-axis metadata mirrors the input format:
    clock input -> time kept as seconds-since-midnight (H:M:S ticks); numeric
    input -> time scaled into the requested unit.
    """
    t_s, y_volts, fs = read_edf(path, channel, start_s, end_s, channel2 if montage else None)
    if y_volts.size == 0:
        raise RuntimeError("The requested window contained no samples.")

    if clock_mode:
        t_display = t_s + start_clock_s      # seconds from start -> since midnight
        x_unit = "h:m:s"
        x_tick_style = "clock"
    else:
        t_display = t_s / time_factor        # seconds -> requested time unit
        x_unit = time_symbol
        x_tick_style = "numeric"

    return Signal(
        t=t_display,
        y=y_volts / volt_factor,             # volts -> requested voltage unit
        fs=fs,
        channel=f"{channel}-{channel2}" if montage else str(channel),
        channel_kind="montage" if montage else "single_channel",
        x_domain="time",
        x_unit=x_unit,
        y_domain="amplitude",
        y_unit=volt_symbol,
        x_tick_style=x_tick_style,
    )
