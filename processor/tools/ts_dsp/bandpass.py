"""
ts_dsp.bandpass — time-domain frequency-selective filters.

Each filter takes a time-domain signal and returns a time-domain signal
(domains carried through unchanged), so filters chain freely among
themselves. scipy is imported lazily inside the function to keep module
import (which the registry does at startup) cheap.
"""

from __future__ import annotations

from dataclasses import replace

from .registry import dsp_tool, ParamSpec, ToolInputError


def _zero_phase_filter(signal, sos, tool_name, order):
    """Apply an SOS cascade with sosfiltfilt, guarding the minimum length.

    sosfiltfilt (zero-phase, forward+backward) needs a minimum length; the
    padlen arithmetic mirrors scipy's default for a SOS cascade so we can
    raise a specific ToolInputError instead of scipy's generic ValueError.
    """
    import numpy as np
    from scipy.signal import sosfiltfilt

    y = np.asarray(signal.y, dtype=float)
    ntaps = 2 * len(sos) + 1
    ntaps -= min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    padlen = 3 * int(ntaps)
    if y.size <= padlen:
        raise ToolInputError(
            f"{tool_name} needs more than {padlen} samples to filter at "
            f"order {order}; the window has only {y.size}."
        )
    return replace(signal, y=sosfiltfilt(sos, y))


@dsp_tool(
    "highpass_filter",
    requires={"x_domain": "time",        # cannot high-pass a frequency-domain signal
              "y_domain": "amplitude"},  # fs-based cutoffs are only truthful for the raw trace
    produces={},                         # stays a time-domain trace, same unit
    params=(
        ParamSpec("cutoff", required=True, description="high-pass cutoff frequency", unit="Hz"),
        ParamSpec("order", required=False, description="Butterworth filter order (default 4)"),
    ),
    description="Zero-phase Butterworth high-pass filter; attenuates content below `cutoff` Hz.",
)
def highpass_filter(signal, cutoff, order=4):
    from scipy.signal import butter

    fs = float(signal.fs)
    cutoff = float(cutoff)
    order = int(order)
    nyq = fs / 2.0
    if order < 1:
        raise ToolInputError(
            f"highpass_filter order {order} must be a positive integer."
        )
    if not (0.0 < cutoff < nyq):
        raise ToolInputError(
            f"highpass_filter cutoff {cutoff} Hz must be between 0 and the "
            f"Nyquist frequency ({nyq:g} Hz) for this {fs:g} Hz signal."
        )

    sos = butter(order, cutoff / nyq, btype="highpass", output="sos")
    return _zero_phase_filter(signal, sos, "highpass_filter", order)


@dsp_tool(
    "lowpass_filter",
    requires={"x_domain": "time",        # cannot low-pass a frequency-domain signal
              "y_domain": "amplitude"},  # fs-based cutoffs are only truthful for the raw trace
    produces={},                         # stays a time-domain trace, same unit
    params=(
        ParamSpec("cutoff", required=True, description="low-pass cutoff frequency", unit="Hz"),
        ParamSpec("order", required=False, description="Butterworth filter order (default 4)"),
    ),
    description="Zero-phase Butterworth low-pass filter; attenuates content above `cutoff` Hz.",
)
def lowpass_filter(signal, cutoff, order=4):
    from scipy.signal import butter

    fs = float(signal.fs)
    cutoff = float(cutoff)
    order = int(order)
    nyq = fs / 2.0
    if order < 1:
        raise ToolInputError(
            f"lowpass_filter order {order} must be a positive integer."
        )
    if not (0.0 < cutoff < nyq):
        raise ToolInputError(
            f"lowpass_filter cutoff {cutoff} Hz must be between 0 and the "
            f"Nyquist frequency ({nyq:g} Hz) for this {fs:g} Hz signal."
        )

    sos = butter(order, cutoff / nyq, btype="lowpass", output="sos")
    return _zero_phase_filter(signal, sos, "lowpass_filter", order)


@dsp_tool(
    "bandpass_filter",
    requires={"x_domain": "time",        # cannot band-pass a frequency-domain signal
              "y_domain": "amplitude"},  # fs-based cutoffs are only truthful for the raw trace
    produces={},                         # stays a time-domain trace, same unit
    params=(
        ParamSpec("low_cutoff", required=True, description="lower band edge", unit="Hz"),
        ParamSpec("high_cutoff", required=True, description="upper band edge", unit="Hz"),
        ParamSpec("order", required=False, description="Butterworth filter order (default 4)"),
    ),
    description=(
        "Zero-phase Butterworth band-pass filter; keeps content between "
        "`low_cutoff` and `high_cutoff` Hz."
    ),
)
def bandpass_filter(signal, low_cutoff, high_cutoff, order=4):
    from scipy.signal import butter

    fs = float(signal.fs)
    low_cutoff = float(low_cutoff)
    high_cutoff = float(high_cutoff)
    order = int(order)
    nyq = fs / 2.0
    if order < 1:
        raise ToolInputError(
            f"bandpass_filter order {order} must be a positive integer."
        )
    if not (0.0 < low_cutoff < high_cutoff < nyq):
        raise ToolInputError(
            f"bandpass_filter band [{low_cutoff}, {high_cutoff}] Hz must satisfy "
            f"0 < low_cutoff < high_cutoff < Nyquist ({nyq:g} Hz) for this "
            f"{fs:g} Hz signal."
        )

    sos = butter(order, [low_cutoff / nyq, high_cutoff / nyq],
                 btype="bandpass", output="sos")
    return _zero_phase_filter(signal, sos, "bandpass_filter", order)


@dsp_tool(
    "notch_filter",
    requires={"x_domain": "time",        # cannot notch a frequency-domain signal
              "y_domain": "amplitude"},  # fs-based cutoffs are only truthful for the raw trace
    produces={},                         # stays a time-domain trace, same unit
    params=(
        ParamSpec("w0", required=False,
                  description="center frequency to notch out (default 60, US mains)",
                  unit="Hz"),
    ),
    description=(
        "Zero-phase IIR notch filter; removes a narrow band centered at `w0` Hz "
        "(default 60 Hz mains interference)."
    ),
)
def notch_filter(signal, w0=60.0, Q=30.0):
    # Q (notch width relative to w0) is deliberately not a ParamSpec: the
    # registry rejects unknown params, so it stays an internal default.
    # fs likewise comes from the signal's own metadata, never from the user.
    from scipy.signal import iirnotch, tf2sos

    fs = float(signal.fs)
    w0 = float(w0)
    nyq = fs / 2.0
    if not (0.0 < w0 < nyq):
        raise ToolInputError(
            f"notch_filter center frequency {w0} Hz must be between 0 and the "
            f"Nyquist frequency ({nyq:g} Hz) for this {fs:g} Hz signal."
        )

    b, a = iirnotch(w0, Q, fs=fs)
    sos = tf2sos(b, a)
    return _zero_phase_filter(signal, sos, "notch_filter", order=2)