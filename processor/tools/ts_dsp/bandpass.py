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


@dsp_tool(
    "highpass_filter",
    requires={"x_domain": "time"},       # cannot high-pass a frequency-domain signal
    produces={},                         # stays a time-domain trace, same unit
    params=(
        ParamSpec("cutoff", required=True, description="high-pass cutoff frequency", unit="Hz"),
        ParamSpec("order", required=False, description="Butterworth filter order (default 4)"),
    ),
    description="Zero-phase Butterworth high-pass filter; attenuates content below `cutoff` Hz.",
)
def highpass_filter(signal, cutoff, order=4):
    import numpy as np
    from scipy.signal import butter, filtfilt

    fs = float(signal.fs)
    cutoff = float(cutoff)
    order = int(order)
    nyq = fs / 2.0
    if not (0.0 < cutoff < nyq):
        raise ToolInputError(
            f"highpass_filter cutoff {cutoff} Hz must be between 0 and the "
            f"Nyquist frequency ({nyq:g} Hz) for this {fs:g} Hz signal."
        )

    y = np.asarray(signal.y, dtype=float)
    b, a = butter(order, cutoff / nyq, btype="highpass")
    # filtfilt (zero-phase, forward+backward) needs a minimum length.
    padlen = 3 * max(len(a), len(b))
    if y.size <= padlen:
        raise ToolInputError(
            f"highpass_filter needs more than {padlen} samples to filter at "
            f"order {order}; the window has only {y.size}."
        )
    y_filtered = filtfilt(b, a, y)
    return replace(signal, y=y_filtered)