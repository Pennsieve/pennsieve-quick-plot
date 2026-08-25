"""
ts_dsp.feature_extraction — time-domain features.

Feature tools take a time-domain amplitude trace and return a (usually
shorter) time-series of a derived quantity, updating the y-axis domain/unit
metadata accordingly. All of them share the same sliding-window skeleton
(`window_size` / `window_stride` in seconds, each value placed at its
window's centre time), factored into _windowed(). numpy/scipy are imported
lazily inside the functions.
"""

from __future__ import annotations

from dataclasses import replace

from .registry import dsp_tool, ParamSpec, ToolInputError


# Every feature is windowed the same way, so they share one ParamSpec pair.
_WINDOW_PARAMS = (
    ParamSpec("window_size", required=True, description="window length", unit="s"),
    ParamSpec("window_stride", required=True, description="hop between successive windows", unit="s"),
)


def _windowed(signal, tool_name, window_size, window_stride, per_window):
    """Slide a window over the signal, apply `per_window` to each chunk.

    Returns (values, centres): one feature value per window, placed at the
    window's centre time (in the input x_unit). Validation mirrors energy's
    original checks: positive seconds, window must fit the signal.
    """
    import numpy as np

    fs = float(signal.fs)
    y = np.asarray(signal.y, dtype=float)
    t = np.asarray(signal.t, dtype=float)

    w = int(round(float(window_size) * fs))
    h = int(round(float(window_stride) * fs))
    if w <= 0 or h <= 0:
        raise ToolInputError(
            f"{tool_name} 'window_size' and 'window_stride' must be positive (seconds)."
        )
    if w > y.size:
        raise ToolInputError(
            f"{tool_name} window_size ({window_size}s ≈ {w} samples) exceeds the "
            f"{y.size}-sample signal."
        )

    starts = list(range(0, y.size - w + 1, h))
    values = np.array([per_window(y[s:s + w]) for s in starts], dtype=float)
    centres = np.array([t[s + w // 2] for s in starts], dtype=float)
    return values, centres


@dsp_tool(
    "energy",
    requires={"x_domain": "time",        # a short-time feature of a time-domain trace
              "y_domain": "amplitude"},  # whose fs-based windowing is truthful (and
                                         # energy-of-energy is meaningless anyway)
    produces={"y_domain": "energy"},     # y becomes energy (unit set from the input unit)
    params=_WINDOW_PARAMS,
    description="Short-time energy: integral of squared amplitude over each sliding window.",
)
def energy(signal, window_size, window_stride):
    import numpy as np

    fs = float(signal.fs)
    # Energy per window = integral of y^2 over the window ≈ sum(y^2)·dt, dt=1/fs.
    values, centres = _windowed(signal, "energy", window_size, window_stride,
                                lambda win: float(np.sum(win ** 2) / fs))
    return replace(signal, t=centres, y=values,
                   y_domain="energy", y_unit=f"{signal.y_unit}²·s")


@dsp_tool(
    "power",
    requires={"x_domain": "time",
              "y_domain": "amplitude"},  # fs-based windowing is only truthful for the raw trace
    produces={"y_domain": "power"},      # y becomes mean squared amplitude per window
    params=_WINDOW_PARAMS,
    description="Short-time power: mean squared amplitude over each sliding window.",
)
def power(signal, window_size, window_stride):
    import numpy as np

    values, centres = _windowed(signal, "power", window_size, window_stride,
                                lambda win: float(np.mean(win ** 2)))
    return replace(signal, t=centres, y=values,
                   y_domain="power", y_unit=f"{signal.y_unit}²")


@dsp_tool(
    "rms",
    requires={"x_domain": "time",
              "y_domain": "amplitude"},  # fs-based windowing is only truthful for the raw trace
    produces={"y_domain": "rms"},        # y becomes RMS amplitude per window (same unit)
    params=_WINDOW_PARAMS,
    description=(
        "Root-mean-square amplitude over each sliding window; an amplitude "
        "envelope in the signal's own unit."
    ),
)
def rms(signal, window_size, window_stride):
    import numpy as np

    values, centres = _windowed(signal, "rms", window_size, window_stride,
                                lambda win: float(np.sqrt(np.mean(win ** 2))))
    return replace(signal, t=centres, y=values,
                   y_domain="rms", y_unit=signal.y_unit)


@dsp_tool(
    "zcr",
    requires={"x_domain": "time",
              "y_domain": "amplitude"},  # fs-based windowing is only truthful for the raw trace
    produces={"y_domain": "zcr"},        # y becomes zero-crossings per second
    params=_WINDOW_PARAMS,
    description=(
        "Zero-crossing rate: sign changes per second within each sliding "
        "window; a cheap proxy for the dominant frequency of the trace."
    ),
)
def zcr(signal, window_size, window_stride):
    import numpy as np

    fs = float(signal.fs)

    def _rate(win):
        crossings = np.count_nonzero(np.diff(np.signbit(win)))
        return float(crossings / (win.size / fs))   # crossings per second

    values, centres = _windowed(signal, "zcr", window_size, window_stride, _rate)
    return replace(signal, t=centres, y=values, y_domain="zcr", y_unit="1/s")


@dsp_tool(
    "line_length",
    requires={"x_domain": "time",
              "y_domain": "amplitude"},  # fs-based windowing is only truthful for the raw trace
    produces={"y_domain": "line_length"},  # y becomes summed |sample-to-sample change|
    params=_WINDOW_PARAMS,
    description=(
        "Line length: sum of absolute sample-to-sample differences in each "
        "sliding window. Sensitive to both amplitude and frequency increases; "
        "a classic iEEG seizure-detection feature."
    ),
)
def line_length(signal, window_size, window_stride):
    import numpy as np

    values, centres = _windowed(signal, "line_length", window_size, window_stride,
                                lambda win: float(np.sum(np.abs(np.diff(win)))))
    return replace(signal, t=centres, y=values,
                   y_domain="line_length", y_unit=signal.y_unit)


@dsp_tool(
    "kurtosis",
    requires={"x_domain": "time",
              "y_domain": "amplitude"},  # fs-based windowing is only truthful for the raw trace
    produces={"y_domain": "kurtosis"},   # y becomes excess kurtosis (dimensionless)
    params=_WINDOW_PARAMS,
    description=(
        "Sliding-window kurtosis (Fisher, excess): tailedness of the amplitude "
        "distribution; large values flag spiky, outlier-heavy segments."
    ),
)
def kurtosis(signal, window_size, window_stride):
    from scipy.stats import kurtosis as _kurtosis

    values, centres = _windowed(signal, "kurtosis", window_size, window_stride,
                                lambda win: float(_kurtosis(win)))
    return replace(signal, t=centres, y=values, y_domain="kurtosis", y_unit="")


@dsp_tool(
    "skewness",
    requires={"x_domain": "time",
              "y_domain": "amplitude"},  # fs-based windowing is only truthful for the raw trace
    produces={"y_domain": "skewness"},   # y becomes skewness (dimensionless)
    params=_WINDOW_PARAMS,
    description=(
        "Sliding-window skewness: asymmetry of the amplitude distribution "
        "around its mean (0 for symmetric signals)."
    ),
)
def skewness(signal, window_size, window_stride):
    from scipy.stats import skew as _skew

    values, centres = _windowed(signal, "skewness", window_size, window_stride,
                                lambda win: float(_skew(win)))
    return replace(signal, t=centres, y=values, y_domain="skewness", y_unit="")
