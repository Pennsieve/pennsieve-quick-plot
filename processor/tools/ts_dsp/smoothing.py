"""
ts_dsp.smoothing — domain-agnostic smoothers.

Each smoother maps a 1D numeric series to an equal-length 1D numeric series,
so it works on any signal regardless of domain (a raw time-domain trace, an
energy series, a spectrum, ...). Domains and units are carried through
unchanged (`requires={}`, `produces={}`), so smoothers chain freely anywhere
in a pipeline. numpy/scipy are imported lazily inside the functions to keep
module import (which the registry does at startup) cheap.
"""

from __future__ import annotations

from dataclasses import replace

from .registry import dsp_tool, ParamSpec, ToolInputError


def _validated_window(tool_name, y, win_size, *, odd=False):
    """Validate a sample-count window against the series length."""
    w = int(win_size)
    if w < 1:
        raise ToolInputError(
            f"{tool_name} win_size must be a positive number of samples; got {win_size!r}."
        )
    if w > y.size:
        raise ToolInputError(
            f"{tool_name} win_size ({w} samples) exceeds the {y.size}-sample signal."
        )
    if odd and w % 2 == 0:
        raise ToolInputError(
            f"{tool_name} win_size must be odd so the window is symmetric "
            f"about each point; got {w}."
        )
    return w


@dsp_tool(
    "moving_average",
    requires={},                         # any 1D numeric series, any domain
    produces={},                         # same domains/units as the input
    params=(
        ParamSpec("win_size", required=True,
                  description="number of data points used to compute the average",
                  unit="samples"),
    ),
    description="Moving average: each point becomes the mean of the `win_size` points around it.",
)
def moving_average(signal, win_size):
    import numpy as np
    from scipy.ndimage import uniform_filter1d

    y = np.asarray(signal.y, dtype=float)
    w = _validated_window("moving_average", y, win_size)
    return replace(signal, y=uniform_filter1d(y, size=w, mode="nearest"))


@dsp_tool(
    "moving_median",
    requires={},                         # any 1D numeric series, any domain
    produces={},                         # same domains/units as the input
    params=(
        ParamSpec("win_size", required=True,
                  description="number of data points used to compute the median",
                  unit="samples"),
    ),
    description=(
        "Moving median: each point becomes the median of the `win_size` points "
        "around it; robust to outliers/spikes."
    ),
)
def moving_median(signal, win_size):
    import numpy as np
    from scipy.ndimage import median_filter

    y = np.asarray(signal.y, dtype=float)
    w = _validated_window("moving_median", y, win_size)
    return replace(signal, y=median_filter(y, size=w, mode="nearest"))


@dsp_tool(
    "savgol_filter",
    requires={},                         # any 1D numeric series, any domain
    produces={},                         # same domains/units as the input
    params=(
        ParamSpec("win_size", required=True,
                  description="window length; must be odd", unit="samples"),
        ParamSpec("polyorder", required=True,
                  description="polynomial degree fitted in each window; "
                              "higher degrees follow more complex local curves"),
    ),
    description=(
        "Savitzky-Golay filter: slides a symmetric `win_size` window along the "
        "series, fits a degree-`polyorder` polynomial in each window, and "
        "replaces the centre point with the polynomial's value."
    ),
)
def savgol_filter(signal, win_size, polyorder):
    import numpy as np
    from scipy.signal import savgol_filter as _savgol

    y = np.asarray(signal.y, dtype=float)
    w = _validated_window("savgol_filter", y, win_size, odd=True)
    p = int(polyorder)
    if p < 0:
        raise ToolInputError(f"savgol_filter polyorder must be >= 0; got {polyorder!r}.")
    if p >= w:
        raise ToolInputError(
            f"savgol_filter polyorder ({p}) must be smaller than win_size ({w})."
        )
    return replace(signal, y=_savgol(y, window_length=w, polyorder=p))


@dsp_tool(
    "gaussian_filter1d",
    requires={},                         # any 1D numeric series, any domain
    produces={},                         # same domains/units as the input
    params=(
        ParamSpec("sigma", required=True,
                  description="spread of the Gaussian weights; larger sigma = "
                              "broader window and stronger smoothing",
                  unit="samples"),
    ),
    description=(
        "Gaussian average: weighted moving average whose weights follow a "
        "Gaussian centred on each point, so nearby points count more than "
        "distant ones."
    ),
)
def gaussian_filter1d(signal, sigma):
    import numpy as np
    from scipy.ndimage import gaussian_filter1d as _gaussian

    y = np.asarray(signal.y, dtype=float)
    s = float(sigma)
    if s <= 0.0:
        raise ToolInputError(f"gaussian_filter1d sigma must be positive; got {sigma!r}.")
    return replace(signal, y=_gaussian(y, sigma=s, mode="nearest"))
