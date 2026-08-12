"""
ts_dsp.feature_extraction — time-domain features.

Feature tools take a time-domain signal and return a (usually shorter)
time-series of a derived quantity, updating the y-axis domain/unit metadata
accordingly. numpy is imported lazily inside the function.
"""

from __future__ import annotations

from dataclasses import replace

from .registry import dsp_tool, ParamSpec, ToolInputError


@dsp_tool(
    "energy",
    requires={"x_domain": "time"},       # a short-time feature of a time-domain trace
    produces={"y_domain": "energy"},     # y becomes energy (unit set from the input unit)
    params=(
        ParamSpec("window_size", required=True, description="window length", unit="s"),
        ParamSpec("window_stride", required=True, description="hop between successive windows", unit="s"),
    ),
    description="Short-time energy: integral of squared amplitude over each sliding window.",
)
def energy(signal, window_size, window_stride):
    import numpy as np

    fs = float(signal.fs)
    y = np.asarray(signal.y, dtype=float)
    t = np.asarray(signal.t, dtype=float)

    w = int(round(float(window_size) * fs))
    h = int(round(float(window_stride) * fs))
    if w <= 0 or h <= 0:
        raise ToolInputError("energy 'window_size' and 'window_stride' must be positive (seconds).")
    if w > y.size:
        raise ToolInputError(
            f"energy window_size ({window_size}s ≈ {w} samples) exceeds the "
            f"{y.size}-sample signal."
        )

    starts = list(range(0, y.size - w + 1, h))
    # Energy per window = integral of y^2 over the window ≈ sum(y^2)·dt, dt=1/fs.
    e = np.array([float(np.sum(y[s:s + w] ** 2) / fs) for s in starts], dtype=float)
    # Place each energy value at its window's centre time (in the input x_unit).
    centres = np.array([t[s + w // 2] for s in starts], dtype=float)

    return replace(signal, t=centres, y=e, y_domain="energy", y_unit=f"{signal.y_unit}²·s")