"""
ts_dsp.frequency — time-domain -> 1D frequency-domain transforms.

Each transform consumes a time-domain signal and returns a one-sided
spectrum: `t` becomes the frequency axis (Hz) and `y` the spectral quantity,
with x/y domains and units updated accordingly. Output stays a 1D Signal, so
domain-agnostic tools (e.g. the smoothers) can still run afterwards, while
time-domain-only tools are rejected by the registry. numpy/scipy are
imported lazily inside the functions to keep module import (which the
registry does at startup) cheap.
"""

from __future__ import annotations

from dataclasses import replace

from .registry import dsp_tool, ParamSpec, ToolInputError


def _time_series(signal, tool_name, min_samples=2):
    """The signal's samples as a float array, guarding a minimum length."""
    import numpy as np

    y = np.asarray(signal.y, dtype=float)
    if y.size < min_samples:
        raise ToolInputError(
            f"{tool_name} needs at least {min_samples} samples; the window "
            f"has only {y.size}."
        )
    return y


def _spectrum(signal, freqs, values, y_domain, y_unit):
    """Repackage a signal as a one-sided spectrum over `freqs` (Hz)."""
    # x_tick_style resets to "numeric": "clock" only makes sense for a
    # seconds-since-midnight time axis, never for a frequency axis.
    return replace(
        signal, t=freqs, y=values,
        x_domain="frequency", x_unit="Hz",
        y_domain=y_domain, y_unit=y_unit,
        x_tick_style="numeric",
    )


@dsp_tool(
    "fft",
    requires={"x_domain": "time",        # spectra are computed from a time-domain trace
              "y_domain": "amplitude"},  # whose fs-derived frequency axis is truthful
    produces={"x_domain": "frequency",   # x becomes frequency (Hz)
              "y_domain": "magnitude"},  # y becomes magnitude (unit set from the input unit)
    params=(),                
    description=(
        "One-sided amplitude spectrum via the FFT: how strongly each frequency "
        "component is present, scaled so a pure sinusoid of amplitude A shows "
        "a peak of height A."
    ),
)
def fft(signal):
    import numpy as np
    from scipy.fft import rfft, rfftfreq

    y = _time_series(signal, "fft")
    fs = float(signal.fs)

    spec = np.abs(rfft(y)) # rfft decompose the signal into its frequency components and returns one complex number each freq bin
    # to convert the complex numbers to magnitudes:
    spec /= y.size  
    spec[1:] *= 2.0 
    if y.size % 2 == 0:
        spec[-1] /= 2.0

    # rfftfreq returns the frequencies-axis labels that correspondes to the FFT bins, in Hz.
    freqs = rfftfreq(y.size, d=1.0 / fs)
    return _spectrum(signal, freqs, spec,
                     y_domain="magnitude", y_unit=signal.y_unit)


@dsp_tool(
    "psd",
    requires={"x_domain": "time",        # spectra are computed from a time-domain trace
              "y_domain": "amplitude"},  # whose fs-derived frequency axis is truthful
    produces={"x_domain": "frequency",   # x becomes frequency (Hz)
              "y_domain": "power"},      # y becomes power per Hz
    params=(
        ParamSpec("win_size", required=False,
                  description="Welch segment length; longer segments give finer "
                              "frequency resolution, shorter ones smoother "
                              "estimates (default 1 s, capped at the signal length)",
                  unit="s"),
    ),
    description=(
        "Power spectral density (Welch's method): how the signal's power "
        "distributes across frequency, separating sustained features from "
        "noise. Useful for characterising brain states, locating seizure-onset "
        "zones, and tracking high-frequency oscillations."
    ),
)
def psd(signal, win_size=None):
    from scipy.signal import welch

    y = _time_series(signal, "psd")
    fs = float(signal.fs)
    if win_size is None:
        nperseg = min(y.size, max(2, int(round(fs))))  # 1-s segments by default
    else:
        nperseg = int(round(float(win_size) * fs))
        if nperseg < 2:
            raise ToolInputError(
                f"psd win_size ({win_size}s ≈ {nperseg} samples) is too short; "
                f"it must cover at least 2 samples."
            )
        if nperseg > y.size:
            raise ToolInputError(
                f"psd win_size ({win_size}s ≈ {nperseg} samples) exceeds the "
                f"{y.size}-sample signal."
            )
    freqs, pxx = welch(y, fs=fs, nperseg=nperseg)
    return _spectrum(signal, freqs, pxx,
                     y_domain="power", y_unit=f"{signal.y_unit}²/Hz")
