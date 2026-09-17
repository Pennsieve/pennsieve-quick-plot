"""
ts_dsp — voltage-timeseries DSP toolkit (EEG + iEEG).

Signal -> Signal transforms a template can chain via its `pipeline` argument,
plus the shared definitions everything codes against.

Layout:
  signal_definition.py  the Signal dataclass + domain vocabulary
  units.py              time/voltage unit tables, H:M:S helpers
  dsp_pipeline.py       the @dsp_tool contract + apply_dsp_pipeline runner
  filters.py            frequency-selective filters (highpass, lowpass, …)
  smoothing.py          domain-agnostic smoothers (moving_average, …)
  frequency.py          time -> frequency-domain transforms (fft, psd)
  feature_extraction.py windowed time-domain features (energy, rms, …)

The tool registry is the explicit list below (`ALL_TOOLS`), mirroring how
`templates/__init__.py` lists the templates: to add a tool, write the
function, decorate it with @dsp_tool, and add it to the list. Nothing
registers itself as a side effect of being imported.

Reading a file into a Signal is not a tool and lives in `processor/readers/`.
"""

from __future__ import annotations

# Shared definitions.
from .signal_definition import Signal, X_DOMAINS, Y_DOMAINS  # noqa: F401
from .units import (  # noqa: F401
    resolve_time_unit,
    resolve_volt_unit,
    is_clock_string,
    clock_to_seconds,
    seconds_to_clock,
)
from .dsp_pipeline import ToolInputError, ToolSpec, apply_dsp_pipeline  # noqa: F401

# The tools.
from .filters import highpass_filter, lowpass_filter, bandpass_filter, notch_filter
from .smoothing import moving_average, moving_median, savgol_filter, gaussian_filter1d
from .frequency import fft, psd
from .feature_extraction import energy, power, rms, zcr, line_length, kurtosis, skewness

############# TOOL REGISTRY ##################
ALL_TOOLS = [
    # filters.py — time/amplitude in, time/amplitude out
    highpass_filter, lowpass_filter, bandpass_filter, notch_filter,
    # smoothing.py — any domain in, same domain out
    moving_average, moving_median, savgol_filter, gaussian_filter1d,
    # frequency.py — time/amplitude in, frequency spectrum out
    fft, psd,
    # feature_extraction.py — time/amplitude in, windowed feature series out
    energy, power, rms, zcr, line_length, kurtosis, skewness,
]

_REGISTRY: dict[str, ToolSpec] = {t.spec.name: t.spec for t in ALL_TOOLS}
assert len(_REGISTRY) == len(ALL_TOOLS), "duplicate DSP tool name in ALL_TOOLS"


def get(name: str) -> "ToolSpec | None":
    return _REGISTRY.get(name)


def known_names() -> list[str]:
    return sorted(_REGISTRY)


def specs() -> list[ToolSpec]:
    """All registered specs, name-sorted — used by the schema generator."""
    return [_REGISTRY[n] for n in known_names()]


__all__ = [
    # definitions
    "Signal", "X_DOMAINS", "Y_DOMAINS",
    "resolve_time_unit", "resolve_volt_unit",
    "is_clock_string", "clock_to_seconds", "seconds_to_clock",
    # pipeline
    "ToolInputError", "apply_dsp_pipeline",
    # registry
    "ALL_TOOLS", "get", "known_names", "specs",
]
