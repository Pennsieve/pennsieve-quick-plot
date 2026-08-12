"""
ts_dsp — voltage-timeseries DSP toolkit (EEG + iEEG).

A template uses this family as a library:

    from processor.tools.ts_dsp import Signal, read_signal, apply_dsp_pipeline

    signal = read_signal(path, channel, start_s, end_s, ...)  # raw acquisition
    signal = apply_dsp_pipeline(signal, steps)                # optional DSP
    # ... template plots `signal` ...

Layout:
  signal.py             the Signal contract + domain/unit vocabulary (pure)
  io.py                 raw acquisition: probe_edf / read_edf / read_signal
                        (read + crop + unit-convert -> raw Signal; not tools)
  bandpass.py           frequency-selective filters (e.g. highpass_filter)
  feature_extraction.py time-domain features (e.g. energy)
  registry.py           the tool registry + apply_dsp_pipeline driver

Raw acquisition (io.py) is the always-on plumbing every EDF template shares;
it is deliberately NOT in the tool registry. Only opt-in, user-composable DSP
transforms register via @dsp_tool. Importing this package imports the tool
modules so their registrations run; tool modules keep heavy imports
(numpy/scipy/pyedflib) inside the functions, so import stays cheap.
"""

from __future__ import annotations

# Contract + unit vocabulary.
from .signal import (  # noqa: F401
    Signal,
    X_DOMAINS,
    Y_DOMAINS,
    resolve_time_unit,
    resolve_volt_unit,
    seconds_to_clock,
)

# Registry machinery + pipeline driver.
from .registry import (  # noqa: F401
    ToolInputError,
    ToolSpec,
    ParamSpec,
    dsp_tool,
    get,
    known_names,
    specs,
    apply_dsp_pipeline,
)

# Raw acquisition (sources / builders — not registered tools).
from .io import probe_edf, read_edf, read_signal  # noqa: F401

# Import the DSP tool modules so their @dsp_tool decorators register.
from . import bandpass as _bandpass  # noqa: F401
from . import feature_extraction as _feature_extraction  # noqa: F401

__all__ = [
    "Signal",
    "X_DOMAINS",
    "Y_DOMAINS",
    "resolve_time_unit",
    "resolve_volt_unit",
    "seconds_to_clock",
    "ToolInputError",
    "ToolSpec",
    "ParamSpec",
    "dsp_tool",
    "get",
    "known_names",
    "specs",
    "apply_dsp_pipeline",
    "probe_edf",
    "read_edf",
    "read_signal",
]