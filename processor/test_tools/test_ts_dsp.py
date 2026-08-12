"""
Unit tests for the ts_dsp toolkit: the registry / pipeline driver and the
individual tools. These build `Signal`s directly from numpy arrays, so they
need no EDF file and run fast.

To run:
pytest processor/test_tools/test_ts_dsp.py
"""

from __future__ import annotations

import numpy as np
import pytest

from processor.tools.ts_dsp import (
    Signal,
    apply_dsp_pipeline,
    known_names,
    ToolInputError,
)


def _ts(y, fs=256.0, y_unit="µV", x_domain="time"):
    """A minimal time-domain Signal wrapping array `y`."""
    y = np.asarray(y, dtype=float)
    t = np.arange(y.size) / fs
    return Signal(t=t, y=y, fs=fs, channel="C", x_domain=x_domain,
                  x_unit="s", y_domain="amplitude", y_unit=y_unit)


# REGISTRY / PIPELINE DRIVER
def test_tools_are_registered():
    # Only opt-in DSP transforms register; raw acquisition (read_signal etc.)
    # is not a tool.
    assert set(known_names()) >= {"highpass_filter", "energy"}


def test_empty_pipeline_returns_signal_unchanged():
    sig = _ts(np.ones(64))
    assert apply_dsp_pipeline(sig, None) is sig
    assert apply_dsp_pipeline(sig, []) is sig


def test_unknown_tool_raises():
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(64)), [{"tool": "no_such_tool"}])


def test_missing_required_param_raises():
    # highpass_filter requires 'cutoff'
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(256)), [{"tool": "highpass_filter", "params": {}}])


def test_unknown_param_raises():
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(256)),
                           [{"tool": "energy",
                             "params": {"window_size": 0.1, "window_stride": 0.1, "bogus": 5}}])


def test_malformed_step_raises():
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(64)), [{"params": {}}])  # no "tool" key


def test_domain_mismatch_raises():
    # highpass_filter requires x_domain == "time"; feed it a frequency signal.
    freq_sig = _ts(np.ones(256), x_domain="frequency")
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(freq_sig, [{"tool": "highpass_filter", "params": {"cutoff": 10}}])


# HIGHPASS_FILTER
def test_highpass_removes_dc_offset():
    fs = 256.0
    t = np.arange(int(2 * fs)) / fs
    y = 50.0 + np.sin(2 * np.pi * 40.0 * t)          # DC 50 + 40 Hz component
    out = apply_dsp_pipeline(_ts(y, fs=fs),
                             [{"tool": "highpass_filter", "params": {"cutoff": 10, "order": 4}}])
    assert abs(float(np.mean(out.y))) < 1.0          # DC (was 50) removed
    assert out.y.size == y.size                      # same length, same domain
    assert out.x_domain == "time" and out.y_domain == "amplitude"


def test_highpass_cutoff_above_nyquist_raises():
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(256), fs=256.0),
                           [{"tool": "highpass_filter", "params": {"cutoff": 200}}])  # nyq=128


def test_highpass_too_few_samples_raises():
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(8), fs=256.0),
                           [{"tool": "highpass_filter", "params": {"cutoff": 10}}])


# ENERGY
def test_energy_windows_and_updates_domain():
    fs = 256.0
    sig = _ts(np.full(512, 2.0), fs=fs, y_unit="µV")   # constant 2 µV
    out = apply_dsp_pipeline(
        sig, [{"tool": "energy", "params": {"window_size": 0.5, "window_stride": 0.5}}]
    )
    # 128-sample windows, energy = sum(2^2)/fs = 4*128/256 = 2.0 per window.
    assert out.y_domain == "energy"
    assert out.y_unit == "µV²·s"
    assert out.x_domain == "time"                      # still a time-series
    assert out.y.size == 4 and np.allclose(out.y, 2.0)
    assert out.y.size < sig.y.size                     # downsampled to windows


def test_energy_window_larger_than_signal_raises():
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(64), fs=256.0),
                           [{"tool": "energy", "params": {"window_size": 10, "window_stride": 1}}])


# COMPOSITION: filters chain, then a feature
def test_filter_then_energy_composes():
    fs = 256.0
    t = np.arange(int(2 * fs)) / fs
    y = 50.0 + np.sin(2 * np.pi * 40.0 * t)
    out = apply_dsp_pipeline(_ts(y, fs=fs), [
        {"tool": "highpass_filter", "params": {"cutoff": 10}},
        {"tool": "energy", "params": {"window_size": 0.25, "window_stride": 0.25}},
    ])
    assert out.y_domain == "energy" and out.y.size > 0