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


# LOWPASS_FILTER
def test_lowpass_attenuates_high_tone():
    fs = 256.0
    t = np.arange(int(2 * fs)) / fs
    low = np.sin(2 * np.pi * 5.0 * t)
    high = np.sin(2 * np.pi * 80.0 * t)
    out = apply_dsp_pipeline(_ts(low + high, fs=fs),
                             [{"tool": "lowpass_filter", "params": {"cutoff": 30, "order": 4}}])
    # 80 Hz tone removed, 5 Hz tone passes: output ≈ the low tone alone.
    assert np.sqrt(np.mean((out.y - low) ** 2)) < 0.1
    assert out.y.size == t.size
    assert out.x_domain == "time" and out.y_domain == "amplitude"


# BANDPASS_FILTER
def test_bandpass_keeps_inband_kills_outband():
    fs = 256.0
    t = np.arange(int(2 * fs)) / fs
    inband = np.sin(2 * np.pi * 20.0 * t)
    below = np.sin(2 * np.pi * 2.0 * t)
    above = np.sin(2 * np.pi * 100.0 * t)
    out = apply_dsp_pipeline(
        _ts(inband + below + above, fs=fs),
        [{"tool": "bandpass_filter",
          "params": {"low_cutoff": 10, "high_cutoff": 40, "order": 4}}])
    # 2 Hz and 100 Hz tones removed, 20 Hz tone passes ≈ unchanged.
    assert np.sqrt(np.mean((out.y - inband) ** 2)) < 0.15
    assert out.x_domain == "time" and out.y_domain == "amplitude"


@pytest.mark.parametrize("low,high", [
    (-5, 40),    # negative lower edge
    (0, 40),     # zero lower edge
    (40, 40),    # empty band
    (50, 40),    # inverted band
    (10, 128),   # upper edge at Nyquist (fs=256)
    (10, 200),   # upper edge above Nyquist
])
def test_bandpass_invalid_band_raises(low, high):
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(512), fs=256.0),
                           [{"tool": "bandpass_filter",
                             "params": {"low_cutoff": low, "high_cutoff": high}}])


# NOTCH_FILTER
def _tone_amplitude(y, t, f):
    """Amplitude of the `f` Hz component of `y` by complex projection."""
    return 2.0 * abs(np.mean(y * np.exp(-2j * np.pi * f * t)))


def test_notch_kills_60hz_keeps_rest():
    fs = 512.0
    t = np.arange(int(2 * fs)) / fs
    y = np.sin(2 * np.pi * 10.0 * t) + np.sin(2 * np.pi * 60.0 * t)
    out = apply_dsp_pipeline(_ts(y, fs=fs), [{"tool": "notch_filter", "params": {}}])
    assert _tone_amplitude(out.y, t, 60.0) < 0.1    # mains tone removed (default w0=60)
    assert abs(_tone_amplitude(out.y, t, 10.0) - 1.0) < 0.05   # signal tone intact
    assert out.x_domain == "time" and out.y_domain == "amplitude"


@pytest.mark.parametrize("w0", [-60, 0, 128, 200])  # Nyquist = 128 for fs=256
def test_notch_w0_out_of_range_raises(w0):
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(512), fs=256.0),
                           [{"tool": "notch_filter", "params": {"w0": w0}}])


# SHARED FILTER PARAM VALIDATION
@pytest.mark.parametrize("tool", ["highpass_filter", "lowpass_filter"])
@pytest.mark.parametrize("cutoff", [-10, 0, 128, 200])  # Nyquist = 128 for fs=256
def test_out_of_range_cutoff_raises(tool, cutoff):
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(512), fs=256.0),
                           [{"tool": tool, "params": {"cutoff": cutoff}}])


@pytest.mark.parametrize("step", [
    {"tool": "highpass_filter", "params": {"cutoff": 10, "order": 0}},
    {"tool": "highpass_filter", "params": {"cutoff": 10, "order": -3}},
    {"tool": "lowpass_filter", "params": {"cutoff": 10, "order": 0}},
    {"tool": "bandpass_filter", "params": {"low_cutoff": 5, "high_cutoff": 40, "order": 0}},
])
def test_nonpositive_order_raises(step):
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(512), fs=256.0), [step])


def test_order_one_is_accepted():
    # Lower bound is inclusive: order 1 is a valid Butterworth filter.
    out = apply_dsp_pipeline(_ts(np.ones(512), fs=256.0),
                             [{"tool": "lowpass_filter", "params": {"cutoff": 30, "order": 1}}])
    assert out.y.size == 512


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