"""
Unit tests for the ts_dsp toolkit: the registry / pipeline driver and the
individual tools. These build `Signal`s directly from numpy arrays, so they
need no EDF file and run fast.

Layout (one banner section per source module / concern):
  1. REGISTRY / PIPELINE DRIVER  — plumbing shared by every tool
  2. FREQUENCY-SELECTIVE FILTERS — filters.py (highpass/lowpass/bandpass/notch)
  3. SMOOTHERS                   — smoothing.py (moving_average/median, savgol, gaussian)
  4. FREQUENCY TRANSFORMS        — frequency.py (fft, psd)
  5. FEATURES                    — feature_extraction.py (energy, power, rms,
                                   zcr, line_length, kurtosis, skewness)
  6. COMPOSITION                 — multi-step pipelines

To run:
pytest processor/test_tools/test_ts_dsp.py
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from processor.tools.ts_dsp import (
    Signal,
    X_DOMAINS,
    Y_DOMAINS,
    apply_dsp_pipeline,
    known_names,
    specs,
    ToolInputError,
)


def _ts(y, fs=256.0, y_unit="µV", x_domain="time"):
    """A minimal time-domain Signal wrapping array `y`."""
    y = np.asarray(y, dtype=float)
    t = np.arange(y.size) / fs
    return Signal(t=t, y=y, fs=fs, channel="C", x_domain=x_domain,
                  x_unit="s", y_domain="amplitude", y_unit=y_unit)


def _tone_amplitude(y, t, f):
    """Amplitude of the `f` Hz component of `y` by complex projection."""
    return 2.0 * abs(np.mean(y * np.exp(-2j * np.pi * f * t)))


# ---------------------------------------------------------------------------
# 1. REGISTRY / PIPELINE DRIVER
#    apply_dsp_pipeline plumbing shared by every tool: step parsing, tool
#    lookup, param checking, and domain gating. Tested once here, not
#    re-tested per tool.
# ---------------------------------------------------------------------------
def test_tools_are_registered():
    # Only opt-in DSP transforms register; raw acquisition (read_signal etc.)
    # is not a tool.
    assert set(known_names()) >= {"highpass_filter", "energy"}


def test_tool_domains_are_in_the_vocabulary():
    # The gating type-system works by exact string match against the
    # X_DOMAINS / Y_DOMAINS vocabulary in signal.py. A typo in a tool's
    # requires/produces (e.g. "magnitde") would otherwise register fine and
    # just silently never match a gate.
    for spec in specs():
        for field, source in (("requires", spec.requires), ("produces", spec.produces)):
            assert set(source) <= {"x_domain", "y_domain"}, (
                f"{spec.name}.{field} has unknown axis key(s): {sorted(set(source) - {'x_domain', 'y_domain'})}"
            )
            if "x_domain" in source:
                assert source["x_domain"] in X_DOMAINS, (
                    f"{spec.name}.{field} x_domain {source['x_domain']!r} not in {X_DOMAINS}"
                )
            if "y_domain" in source:
                assert source["y_domain"] in Y_DOMAINS, (
                    f"{spec.name}.{field} y_domain {source['y_domain']!r} not in {Y_DOMAINS}"
                )


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


# ---------------------------------------------------------------------------
# 2. FREQUENCY-SELECTIVE FILTERS (tools/ts_dsp/filters.py)
#    highpass_filter, lowpass_filter, bandpass_filter, notch_filter.
#    Per tool: one numeric-correctness test on known sinusoids, then the
#    invalid-parameter error paths. Shared cutoff/order validation is
#    parametrized across tools at the end of the section.
# ---------------------------------------------------------------------------

# -- highpass_filter: correctness + error paths --
def test_highpass_removes_dc_offset():
    fs = 256.0
    t = np.arange(int(2 * fs)) / fs
    y = 50.0 + np.sin(2 * np.pi * 40.0 * t)          # DC 50 + 40 Hz component
    out = apply_dsp_pipeline(_ts(y, fs=fs),
                             [{"tool": "highpass_filter", "params": {"cutoff": 10, "order": 4}}])
    assert abs(float(np.mean(out.y))) < 1.0          # DC (was 50) removed
    assert out.y.size == y.size                      # same length, same domain
    assert out.x_domain == "time" and out.y_domain == "amplitude"


def test_highpass_too_few_samples_raises():
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(8), fs=256.0),
                           [{"tool": "highpass_filter", "params": {"cutoff": 10}}])


# -- lowpass_filter: correctness --
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


# -- bandpass_filter: correctness + invalid band shapes --
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


# -- notch_filter: correctness + invalid center frequency --
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


# -- shared validation across highpass/lowpass/bandpass --
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


# ---------------------------------------------------------------------------
# 3. SMOOTHERS (tools/ts_dsp/smoothing.py)
#    moving_average, moving_median, savgol_filter, gaussian_filter1d.
#    Domain-agnostic and length-preserving. Per tool: one numeric-correctness
#    test on a known series, then the invalid-parameter error paths.
# ---------------------------------------------------------------------------

# -- moving_average: correctness --
def test_moving_average_of_constant_is_constant():
    out = apply_dsp_pipeline(_ts(np.full(64, 3.5)),
                             [{"tool": "moving_average", "params": {"win_size": 5}}])
    assert np.allclose(out.y, 3.5)
    assert out.y.size == 64                            # length preserved
    assert out.x_domain == "time" and out.y_domain == "amplitude"


def test_moving_average_known_values():
    # Hand-computed: size-3 mean over an isolated unit-3 spike, edges repeated.
    out = apply_dsp_pipeline(_ts([0.0, 0.0, 3.0, 0.0, 0.0]),
                             [{"tool": "moving_average", "params": {"win_size": 3}}])
    assert np.allclose(out.y, [0.0, 1.0, 1.0, 1.0, 0.0])


# -- moving_median: correctness --
def test_moving_median_removes_isolated_spike():
    y = np.ones(32)
    y[10] = 100.0
    out = apply_dsp_pipeline(_ts(y),
                             [{"tool": "moving_median", "params": {"win_size": 3}}])
    assert np.allclose(out.y, 1.0)                     # spike gone, baseline intact


# -- savgol_filter: correctness --
def test_savgol_reproduces_polynomial_exactly():
    # A degree-`polyorder` polynomial is a fixed point of Savitzky-Golay.
    x = np.arange(64, dtype=float)
    y = 0.5 * x**2 - 3.0 * x + 2.0
    out = apply_dsp_pipeline(_ts(y),
                             [{"tool": "savgol_filter",
                               "params": {"win_size": 11, "polyorder": 2}}])
    assert np.allclose(out.y, y)


# -- gaussian_filter1d: correctness --
def test_gaussian_preserves_constant_and_spreads_spike():
    out = apply_dsp_pipeline(_ts(np.full(64, 2.0)),
                             [{"tool": "gaussian_filter1d", "params": {"sigma": 3}}])
    assert np.allclose(out.y, 2.0)                     # constant is a fixed point

    y = np.zeros(64)
    y[32] = 1.0
    out = apply_dsp_pipeline(_ts(y),
                             [{"tool": "gaussian_filter1d", "params": {"sigma": 2}}])
    assert float(out.y.max()) < 0.3                    # spike attenuated...
    assert np.isclose(float(out.y.sum()), 1.0)         # ...but total mass preserved


# -- all smoothers: domain handling --
def test_smoothers_are_domain_agnostic():
    # requires={} means smoothers accept non-time-domain signals (unlike the
    # frequency-selective filters, which are gated to x_domain == "time").
    freq_sig = _ts(np.ones(64), x_domain="frequency")
    out = apply_dsp_pipeline(freq_sig,
                             [{"tool": "moving_average", "params": {"win_size": 5}}])
    assert out.x_domain == "frequency"                 # domain carried through unchanged


# -- shared validation across the windowed smoothers --
@pytest.mark.parametrize("tool,params", [
    ("moving_average", {"win_size": 0}),
    ("moving_average", {"win_size": -3}),
    ("moving_median", {"win_size": 0}),
    ("savgol_filter", {"win_size": -5, "polyorder": 2}),
])
def test_smoother_nonpositive_win_size_raises(tool, params):
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(64)), [{"tool": tool, "params": params}])


@pytest.mark.parametrize("tool,params", [
    ("moving_average", {"win_size": 100}),
    ("moving_median", {"win_size": 100}),
    ("savgol_filter", {"win_size": 101, "polyorder": 2}),
])
def test_smoother_win_size_exceeds_signal_raises(tool, params):
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(64)), [{"tool": tool, "params": params}])


# -- savgol_filter: window/polyorder constraints --
def test_savgol_even_win_size_raises():
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(64)),
                           [{"tool": "savgol_filter",
                             "params": {"win_size": 10, "polyorder": 2}}])


@pytest.mark.parametrize("polyorder", [5, 7])          # == and > win_size
def test_savgol_polyorder_not_below_win_size_raises(polyorder):
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(64)),
                           [{"tool": "savgol_filter",
                             "params": {"win_size": 5, "polyorder": polyorder}}])


def test_savgol_negative_polyorder_raises():
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(64)),
                           [{"tool": "savgol_filter",
                             "params": {"win_size": 5, "polyorder": -1}}])


# -- gaussian_filter1d: sigma constraint --
@pytest.mark.parametrize("sigma", [0, -1.5])
def test_gaussian_nonpositive_sigma_raises(sigma):
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(64)),
                           [{"tool": "gaussian_filter1d", "params": {"sigma": sigma}}])


# ---------------------------------------------------------------------------
# 4. FREQUENCY TRANSFORMS (tools/ts_dsp/frequency.py)
#    fft, psd: consume a time-domain trace, return a one-sided spectrum
#    (x becomes frequency in Hz). Per tool: numeric correctness on known
#    sinusoids, then param edge cases, then the domain gating that governs
#    what may follow a spectrum in a pipeline.
# ---------------------------------------------------------------------------

# -- fft: correctness --
def test_fft_peak_height_equals_sinusoid_amplitude():
    fs = 256.0
    t = np.arange(int(2 * fs)) / fs
    y = 3.0 * np.sin(2 * np.pi * 8.0 * t)              # amplitude 3 at 8 Hz (exact bin)
    out = apply_dsp_pipeline(_ts(y, fs=fs, y_unit="µV"), [{"tool": "fft", "params": {}}])
    peak = int(np.argmax(out.y))
    assert out.t[peak] == pytest.approx(8.0)           # peak sits at the tone's frequency
    assert out.y[peak] == pytest.approx(3.0)           # scaled so peak height = amplitude
    assert out.y.size == t.size // 2 + 1               # one-sided spectrum
    assert out.x_domain == "frequency" and out.x_unit == "Hz"
    assert out.y_domain == "magnitude" and out.y_unit == "µV"


def test_fft_of_constant_is_pure_dc():
    # The DC bin must not get the one-sided ×2 that the other bins get.
    out = apply_dsp_pipeline(_ts(np.full(256, 2.0)), [{"tool": "fft", "params": {}}])
    assert out.y[0] == pytest.approx(2.0)
    assert np.all(out.y[1:] < 1e-9)                    # no spurious content elsewhere


def test_fft_resets_clock_ticks_to_numeric():
    # "clock" tick style only makes sense on a seconds-since-midnight time
    # axis; a spectrum must come back with numeric ticks.
    sig = replace(_ts(np.ones(256)), x_tick_style="clock")
    out = apply_dsp_pipeline(sig, [{"tool": "fft", "params": {}}])
    assert out.x_tick_style == "numeric"


def test_fft_too_few_samples_raises():
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(1)), [{"tool": "fft", "params": {}}])


# -- psd: correctness + win_size edge cases --
def test_psd_peaks_at_tone_frequency():
    fs = 256.0
    t = np.arange(int(4 * fs)) / fs
    y = np.sin(2 * np.pi * 20.0 * t)
    out = apply_dsp_pipeline(_ts(y, fs=fs, y_unit="µV"), [{"tool": "psd", "params": {}}])
    assert out.t[int(np.argmax(out.y))] == pytest.approx(20.0)
    assert out.x_domain == "frequency" and out.x_unit == "Hz"
    assert out.y_domain == "power" and out.y_unit == "µV²/Hz"


def test_psd_default_win_size_caps_at_short_signal():
    # Default is 1-s Welch segments capped at the signal length, so a 0.25-s
    # signal must run with the shorter segment, not raise.
    out = apply_dsp_pipeline(_ts(np.ones(64), fs=256.0), [{"tool": "psd", "params": {}}])
    assert out.x_domain == "frequency" and out.y.size > 0


def test_psd_win_size_longer_than_signal_raises():
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(64), fs=256.0),   # 0.25-s signal
                           [{"tool": "psd", "params": {"win_size": 10}}])


def test_psd_win_size_too_short_raises():
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(256), fs=256.0),
                           [{"tool": "psd", "params": {"win_size": 0.001}}])  # < 2 samples


# -- pipeline gating after a spectrum --
def test_fft_then_time_domain_filter_raises():
    # Filters require x_domain == "time"; a spectrum must be rejected.
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(256)), [
            {"tool": "fft", "params": {}},
            {"tool": "highpass_filter", "params": {"cutoff": 10}},
        ])


def test_fft_then_fft_raises():
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(256)), [
            {"tool": "fft", "params": {}},
            {"tool": "fft", "params": {}},
        ])


def test_fft_then_smoother_runs():
    # Domain-agnostic smoothers may follow a spectrum and must preserve it.
    fs = 256.0
    t = np.arange(int(2 * fs)) / fs
    out = apply_dsp_pipeline(_ts(np.sin(2 * np.pi * 8.0 * t), fs=fs), [
        {"tool": "fft", "params": {}},
        {"tool": "moving_average", "params": {"win_size": 5}},
    ])
    assert out.x_domain == "frequency"
    assert out.y.size == t.size // 2 + 1


# ---------------------------------------------------------------------------
# 5. FEATURES (tools/ts_dsp/feature_extraction.py)
#    energy, power, rms, zcr, line_length, kurtosis, skewness: sliding-window
#    features that change the y-domain/unit and downsample to one value per
#    window. All share the _windowed() skeleton, so the window mechanics
#    (centre placement, param validation) are tested once at the end of the
#    section; per tool there is one numeric-correctness test on a signal
#    whose feature value is known exactly.
# ---------------------------------------------------------------------------

# -- energy: correctness + window-fits-signal guard --
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


# -- power: mean squared amplitude --
def test_power_of_constant_is_its_square():
    out = apply_dsp_pipeline(
        _ts(np.full(512, 2.0), fs=256.0, y_unit="µV"),
        [{"tool": "power", "params": {"window_size": 0.5, "window_stride": 0.5}}])
    assert out.y.size == 4 and np.allclose(out.y, 4.0)   # mean(2²) = 4 per window
    assert out.y_domain == "power" and out.y_unit == "µV²"


# -- rms: root mean square --
def test_rms_of_constant_equals_it():
    out = apply_dsp_pipeline(
        _ts(np.full(512, 2.0), fs=256.0, y_unit="µV"),
        [{"tool": "rms", "params": {"window_size": 0.5, "window_stride": 0.5}}])
    assert np.allclose(out.y, 2.0)                       # RMS of a constant is the constant
    assert out.y_domain == "rms" and out.y_unit == "µV"  # an envelope keeps the input unit


# -- zcr: zero-crossing rate --
def test_zcr_of_sine_is_twice_its_frequency():
    # A sinusoid at f Hz crosses zero 2f times per second. The small phase
    # offset keeps the zeros between samples so the count is unambiguous.
    fs, f = 256.0, 10.0
    t = np.arange(int(2 * fs)) / fs
    y = np.sin(2 * np.pi * f * t + 0.1)
    out = apply_dsp_pipeline(_ts(y, fs=fs),
                             [{"tool": "zcr", "params": {"window_size": 1, "window_stride": 1}}])
    assert np.allclose(out.y, 2 * f, atol=2)             # ≈ 20 crossings/s
    assert out.y_domain == "zcr" and out.y_unit == "1/s"


# -- line_length: summed absolute sample-to-sample change --
def test_line_length_of_ramp_is_exact():
    # A unit-step ramp changes by exactly 1 per sample: a 32-sample window
    # has 31 steps, so line length = 31.
    out = apply_dsp_pipeline(
        _ts(np.arange(64, dtype=float), fs=256.0, y_unit="µV"),
        [{"tool": "line_length", "params": {"window_size": 0.125, "window_stride": 0.125}}])
    assert out.y.size == 2 and np.allclose(out.y, 31.0)
    assert out.y_domain == "line_length" and out.y_unit == "µV"


# -- kurtosis / skewness: known values for a sinusoid --
def test_kurtosis_of_sine_is_minus_1_5():
    # Excess kurtosis of a sinusoid is exactly -1.5: the normalized 4th
    # moment E[sin⁴]/E[sin²]² is 1.5, and Fisher's definition subtracts 3.
    # Full cycles per window make the sampled moments exact.
    fs = 256.0
    t = np.arange(int(2 * fs)) / fs
    y = np.sin(2 * np.pi * 8.0 * t)
    out = apply_dsp_pipeline(_ts(y, fs=fs),
                             [{"tool": "kurtosis",
                               "params": {"window_size": 1, "window_stride": 1}}])
    assert np.allclose(out.y, -1.5)
    assert out.y_domain == "kurtosis" and out.y_unit == ""   # dimensionless


def test_skewness_of_sine_is_zero():
    fs = 256.0
    t = np.arange(int(2 * fs)) / fs
    y = np.sin(2 * np.pi * 8.0 * t)                      # symmetric about 0
    out = apply_dsp_pipeline(_ts(y, fs=fs),
                             [{"tool": "skewness",
                               "params": {"window_size": 1, "window_stride": 1}}])
    assert np.allclose(out.y, 0.0, atol=1e-9)
    assert out.y_domain == "skewness" and out.y_unit == ""   # dimensionless


# -- shared _windowed() skeleton: centre placement + param validation --
def test_feature_values_sit_at_window_centres():
    # 0.5-s windows with 0.5-s stride over 2 s: centres at 0.25/0.75/1.25/1.75.
    out = apply_dsp_pipeline(
        _ts(np.ones(512), fs=256.0),
        [{"tool": "rms", "params": {"window_size": 0.5, "window_stride": 0.5}}])
    assert np.allclose(out.t, [0.25, 0.75, 1.25, 1.75])


@pytest.mark.parametrize("tool", [
    "energy", "power", "rms", "zcr", "line_length", "kurtosis", "skewness",
])
def test_feature_nonpositive_window_size_raises(tool):
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(256)),
                           [{"tool": tool,
                             "params": {"window_size": 0, "window_stride": 0.5}}])


def test_feature_nonpositive_window_stride_raises():
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(256)),
                           [{"tool": "rms",
                             "params": {"window_size": 0.5, "window_stride": 0}}])


def test_feature_after_feature_raises():
    # Features require y_domain == "amplitude": once energy has changed the
    # y-domain, a second feature must be rejected by the registry gate.
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(512)), [
            {"tool": "energy", "params": {"window_size": 0.5, "window_stride": 0.5}},
            {"tool": "rms", "params": {"window_size": 0.5, "window_stride": 0.5}},
        ])


# ---------------------------------------------------------------------------
# 6. COMPOSITION
#    Multi-step pipelines: domains propagate step to step, the gate applies
#    to each step's *incoming* signal (not the pipeline's original input),
#    and per-step validation sees the signal as reshaped by earlier steps.
# ---------------------------------------------------------------------------
def test_filter_then_energy_composes():
    fs = 256.0
    t = np.arange(int(2 * fs)) / fs
    y = 50.0 + np.sin(2 * np.pi * 40.0 * t)
    out = apply_dsp_pipeline(_ts(y, fs=fs), [
        {"tool": "highpass_filter", "params": {"cutoff": 10}},
        {"tool": "energy", "params": {"window_size": 0.25, "window_stride": 0.25}},
    ])
    assert out.y_domain == "energy" and out.y.size > 0


def test_filters_chain_freely_among_themselves():
    # Filters keep x=time / y=amplitude, so any filter may follow any other.
    fs = 256.0
    t = np.arange(int(2 * fs)) / fs
    y = np.sin(2 * np.pi * 20.0 * t)
    out = apply_dsp_pipeline(_ts(y, fs=fs), [
        {"tool": "highpass_filter", "params": {"cutoff": 5}},
        {"tool": "notch_filter", "params": {"w0": 60}},
        {"tool": "lowpass_filter", "params": {"cutoff": 40}},
    ])
    assert out.x_domain == "time" and out.y_domain == "amplitude"
    assert out.y.size == t.size                        # filters never resample


def test_filter_smoother_feature_chain_propagates_domains():
    fs = 256.0
    t = np.arange(int(2 * fs)) / fs
    y = 50.0 + np.sin(2 * np.pi * 20.0 * t)
    out = apply_dsp_pipeline(_ts(y, fs=fs, y_unit="µV"), [
        {"tool": "highpass_filter", "params": {"cutoff": 5}},
        {"tool": "moving_average", "params": {"win_size": 5}},
        {"tool": "rms", "params": {"window_size": 0.5, "window_stride": 0.5}},
    ])
    assert out.y_domain == "rms" and out.y_unit == "µV"
    assert out.y.size == 4                             # windowed down by the feature


def test_gate_rejects_mid_chain_not_just_first_step():
    # Step 2 (smoother) legally follows the fft; step 3 must still be gated
    # on the *spectrum* it receives, not on the pipeline's time-domain input.
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(512)), [
            {"tool": "fft", "params": {}},
            {"tool": "moving_average", "params": {"win_size": 5}},
            {"tool": "lowpass_filter", "params": {"cutoff": 30}},
        ])


def test_feature_output_rejects_filter():
    # energy flips y_domain to "energy", so an amplitude-gated filter must be
    # rejected even though the signal is still a time-series.
    with pytest.raises(ToolInputError):
        apply_dsp_pipeline(_ts(np.ones(512)), [
            {"tool": "energy", "params": {"window_size": 0.5, "window_stride": 0.5}},
            {"tool": "notch_filter", "params": {}},
        ])


def test_downsampled_output_too_short_for_next_window():
    # energy reduces 512 samples to 4 window values; a win_size-10 smoother
    # must then fail against the 4-sample series it actually receives — and
    # the error must name the step that failed.
    with pytest.raises(ToolInputError) as excinfo:
        apply_dsp_pipeline(_ts(np.ones(512), fs=256.0), [
            {"tool": "energy", "params": {"window_size": 0.5, "window_stride": 0.5}},
            {"tool": "moving_average", "params": {"win_size": 10}},
        ])
    assert "moving_average" in str(excinfo.value)      # error points at the failing step
    assert "4" in str(excinfo.value)                   # ...and at the post-energy length
