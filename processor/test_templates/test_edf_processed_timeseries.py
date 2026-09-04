"""
Tests for the edf_processed_timeseries template.

Layout (one banner section per stage of render()'s pipeline):
  1. HAPPY PATHS              — valid inputs -> a PNG on disk
  2. FILE / FORMAT ERRORS     — wrong extension, missing file, unknown channel
  3. DATA-SELECTION INPUTS    — required channel/time args, missing or inconsistent
  4. DISPLAY SETTINGS         — y_unit / y_range: header defaults + validation
  5. TIME-WINDOW VALIDATION   — window vs. recording length and size limits
  6. MONTAGE VALIDATION       — two-channel (bipolar) specific rules
  7. PLOT CONTENT             — spies on matplotlib: data, title, axis labels
  8. DSP PIPELINES            — end-to-end pipelines through render()
  9. SPECTRUM PIPELINES       — fft/psd move the x-axis off the time domain

EDF is a binary format, so instead of a text fixture (as the TSV tests use)
these tests synthesize a tiny, deterministic EDF with pyedflib.EdfWriter via
`_write_edf`. Each channel is a fixed sine wave, so reads are reproducible and
a montage (channel - channel2) has a known value.

Plot-content tests use `_spy_first_arg` / `_spy_ylim` to wrap a matplotlib
Axes method and record what render() passes to it — asserting on the figure's
*content* without parsing the PNG.

To run:
Requires pyedflib + numpy + matplotlib + pytest to be importable
pip install -r requirements-dev.txt
pytest processor/test_templates/test_edf_processed_timeseries.py
"""

# SET UP
from __future__ import annotations
from datetime import datetime

import numpy as np
import pyedflib
import pytest
import matplotlib.axes
from processor.templates import edf_processed_timeseries as template
from processor.tools.ts_dsp import resolve_time_unit

# Anchor clock for the synthetic recording (matches how a real EDF carries a
# start-of-day time); clock-string window inputs resolve against this.
_START = datetime(2020, 9, 24, 13, 12, 36)


def _write_edf(path, labels=("F7", "F8", "FP1", "EKG2"), fs=256, seconds=20,
               freqs=None, dimension="uV", start=_START) -> str:
    """Write a small deterministic EDF and return its path as a string.

    Each channel i is amplitude 100*(i+1) sine at (i+1) Hz, so channels differ
    and a montage difference is non-zero. `freqs` overrides the per-channel
    sample rate (for the montage rate-mismatch test); `dimension` overrides
    the declared physical unit (for the header-unit tests).
    """
    n = len(labels)
    if freqs is None:
        freqs = [fs] * n
    writer = pyedflib.EdfWriter(str(path), n, file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        headers = [
            {
                "label": lab, "dimension": dimension, "sample_frequency": f,
                "physical_max": 5000.0, "physical_min": -5000.0,
                "digital_max": 32767, "digital_min": -32768,
                "transducer": "", "prefilter": "",
            }
            for lab, f in zip(labels, freqs)
        ]
        writer.setSignalHeaders(headers)
        writer.setStartdatetime(start)
        data = []
        for i, f in enumerate(freqs):
            t = np.arange(int(f * seconds)) / float(f)
            data.append((100.0 * (i + 1)) * np.sin(2 * np.pi * (i + 1) * t))
        writer.writeSamples(data)
    finally:
        writer.close()
        del writer
    return str(path)


# Keyword args shared by many happy-path calls (a valid numeric window).
_OK = dict(start_time=0, duration=2, time_unit="s", y_range=100, y_unit="uV")


def _spy_first_arg(monkeypatch, method: str) -> list:
    """Wrap Axes.<method> to record its first positional arg on every call.

    Returns the (initially empty) list the spy appends to. Works for the
    label/title setters, whose first arg is the string we assert on.
    """
    seen: list = []
    original = getattr(matplotlib.axes.Axes, method)

    def spy(self, value, *args, **kwargs):
        seen.append(value)
        return original(self, value, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, method, spy)
    return seen


def _spy_ylim(monkeypatch) -> list:
    """Wrap Axes.set_ylim to record every call as a flat (lo, hi) tuple.

    set_ylim may be called as set_ylim((lo, hi)) or set_ylim(lo, hi); both
    are normalised so tests can assert `(lo, hi) in calls`.
    """
    calls: list = []
    original = matplotlib.axes.Axes.set_ylim

    def spy(self, *args, **kwargs):
        flat = tuple(args[0]) if len(args) == 1 and np.iterable(args[0]) else args
        calls.append(flat)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylim", spy)
    return calls


# ---------------------------------------------------------------------------
# 1. HAPPY PATHS
#    Valid inputs produce a PNG on disk, across the supported input styles
#    (numeric window, clock-string window, montage, y_range forms).
# ---------------------------------------------------------------------------
def test_render_writes_a_png(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    out = tmp_path / "figure.png"

    template.render(src, str(out), channel="F7", **_OK)

    assert out.is_file()
    assert out.read_bytes().startswith(b"\x89PNG")  # PNG file signature


def test_render_writes_a_png_from_clock_window(tmp_path):
    # Clock-string times carry their own unit, so no time_unit is needed.
    src = _write_edf(tmp_path / "rec.edf")
    out = tmp_path / "figure.png"

    template.render(src, str(out), channel="F7",
                    start_time="13:12:36", end_time="13:12:38",
                    y_range=100, y_unit="uV")  # no time_unit on purpose

    assert out.read_bytes().startswith(b"\x89PNG")


def test_render_writes_a_png_for_montage(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    out = tmp_path / "figure.png"

    template.render(src, str(out), channel="F7", channel2="F8", **_OK)

    assert out.read_bytes().startswith(b"\x89PNG")


def test_render_channel_match_is_case_insensitive(tmp_path):
    # "Fp1" resolves to the file's label "FP1".
    src = _write_edf(tmp_path / "rec.edf")
    out = tmp_path / "figure.png"

    template.render(src, str(out), channel="Fp1", **_OK)

    assert out.read_bytes().startswith(b"\x89PNG")


def test_render_accepts_explicit_y_range_pair(tmp_path):
    # y_range's second form: an explicit [min, max] instead of ± shorthand.
    src = _write_edf(tmp_path / "rec.edf")
    out = tmp_path / "figure.png"

    template.render(src, str(out), channel="F7",
                    start_time=0, duration=2, time_unit="s",
                    y_range=[-100, 115], y_unit="uV")

    assert out.read_bytes().startswith(b"\x89PNG")


# ---------------------------------------------------------------------------
# 2. FILE / FORMAT ERRORS
#    Problems with the target file itself, caught before any samples are read.
# ---------------------------------------------------------------------------
def test_render_raises_on_non_edf_extension(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("not an edf")
    with pytest.raises(RuntimeError):
        template.render(str(bad), str(tmp_path / "figure.png"), channel="F7", **_OK)


def test_render_raises_when_file_missing(tmp_path):
    with pytest.raises(RuntimeError):
        template.render(str(tmp_path / "nope.edf"), str(tmp_path / "figure.png"),
                        channel="F7", **_OK)


def test_render_raises_on_unknown_channel(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="ZZ9", **_OK)


# ---------------------------------------------------------------------------
# 3. DATA-SELECTION INPUTS
#    channel + time window have NO defaults: missing or self-contradictory
#    values fail the render (which then falls back to the agent path).
# ---------------------------------------------------------------------------
def test_render_raises_when_channel_missing(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), **_OK)


def test_render_raises_when_start_missing(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        duration=2, time_unit="s", y_range=100, y_unit="uV")


def test_render_raises_when_no_end_or_duration(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, time_unit="s", y_range=100, y_unit="uV")


def test_render_raises_when_end_and_duration_disagree(tmp_path):
    # Both are allowed together only when they imply the same window.
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, end_time=5, duration=3, time_unit="s",
                        y_range=100, y_unit="uV")


def test_render_raises_when_time_unit_missing_for_numeric(tmp_path):
    # time_unit is required whenever a time is a plain number (unlike the
    # clock-string happy path above, which needs none).
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, duration=2, y_range=100, y_unit="uV")


def test_render_raises_on_bad_time_unit(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, duration=2, time_unit="parsecs",
                        y_range=100, y_unit="uV")


# ---------------------------------------------------------------------------
# 4. DISPLAY SETTINGS (y_unit / y_range)
#    Optional args with truthful defaults: omitted y_unit displays in the
#    unit the file's header declares; omitted y_range auto-scales. Supplied
#    values are still validated, and an unreadable header unit is fatal.
# ---------------------------------------------------------------------------

# -- y_unit: header default + validation --
def test_render_defaults_y_unit_to_file_header(monkeypatch, tmp_path):
    src = _write_edf(tmp_path / "rec.edf")   # header dimension is "uV"
    ylabels = _spy_first_arg(monkeypatch, "set_ylabel")
    out = tmp_path / "figure.png"

    template.render(src, str(out), channel="F7",
                    start_time=0, duration=2, time_unit="s", y_range=100)

    assert out.read_bytes().startswith(b"\x89PNG")
    assert ylabels[-1] == "amplitude (µV)"   # canonical symbol for header "uV"


def test_render_raises_on_unrecognized_header_unit(tmp_path):
    # No y_unit and a non-voltage header: there is no truthful display unit
    # to fall back to.
    src = _write_edf(tmp_path / "rec.edf", dimension="degC")
    with pytest.raises(RuntimeError, match="not a recognized voltage unit"):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, duration=2, time_unit="s", y_range=100)


def test_render_raises_on_unrecognized_header_unit_despite_y_unit(tmp_path):
    # ...and an explicit y_unit cannot rescue it: the read itself needs the
    # header unit to convert samples, so it fails with the same message.
    src = _write_edf(tmp_path / "rec.edf", dimension="degC")
    with pytest.raises(RuntimeError, match="not a recognized voltage unit"):
        template.render(src, str(tmp_path / "figure.png"), channel="F7", **_OK)


def test_render_raises_on_bad_y_unit(tmp_path):
    # A supplied y_unit must still be a voltage unit.
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, duration=2, time_unit="s",
                        y_range=100, y_unit="amperes")


# -- y_range: auto-scale default + validation --
def test_render_auto_scales_when_y_range_missing(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    out = tmp_path / "figure.png"

    template.render(src, str(out), channel="F7",
                    start_time=0, duration=2, time_unit="s", y_unit="uV")

    assert out.read_bytes().startswith(b"\x89PNG")


def test_render_y_range_without_y_unit_uses_header_unit(monkeypatch, tmp_path):
    # A y_range given alone is interpreted in the header's unit.
    src = _write_edf(tmp_path / "rec.edf")   # header dimension is "uV"
    ylim_calls = _spy_ylim(monkeypatch)

    template.render(src, str(tmp_path / "figure.png"), channel="F7",
                    start_time=0, duration=2, time_unit="s", y_range=100)

    assert (-100.0, 100.0) in ylim_calls     # ±100 in the header's µV


def test_render_raises_when_y_range_min_not_below_max(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, duration=2, time_unit="s",
                        y_range=[10, 2], y_unit="uV")


# ---------------------------------------------------------------------------
# 5. TIME-WINDOW VALIDATION
#    The window must lie inside the recording and respect the size limits
#    (MAX_DURATION_S at the top, MIN_SAMPLES at the bottom).
# ---------------------------------------------------------------------------
def test_render_raises_when_start_past_recording(tmp_path):
    src = _write_edf(tmp_path / "rec.edf", seconds=20)
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=25, duration=1, time_unit="s",
                        y_range=100, y_unit="uV")


def test_render_raises_when_end_past_recording(tmp_path):
    src = _write_edf(tmp_path / "rec.edf", seconds=20)
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, duration=25, time_unit="s",
                        y_range=100, y_unit="uV")


def test_render_raises_when_start_after_end(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=10, end_time=5, time_unit="s",
                        y_range=100, y_unit="uV")


def test_render_raises_when_duration_over_max(tmp_path):
    # Needs a recording longer than MAX_DURATION_S so only the cap can fail.
    src = _write_edf(tmp_path / "long.edf", labels=("F7", "F8"), fs=32, seconds=650)
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, duration=template.MAX_DURATION_S + 5,
                        time_unit="s", y_range=100, y_unit="uV")


def test_render_raises_when_window_too_short(tmp_path):
    src = _write_edf(tmp_path / "rec.edf", fs=256)
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, duration=0.01, time_unit="s",
                        y_range=100, y_unit="uV")  # ~3 samples < MIN_SAMPLES


# ---------------------------------------------------------------------------
# 6. MONTAGE VALIDATION
#    Rules specific to the two-channel (bipolar A-B) mode.
# ---------------------------------------------------------------------------
def test_render_raises_on_montage_same_channel(tmp_path):
    # Compared case-insensitively: F7 vs f7 is still the same channel.
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"),
                        channel="F7", channel2="f7", **_OK)


def test_render_raises_on_montage_unknown_channel2(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"),
                        channel="F7", channel2="ZZ9", **_OK)


def test_render_raises_on_montage_rate_mismatch(tmp_path):
    src = _write_edf(tmp_path / "mix.edf", labels=("F7", "F8"),
                     freqs=[256, 128], seconds=10)
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"),
                        channel="F7", channel2="F8", **_OK)


# ---------------------------------------------------------------------------
# 7. PLOT CONTENT
#    Spies on matplotlib Axes methods assert what actually lands in the
#    figure: the plotted samples, the title, and the axis labels.
# ---------------------------------------------------------------------------
def test_render_montage_plots_the_difference(monkeypatch, tmp_path):
    # A montage plots channel - channel2, sample by sample.
    src = _write_edf(tmp_path / "rec.edf")
    captured = {}
    original_plot = matplotlib.axes.Axes.plot

    def spy_plot(self, *args, **kwargs):
        # _plot_timeseries calls ax.plot(t, y, ...); grab y.
        if len(args) >= 2:
            captured["y"] = np.asarray(args[1], dtype=float).copy()
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy_plot)

    template.render(src, str(tmp_path / "a.png"), channel="F7", **_OK)
    y_a = captured["y"]
    template.render(src, str(tmp_path / "b.png"), channel="F8", **_OK)
    y_b = captured["y"]
    template.render(src, str(tmp_path / "m.png"), channel="F7", channel2="F8", **_OK)
    y_m = captured["y"]

    assert np.allclose(y_m, y_a - y_b)


def test_render_title_reflects_mode(monkeypatch, tmp_path):
    # The title noun follows the mode: "channel" for one, "montage" for two.
    src = _write_edf(tmp_path / "rec.edf")
    titles = _spy_first_arg(monkeypatch, "set_title")

    template.render(src, str(tmp_path / "s.png"), channel="F7", **_OK)
    template.render(src, str(tmp_path / "m.png"), channel="F7", channel2="F8", **_OK)

    assert titles[0] == "Processed timeseries for channel F7"
    assert titles[1] == "Processed timeseries for montage F7-F8"


def test_render_x_label_follows_input_format(monkeypatch, tmp_path):
    # The x-axis label mirrors the input format: seconds, microseconds, clock.
    src = _write_edf(tmp_path / "rec.edf")
    labels = _spy_first_arg(monkeypatch, "set_xlabel")

    template.render(src, str(tmp_path / "s.png"), channel="F7",
                    start_time=0, duration=2, time_unit="s", y_range=100, y_unit="uV")
    template.render(src, str(tmp_path / "u.png"), channel="F7",
                    start_time=0, end_time=1_000_000, time_unit="usec",
                    y_range=100, y_unit="uV")
    template.render(src, str(tmp_path / "c.png"), channel="F7",
                    start_time="13:12:36", end_time="13:12:38",
                    y_range=100, y_unit="uV")

    # canonical microsecond symbol, from the ts_dsp unit vocabulary
    usym = resolve_time_unit("usec", required=True)[1]
    assert labels == ["time (s)", f"time ({usym})", "time (h:m:s)"]


# ---------------------------------------------------------------------------
# 8. DSP PIPELINES (end-to-end through render)
#    The optional `pipeline` arg runs ts_dsp tools between read and plot;
#    tool errors surface as RuntimeError (ToolInputError subclasses it).
# ---------------------------------------------------------------------------
def test_render_pipeline_highpass_writes_a_png(tmp_path):
    # A filter pipeline still produces a PNG; the trace stays voltage.
    src = _write_edf(tmp_path / "rec.edf")
    out = tmp_path / "figure.png"
    template.render(src, str(out), channel="F7",
                    start_time=0, duration=5, time_unit="s", y_range=200, y_unit="uV",
                    pipeline=[{"tool": "highpass_filter", "params": {"cutoff": 1.0}}])
    assert out.read_bytes().startswith(b"\x89PNG")


def test_render_pipeline_energy_relabels_y_axis(monkeypatch, tmp_path):
    # A feature pipeline (energy) relabels the y-axis to its domain/unit.
    src = _write_edf(tmp_path / "rec.edf")
    ylabels = _spy_first_arg(monkeypatch, "set_ylabel")

    template.render(src, str(tmp_path / "figure.png"), channel="F7",
                    start_time=0, duration=5, time_unit="s", y_range=200, y_unit="uV",
                    pipeline=[{"tool": "energy",
                               "params": {"window_size": 0.5, "window_stride": 0.5}}])

    assert ylabels[-1] == "energy (µV²·s)"


def test_render_pipeline_unknown_tool_raises(tmp_path):
    # An invalid pipeline raises — surfaced, not silently ignored.
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):   # ToolInputError subclasses RuntimeError
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, duration=2, time_unit="s", y_range=100, y_unit="uV",
                        pipeline=[{"tool": "no_such_tool"}])


# ---------------------------------------------------------------------------
# 9. SPECTRUM PIPELINES
#    fft/psd move the x-axis off the time domain: both axes relabel and the
#    voltage y_range disengages (the y-axis auto-scales instead).
# ---------------------------------------------------------------------------
def test_render_pipeline_psd_labels_both_axes(monkeypatch, tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    xlabels = _spy_first_arg(monkeypatch, "set_xlabel")
    ylabels = _spy_first_arg(monkeypatch, "set_ylabel")

    template.render(src, str(tmp_path / "figure.png"), channel="F7",
                    start_time=0, duration=5, time_unit="s", y_range=200, y_unit="uV",
                    pipeline=[{"tool": "psd", "params": {}}])

    assert xlabels[-1] == "frequency (Hz)"
    assert ylabels[-1] == "power (µV²/Hz)"


def test_render_pipeline_psd_does_not_apply_y_range(monkeypatch, tmp_path):
    # A spectrum auto-scales the y-axis: a supplied voltage y_range must NOT
    # be applied (power/Hz is not voltage).
    src = _write_edf(tmp_path / "rec.edf")
    ylim_calls = _spy_ylim(monkeypatch)

    # Positive control: a raw trace DOES clamp y to the requested ±100 µV.
    template.render(src, str(tmp_path / "raw.png"), channel="F7", **_OK)
    assert (-100.0, 100.0) in ylim_calls

    # A psd pipeline must not receive that voltage range.
    ylim_calls.clear()
    template.render(src, str(tmp_path / "psd.png"), channel="F7",
                    start_time=0, duration=5, time_unit="s", y_range=100, y_unit="uV",
                    pipeline=[{"tool": "psd", "params": {}}])
    assert (-100.0, 100.0) not in ylim_calls


def test_render_pipeline_fft_writes_a_png_with_magnitude_axis(monkeypatch, tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    ylabels = _spy_first_arg(monkeypatch, "set_ylabel")
    out = tmp_path / "figure.png"

    template.render(src, str(out), channel="F7",
                    start_time=0, duration=5, time_unit="s", y_range=200, y_unit="uV",
                    pipeline=[{"tool": "fft", "params": {}}])

    assert out.read_bytes().startswith(b"\x89PNG")
    assert ylabels[-1] == "magnitude (µV)"
