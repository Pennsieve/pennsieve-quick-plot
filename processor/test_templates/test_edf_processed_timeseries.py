"""
Tests for the edf_processed_timeseries template.

To run:
Requires pyedflib + numpy + matplotlib + pytest to be importable
pip install -r requirements-dev.txt
pytest processor/test_templates/test_edf_processed_timeseries.py

EDF is a binary format, so instead of a text fixture (as the TSV tests use)
these tests synthesize a tiny, deterministic EDF with pyedflib.EdfWriter via
`_write_edf`. Each channel is a fixed sine wave, so reads are reproducible and
a montage (channel - channel2) has a known value.
"""

# SET UP
from __future__ import annotations
from datetime import datetime

import numpy as np
import pyedflib
import pytest
import matplotlib.axes
from processor.templates import edf_processed_timeseries as template

# Anchor clock for the synthetic recording (matches how a real EDF carries a
# start-of-day time); clock-string window inputs resolve against this.
_START = datetime(2020, 9, 24, 13, 12, 36)


def _write_edf(path, labels=("F7", "F8", "FP1", "EKG2"), fs=256, seconds=20,
               freqs=None, dimension="uV", start=_START) -> str:
    """Write a small deterministic EDF and return its path as a string.

    Each channel i is amplitude 100*(i+1) sine at (i+1) Hz, so channels differ
    and a montage difference is non-zero. `freqs` overrides the per-channel
    sample rate (for the montage rate-mismatch test).
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


# TEST FOR GOOD PATH AND INPUT
# 1. single channel, numeric window -> produce a PNG
def test_render_writes_a_png(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    out = tmp_path / "figure.png"

    template.render(src, str(out), channel="F7", **_OK)

    assert out.is_file()
    assert out.read_bytes().startswith(b"\x89PNG")  # PNG file signature


# 2. clock-string window needs no time_unit and still produces a PNG
def test_render_writes_a_png_from_clock_window(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    out = tmp_path / "figure.png"

    template.render(src, str(out), channel="F7",
                    start_time="13:12:36", end_time="13:12:38",
                    y_range=100, y_unit="uV")  # no time_unit on purpose

    assert out.read_bytes().startswith(b"\x89PNG")


# 3. montage (two channels) produces a PNG
def test_render_writes_a_png_for_montage(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    out = tmp_path / "figure.png"

    template.render(src, str(out), channel="F7", channel2="F8", **_OK)

    assert out.read_bytes().startswith(b"\x89PNG")


# 4. channel match is case-insensitive ("Fp1" resolves to file label "FP1")
def test_render_channel_match_is_case_insensitive(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    out = tmp_path / "figure.png"

    template.render(src, str(out), channel="Fp1", **_OK)

    assert out.read_bytes().startswith(b"\x89PNG")


# 5. explicit [min, max] y-range is accepted
def test_render_accepts_explicit_y_range_pair(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    out = tmp_path / "figure.png"

    template.render(src, str(out), channel="F7",
                    start_time=0, duration=2, time_unit="s",
                    y_range=[-100, 115], y_unit="uV")

    assert out.read_bytes().startswith(b"\x89PNG")


# TEST FOR BAD FILE / FORMAT
# 1. a non-EDF extension is rejected
def test_render_raises_on_non_edf_extension(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("not an edf")
    with pytest.raises(RuntimeError):
        template.render(str(bad), str(tmp_path / "figure.png"), channel="F7", **_OK)


# 2. a missing input file is rejected
def test_render_raises_when_file_missing(tmp_path):
    with pytest.raises(RuntimeError):
        template.render(str(tmp_path / "nope.edf"), str(tmp_path / "figure.png"),
                        channel="F7", **_OK)


# 3. a channel not in the file is rejected
def test_render_raises_on_unknown_channel(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="ZZ9", **_OK)


# TEST FOR MISSING / INCONSISTENT REQUIRED INPUTS
# 1. missing channel
def test_render_raises_when_channel_missing(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), **_OK)


# 2. missing start_time
def test_render_raises_when_start_missing(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        duration=2, time_unit="s", y_range=100, y_unit="uV")


# 3. neither end_time nor duration
def test_render_raises_when_no_end_or_duration(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, time_unit="s", y_range=100, y_unit="uV")


# 4. end_time and duration given but inconsistent
def test_render_raises_when_end_and_duration_disagree(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, end_time=5, duration=3, time_unit="s",
                        y_range=100, y_unit="uV")


# TEST FOR UNIT HANDLING (no silent defaults)
# 1. missing y_unit
def test_render_raises_when_y_unit_missing(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, duration=2, time_unit="s", y_range=100)


# 2. missing time_unit when a numeric time is given
def test_render_raises_when_time_unit_missing_for_numeric(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, duration=2, y_range=100, y_unit="uV")


# 3. a non-time time_unit
def test_render_raises_on_bad_time_unit(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, duration=2, time_unit="parsecs",
                        y_range=100, y_unit="uV")


# 4. a non-voltage y_unit
def test_render_raises_on_bad_y_unit(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, duration=2, time_unit="s",
                        y_range=100, y_unit="amperes")


# 5. missing y_range
def test_render_raises_when_y_range_missing(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, duration=2, time_unit="s", y_unit="uV")


# 6. an explicit y-range whose min is not below its max
def test_render_raises_when_y_range_min_not_below_max(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, duration=2, time_unit="s",
                        y_range=[10, 2], y_unit="uV")


# TEST FOR TIME-WINDOW VALIDATION
# 1. start at/after the end of the recording (recording is 20 s)
def test_render_raises_when_start_past_recording(tmp_path):
    src = _write_edf(tmp_path / "rec.edf", seconds=20)
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=25, duration=1, time_unit="s",
                        y_range=100, y_unit="uV")


# 2. computed end past the end of the recording
def test_render_raises_when_end_past_recording(tmp_path):
    src = _write_edf(tmp_path / "rec.edf", seconds=20)
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, duration=25, time_unit="s",
                        y_range=100, y_unit="uV")


# 3. start after end
def test_render_raises_when_start_after_end(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=10, end_time=5, time_unit="s",
                        y_range=100, y_unit="uV")


# 4. computed duration over MAX_DURATION_S (needs a >600 s recording)
def test_render_raises_when_duration_over_max(tmp_path):
    src = _write_edf(tmp_path / "long.edf", labels=("F7", "F8"), fs=32, seconds=650)
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, duration=template.MAX_DURATION_S + 5,
                        time_unit="s", y_range=100, y_unit="uV")


# 5. a window too short to yield MIN_SAMPLES samples
def test_render_raises_when_window_too_short(tmp_path):
    src = _write_edf(tmp_path / "rec.edf", fs=256)
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"), channel="F7",
                        start_time=0, duration=0.01, time_unit="s",
                        y_range=100, y_unit="uV")  # ~3 samples < MIN_SAMPLES


# TEST FOR MONTAGE-SPECIFIC VALIDATION
# 1. the two channels are the same
def test_render_raises_on_montage_same_channel(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"),
                        channel="F7", channel2="f7", **_OK)


# 2. the second channel is not in the file
def test_render_raises_on_montage_unknown_channel2(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"),
                        channel="F7", channel2="ZZ9", **_OK)


# 3. the two channels have different sampling rates
def test_render_raises_on_montage_rate_mismatch(tmp_path):
    src = _write_edf(tmp_path / "mix.edf", labels=("F7", "F8"),
                     freqs=[256, 128], seconds=10)
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"),
                        channel="F7", channel2="F8", **_OK)


# PLOTTING CHECKS
# 1. a montage plots channel - channel2, sample by sample.
def test_render_montage_plots_the_difference(monkeypatch, tmp_path):
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


# 2. the title noun follows the mode: "channel" for one, "montage" for two.
def test_render_title_reflects_mode(monkeypatch, tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    titles = []
    original_set_title = matplotlib.axes.Axes.set_title

    def spy_set_title(self, label, *args, **kwargs):
        titles.append(label)
        return original_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", spy_set_title)

    template.render(src, str(tmp_path / "s.png"), channel="F7", **_OK)
    template.render(src, str(tmp_path / "m.png"), channel="F7", channel2="F8", **_OK)

    assert titles[0] == "Processed timeseries for channel F7"
    assert titles[1] == "Processed timeseries for montage F7-F8"


# 3. the x-axis label mirrors the input format: seconds, microseconds, clock.
def test_render_x_label_follows_input_format(monkeypatch, tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    labels = []
    original_set_xlabel = matplotlib.axes.Axes.set_xlabel

    def spy_set_xlabel(self, xlabel, *args, **kwargs):
        labels.append(xlabel)
        return original_set_xlabel(self, xlabel, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_xlabel", spy_set_xlabel)

    template.render(src, str(tmp_path / "s.png"), channel="F7",
                    start_time=0, duration=2, time_unit="s", y_range=100, y_unit="uV")
    template.render(src, str(tmp_path / "u.png"), channel="F7",
                    start_time=0, end_time=1_000_000, time_unit="usec",
                    y_range=100, y_unit="uV")
    template.render(src, str(tmp_path / "c.png"), channel="F7",
                    start_time="13:12:36", end_time="13:12:38",
                    y_range=100, y_unit="uV")

    usym = template._TIME_SYMBOL[1e-6]  # canonical microsecond symbol
    assert labels == ["time (s)", f"time ({usym})", "time (h:m:s)"]
