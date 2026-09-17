"""
Tests for processor.readers — the file -> Signal step on its own.

The template tests (test_templates/test_edf_processed_timeseries.py) drive
the whole render() path end to end. These tests pin the reader's own
contract at its seams, so a change in how the file is read is caught here
rather than as a mysterious template failure:

  1. DISPATCH          load_signal() picks the reader by extension
  2. HEADER            _read_header: metadata only, case-insensitive channels
  3. WINDOW vs HEADER  _check_against_header: clock anchoring, limits, units
  4. THE SIGNAL        what comes out: seconds from recording start, display
                       voltage, montage difference, recording_start_clock_s

The synthetic EDF is written the same way as in the template tests (fixed
sine per channel), so values are predictable.

To run:
pip install -r requirements-dev.txt
pytest processor/test_readers
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pyedflib
import pytest

from processor import readers
from processor.readers import edf_to_signal as reader
from processor.templates.edf_processed_timeseries import RenderParams, TimePoint

_START = datetime(2020, 9, 24, 13, 12, 36)
_START_CLOCK_S = 13 * 3600 + 12 * 60 + 36


def _write_edf(path, labels=("F7", "F8", "FP1"), fs=256, seconds=20,
               freqs=None, dimension="uV", start=_START) -> str:
    n = len(labels)
    if freqs is None:
        freqs = [fs] * n
    writer = pyedflib.EdfWriter(str(path), n, file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setSignalHeaders([
            {"label": lab, "dimension": dimension, "sample_frequency": f,
             "physical_max": 5000.0, "physical_min": -5000.0,
             "digital_max": 32767, "digital_min": -32768,
             "transducer": "", "prefilter": ""}
            for lab, f in zip(labels, freqs)
        ])
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


def _params(channel="F7", channel2=None, start=TimePoint(0.0, False),
            end=None, duration_s=2.0, volt=None, clock_mode=False) -> RenderParams:
    """A valid RenderParams with the reader-relevant fields filled in."""
    return RenderParams(
        channel=channel, channel2=channel2, montage=channel2 is not None,
        start=start, end=end, duration_s=duration_s,
        time_factor=1.0, time_symbol="s", clock_mode=clock_mode,
        y_limits=None, volt=volt, pipeline=None,
    )


# ---------------------------------------------------------------------------
# 1. DISPATCH
# ---------------------------------------------------------------------------
def test_load_signal_dispatches_on_extension(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    sig = readers.load_signal(src, _params())
    assert sig.channel == "F7" and sig.x_domain == "time"


def test_load_signal_rejects_unknown_extension(tmp_path):
    (tmp_path / "rec.xyz").write_bytes(b"not a recording")
    with pytest.raises(RuntimeError, match="No reader for '.xyz'"):
        readers.load_signal(str(tmp_path / "rec.xyz"), _params())


def test_load_signal_rejects_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="does not exist"):
        readers.load_signal(str(tmp_path / "nope.edf"), _params())


def test_supported_extensions_lists_edf():
    assert ".edf" in readers.supported_extensions()


# ---------------------------------------------------------------------------
# 2. HEADER
# ---------------------------------------------------------------------------
def test_read_header_is_metadata_only(tmp_path):
    header = reader._read_header(_write_edf(tmp_path / "rec.edf", fs=128, seconds=10))
    assert header.labels == ("F7", "F8", "FP1")
    assert header.fs["F8"] == 128.0
    assert header.duration_s["F8"] == pytest.approx(10.0)
    assert header.unit["F8"] == "uV"
    assert header.start_clock_s == pytest.approx(_START_CLOCK_S)


def test_header_channel_lookup_is_case_insensitive(tmp_path):
    header = reader._read_header(_write_edf(tmp_path / "rec.edf"))
    assert header.index("fp1") == "FP1"


def test_header_unknown_channel_lists_available(tmp_path):
    header = reader._read_header(_write_edf(tmp_path / "rec.edf"))
    with pytest.raises(RuntimeError, match="Available channels: F7, F8, FP1"):
        header.index("Cz")


# ---------------------------------------------------------------------------
# 3. WINDOW vs HEADER
# ---------------------------------------------------------------------------
def test_clock_start_is_anchored_to_recording_start(tmp_path):
    header = reader._read_header(_write_edf(tmp_path / "rec.edf"))
    # 13:12:38 is two seconds after the recording began at 13:12:36.
    p = _params(start=TimePoint(_START_CLOCK_S + 2, is_clock=True), duration_s=1.0, clock_mode=True)
    window = reader._check_against_header(p, header)
    assert window.start_s == pytest.approx(2.0)
    assert window.end_s == pytest.approx(3.0)


def test_clock_start_without_recording_timestamp_raises(tmp_path):
    header = reader._read_header(_write_edf(tmp_path / "rec.edf"))
    header = header.__class__(header.labels, header.fs, header.duration_s, header.unit, None)
    p = _params(start=TimePoint(_START_CLOCK_S, is_clock=True), duration_s=1.0, clock_mode=True)
    with pytest.raises(RuntimeError, match="no start timestamp"):
        reader._check_against_header(p, header)


def test_end_and_duration_must_agree(tmp_path):
    header = reader._read_header(_write_edf(tmp_path / "rec.edf"))
    p = _params(start=TimePoint(0.0, False), end=TimePoint(5.0, False), duration_s=2.0)
    with pytest.raises(RuntimeError, match="disagree"):
        reader._check_against_header(p, header)


def test_window_limits_enforced(tmp_path):
    header = reader._read_header(_write_edf(tmp_path / "rec.edf", fs=4, seconds=1000))
    over = _params(duration_s=readers.MAX_DURATION_S + 5)
    with pytest.raises(RuntimeError, match="maximum"):
        reader._check_against_header(over, header)
    tiny = _params(duration_s=0.5)          # 2 samples at 4 Hz < MIN_SAMPLES
    with pytest.raises(RuntimeError, match="Widen the window"):
        reader._check_against_header(tiny, header)


def test_montage_needs_equal_sampling_rates(tmp_path):
    header = reader._read_header(_write_edf(tmp_path / "rec.edf", freqs=[256, 128, 256]))
    with pytest.raises(RuntimeError, match="same sampling rate"):
        reader._check_against_header(_params(channel="F7", channel2="F8"), header)


def test_volt_unit_comes_from_header_unless_given(tmp_path):
    header = reader._read_header(_write_edf(tmp_path / "rec.edf", dimension="mV"))
    assert reader._check_against_header(_params(), header).volt_symbol == "mV"
    assert reader._check_against_header(_params(volt=(1e-6, "µV")), header).volt_symbol == "µV"


def test_unrecognised_header_unit_raises(tmp_path):
    header = reader._read_header(_write_edf(tmp_path / "rec.edf", dimension="furlongs"))
    with pytest.raises(RuntimeError, match="not a recognized voltage unit"):
        reader._check_against_header(_params(), header)


# ---------------------------------------------------------------------------
# 4. THE SIGNAL
# ---------------------------------------------------------------------------
def test_signal_time_is_seconds_from_recording_start(tmp_path):
    src = _write_edf(tmp_path / "rec.edf", fs=256)
    sig = readers.load_signal(src, _params(start=TimePoint(3.0, False), duration_s=2.0))
    assert sig.x_unit == "s"
    assert sig.t[0] == pytest.approx(3.0)
    assert sig.t[-1] == pytest.approx(5.0 - 1 / 256)
    assert sig.t.size == 2 * 256
    assert sig.fs == 256.0


def test_signal_voltage_is_in_display_unit(tmp_path):
    # Channel F7 is a 100 uV-amplitude sine. Displayed in mV its peak is 0.1.
    src = _write_edf(tmp_path / "rec.edf")
    sig = readers.load_signal(src, _params(volt=(1e-3, "mV")))
    assert sig.y_unit == "mV"
    assert np.max(np.abs(sig.y)) == pytest.approx(0.1, rel=1e-2)


def test_montage_is_channel_minus_channel2(tmp_path):
    src = _write_edf(tmp_path / "rec.edf")
    a = readers.load_signal(src, _params(channel="F7"))
    b = readers.load_signal(src, _params(channel="F8"))
    m = readers.load_signal(src, _params(channel="F7", channel2="F8"))
    assert m.channel == "F7-F8" and m.channel_kind == "montage"
    assert np.allclose(m.y, a.y - b.y)


def test_signal_carries_recording_start_clock(tmp_path):
    sig = readers.load_signal(_write_edf(tmp_path / "rec.edf"), _params())
    assert sig.recording_start_clock_s == pytest.approx(_START_CLOCK_S)
    assert sig.y_domain == "amplitude"
