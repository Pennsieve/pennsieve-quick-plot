"""
Tests for processor.readers.nwb_to_signal — NWB (ElectricalSeries) -> Signal.

Same seams as the EDF reader tests:

  1. DISPATCH          load_signal() picks the reader for .nwb
  2. HEADER            _read_header: names, rate, length, start, scaling,
                       continuous vs gapped; the series lookup
  3. WINDOW vs HEADER  channels, clock/date anchoring (UTC), default unit
  4. THE SIGNAL        seconds from recording start, volts from counts via
                       conversion × channel_conversion + offset, montage,
                       recording_start, and a gapped recording read in
                       the right place
  5. GAPS              a window that crosses a hole is refused and the hole
                       is named

The files are written with h5py directly, in the layout pynwb writes (the
datasets and attributes the reader relies on, nothing more), so the tests
need neither pynwb nor a checked-in binary. test_nwb_fixture.py reads a
real file written by processor-plot-prep's writer.

To run:
pip install -r requirements-dev.txt
pytest processor/test_readers/test_nwb_to_signal.py
"""

from __future__ import annotations

from datetime import datetime, timezone

import h5py
import numpy as np
import pytest

from processor import readers
from processor.errors import PlotDataUnavailableError, PlotErrorCode, PlotInvalidInputError
from processor.readers import nwb_to_signal as reader
from processor.templates.edf_processed_timeseries import RenderParams
from processor.tools.ts_dsp import TimePoint

_START = datetime(2020, 9, 24, 13, 12, 36, tzinfo=timezone.utc)
_START_CLOCK_S = 13 * 3600 + 12 * 60 + 36
_FS = 256.0
_UV_TO_V = 1e-6


def _counts(labels, n):
    """Channel i: amplitude 100*(i+1) counts at (i+1) Hz — distinct per channel."""
    t = np.arange(n) / _FS
    return np.stack([(100.0 * (i + 1)) * np.sin(2 * np.pi * (i + 1) * t)
                     for i in range(len(labels))], axis=1)


def _write_nwb(path, labels=("LA01", "LA02", "LA03"), seconds=20, *,
               gap=None, start=_START, channel_conversion=None,
               starting_time=0.0, name_column="channel_name",
               series_name="ElectricalSeries", extra_series=()) -> str:
    """Write the NWB datasets the reader relies on, as pynwb lays them out.

    `gap=(after_s, length_s)` writes per-sample timestamps with a hole of
    `length_s` seconds after `after_s`; otherwise the series is continuous
    (`starting_time` + `rate`). `channel_conversion` is µV per count, per
    channel (default 1 for all).
    """
    n = int(seconds * _FS)
    if channel_conversion is None:
        channel_conversion = [1.0] * len(labels)
    with h5py.File(str(path), "w") as f:
        if start is not None:
            f["session_start_time"] = start.isoformat()
        table = f.create_group("/general/extracellular_ephys/electrodes")
        table["id"] = np.arange(len(labels))
        if name_column:
            table[name_column] = np.array([s.encode() for s in labels], dtype=object)
        for sname in (series_name, *extra_series):
            es = f.create_group(f"/acquisition/{sname}")
            es.attrs["neurodata_type"] = "ElectricalSeries"
            data = es.create_dataset("data", data=_counts(labels, n))
            data.attrs["conversion"] = _UV_TO_V
            data.attrs["offset"] = 0.0
            data.attrs["unit"] = "volts"
            es["channel_conversion"] = np.asarray(channel_conversion, dtype=float)
            region = es.create_dataset("electrodes", data=np.arange(len(labels)))
            region.attrs["table"] = table.ref
            if gap is None:
                st = es.create_dataset("starting_time", data=float(starting_time))
                st.attrs["rate"] = _FS
                st.attrs["unit"] = "seconds"
            else:
                after_s, length_s = gap
                ts = np.arange(n) / _FS
                ts[ts >= after_s] += length_s
                es.create_dataset("timestamps", data=ts)
    return str(path)


def _params(channel="LA01", channel2=None, start=TimePoint("relative", 0.0),
            end=None, duration_s=2.0, volt=None, clock_mode=False,
            time_factor=1.0, time_symbol="s") -> RenderParams:
    return RenderParams(
        channel=channel, channel2=channel2, montage=channel2 is not None,
        start=start, end=end, duration_s=duration_s,
        time_factor=time_factor, time_symbol=time_symbol, clock_mode=clock_mode,
        y_limits=None, volt=volt, pipeline=None,
    )


# ---------------------------------------------------------------------------
# 1. DISPATCH
# ---------------------------------------------------------------------------
def test_load_signal_dispatches_nwb(tmp_path):
    sig = readers.load_signal(_write_nwb(tmp_path / "rec.nwb"), _params())
    assert sig.channel == "LA01" and sig.x_domain == "time"
    assert ".nwb" in readers.supported_extensions()


def test_unreadable_file_is_data_unavailable(tmp_path):
    bad = tmp_path / "rec.nwb"
    bad.write_text("not an hdf5 file")
    with pytest.raises(PlotDataUnavailableError) as ei:
        readers.load_signal(str(bad), _params())
    assert ei.value.error_code is PlotErrorCode.UNREADABLE_FILE
    assert ei.value.error_facts["file"] == "rec.nwb"


def test_missing_library_is_not_blamed_on_the_file(tmp_path, monkeypatch):
    def no_lib(path):
        raise ImportError("No module named 'h5py'", name="h5py")

    monkeypatch.setattr(reader, "_read_header", no_lib)
    with pytest.raises(ImportError):
        reader.load_signal(_write_nwb(tmp_path / "rec.nwb"), _params())


# ---------------------------------------------------------------------------
# 2. HEADER
# ---------------------------------------------------------------------------
def test_read_header_continuous(tmp_path):
    h = reader._read_header(_write_nwb(tmp_path / "rec.nwb", seconds=10,
                                       channel_conversion=[0.5, 0.25, 1.0]))
    assert h.series == "/acquisition/ElectricalSeries"
    assert h.labels == ("LA01", "LA02", "LA03")
    assert h.fs == _FS and h.n_samples == 2560 and not h.gapped
    assert (h.first_s, h.last_s) == (0.0, pytest.approx(10.0 - 1 / _FS))
    assert h.duration_s == pytest.approx(10.0)
    assert h.start == _START and h.start.tzinfo is not None
    assert (h.conversion, h.offset) == (_UV_TO_V, 0.0)
    assert h.channel_conversion == (0.5, 0.25, 1.0)


def test_read_header_gapped_derives_the_rate_from_timestamps(tmp_path):
    # 20 s of samples with a 5 s hole after 10 s: the rate is still 256 Hz
    # and the last sample is at 25 s.
    h = reader._read_header(_write_nwb(tmp_path / "rec.nwb", gap=(10.0, 5.0)))
    assert h.gapped and h.fs == pytest.approx(_FS)
    assert h.last_s == pytest.approx(25.0 - 1 / _FS)


def test_header_channel_lookup_is_case_insensitive_and_lists_channels(tmp_path):
    h = reader._read_header(_write_nwb(tmp_path / "rec.nwb"))
    assert h.index("la02") == 1
    with pytest.raises(PlotInvalidInputError, match="Available channels: LA01, LA02, LA03"):
        h.index("Cz")


def test_channel_names_fall_back_to_label_then_id(tmp_path):
    by_label = reader._read_header(_write_nwb(tmp_path / "a.nwb", name_column="label"))
    assert by_label.labels == ("LA01", "LA02", "LA03")
    by_id = reader._read_header(_write_nwb(tmp_path / "b.nwb", name_column=None))
    assert by_id.labels == ("0", "1", "2")


def test_missing_start_time_is_allowed(tmp_path):
    h = reader._read_header(_write_nwb(tmp_path / "rec.nwb", start=None))
    assert h.start is None


def test_series_lookup(tmp_path):
    # Several series: the converter's name wins; otherwise it is ambiguous.
    h = reader._read_header(_write_nwb(tmp_path / "a.nwb", extra_series=("Other",)))
    assert h.series == "/acquisition/ElectricalSeries"
    with pytest.raises(PlotDataUnavailableError, match="several ElectricalSeries"):
        reader._read_header(_write_nwb(tmp_path / "b.nwb", series_name="A", extra_series=("B",)))
    # No series at all.
    with h5py.File(str(tmp_path / "c.nwb"), "w") as f:
        f.create_group("acquisition")
    with pytest.raises(PlotDataUnavailableError) as ei:
        reader._read_header(str(tmp_path / "c.nwb"))
    assert ei.value.error_code is PlotErrorCode.MISSING_REQUIRED_METADATA


# ---------------------------------------------------------------------------
# 3. WINDOW vs HEADER
# ---------------------------------------------------------------------------
def test_window_resolves_columns_and_clock_times_in_utc(tmp_path):
    h = reader._read_header(_write_nwb(tmp_path / "rec.nwb"))
    # 13:12:38 UTC is two seconds after the recording began.
    p = _params(channel="LA02", channel2="LA03",
                start=TimePoint("clock", _START_CLOCK_S + 2), duration_s=1.0, clock_mode=True)
    w = reader._check_against_header(p, h)
    assert (w.channel, w.channel2) == (1, 2)
    assert (w.start_s, w.end_s) == (pytest.approx(2.0), pytest.approx(3.0))


def test_date_input_resolves_against_the_session_start(tmp_path):
    h = reader._read_header(_write_nwb(tmp_path / "rec.nwb"))
    p = _params(start=TimePoint("datetime", at=datetime(2020, 9, 24, 13, 12, 40)),
                duration_s=1.0, clock_mode=True)
    assert reader._check_against_header(p, h).start_s == pytest.approx(4.0)


def test_default_display_unit_is_microvolts(tmp_path):
    h = reader._read_header(_write_nwb(tmp_path / "rec.nwb"))
    assert reader._check_against_header(_params(), h).volt_symbol == "µV"
    assert reader._check_against_header(_params(volt=(1e-3, "mV")), h).volt_symbol == "mV"


def test_window_before_the_first_sample_is_rejected(tmp_path):
    # A series that begins 5 s into the session has no data before that.
    h = reader._read_header(_write_nwb(tmp_path / "rec.nwb", starting_time=5.0))
    with pytest.raises(PlotInvalidInputError, match="before the first sample") as ei:
        reader._check_against_header(_params(start=TimePoint("relative", 3.0)), h)
    assert ei.value.error_facts["first_sample_s"] == 5.0


def test_shared_window_rules_apply(tmp_path):
    h = reader._read_header(_write_nwb(tmp_path / "rec.nwb", seconds=20))
    with pytest.raises(PlotInvalidInputError, match="past the end"):
        reader._check_against_header(_params(start=TimePoint("relative", 19.0), duration_s=5.0), h)
    with pytest.raises(PlotDataUnavailableError, match="Widen the window"):
        reader._check_against_header(_params(duration_s=0.01), h)


# ---------------------------------------------------------------------------
# 4. THE SIGNAL
# ---------------------------------------------------------------------------
def test_signal_time_is_seconds_from_recording_start(tmp_path):
    sig = readers.load_signal(_write_nwb(tmp_path / "rec.nwb"),
                              _params(start=TimePoint("relative", 3.0), duration_s=2.0))
    assert sig.x_unit == "s" and sig.fs == _FS
    assert sig.t[0] == pytest.approx(3.0)
    assert sig.t[-1] == pytest.approx(5.0 - 1 / _FS)
    assert sig.t.size == 2 * 256


def test_signal_voltage_applies_conversion_and_channel_conversion(tmp_path):
    # LA02 is a 200-count sine; 0.25 µV/count makes it 50 µV, i.e. 0.05 mV.
    src = _write_nwb(tmp_path / "rec.nwb", channel_conversion=[0.5, 0.25, 1.0])
    uv = readers.load_signal(src, _params(channel="LA02"))
    assert uv.y_unit == "µV" and np.max(np.abs(uv.y)) == pytest.approx(50.0, rel=1e-2)
    mv = readers.load_signal(src, _params(channel="LA02", volt=(1e-3, "mV")))
    assert mv.y_unit == "mV" and np.max(np.abs(mv.y)) == pytest.approx(0.05, rel=1e-2)


def test_montage_is_channel_minus_channel2(tmp_path):
    src = _write_nwb(tmp_path / "rec.nwb")
    a = readers.load_signal(src, _params(channel="LA01"))
    b = readers.load_signal(src, _params(channel="LA02"))
    m = readers.load_signal(src, _params(channel="LA01", channel2="LA02"))
    assert m.channel == "LA01-LA02" and m.channel_kind == "montage"
    assert np.allclose(m.y, a.y - b.y)


def test_signal_carries_the_utc_recording_start(tmp_path):
    sig = readers.load_signal(_write_nwb(tmp_path / "rec.nwb"), _params())
    assert sig.recording_start == _START
    assert sig.recording_start.tzinfo is not None and sig.y_domain == "amplitude"


def test_gapped_recording_reads_the_window_after_the_hole(tmp_path):
    # Samples resume at 15 s after a 5 s hole; a window at 16–18 s is read
    # from the right samples and its times come from the timestamps.
    src = _write_nwb(tmp_path / "rec.nwb", gap=(10.0, 5.0))
    sig = readers.load_signal(src, _params(start=TimePoint("relative", 16.0), duration_s=2.0))
    assert sig.t[0] == pytest.approx(16.0) and sig.t.size == 512
    assert np.all(np.diff(sig.t) > 0)
    # The same samples as in the continuous file (the hole only shifts time).
    cont = readers.load_signal(_write_nwb(tmp_path / "cont.nwb"),
                               _params(start=TimePoint("relative", 11.0), duration_s=2.0))
    assert np.allclose(sig.y, cont.y)


# ---------------------------------------------------------------------------
# 5. GAPS
# ---------------------------------------------------------------------------
def _expect_gap(src, params):
    with pytest.raises(PlotInvalidInputError) as ei:
        readers.load_signal(src, params)
    assert ei.value.error_code is PlotErrorCode.WINDOW_SPANS_GAP
    return ei.value


def test_window_crossing_a_gap_is_refused_and_the_gap_named(tmp_path):
    # Hole from 10 s to 15 s. The window 8–18 s crosses it.
    src = _write_nwb(tmp_path / "rec.nwb", gap=(10.0, 5.0))
    err = _expect_gap(src, _params(start=TimePoint("relative", 8.0), duration_s=10.0))
    assert str(err) == ("The requested window 8 s–18 s crosses a gap in the recording "
                        "(no data from 9.99609 s to 15 s). Choose a different window.")
    assert err.error_facts["gap_start_s"] == pytest.approx(10.0 - 1 / _FS)
    assert err.error_facts["gap_end_s"] == pytest.approx(15.0)
    assert "ends before" in err.user_next_step


def test_window_starting_or_ending_inside_a_gap_is_refused(tmp_path):
    src = _write_nwb(tmp_path / "rec.nwb", gap=(10.0, 5.0))
    # Starts inside the hole (12 s), ends after it.
    err = _expect_gap(src, _params(start=TimePoint("relative", 12.0), duration_s=5.0))
    assert err.error_facts["gap_end_s"] == pytest.approx(15.0)
    # Starts before the hole, ends inside it (9–12 s).
    err = _expect_gap(src, _params(start=TimePoint("relative", 9.0), duration_s=3.0))
    assert err.error_facts["gap_start_s"] == pytest.approx(10.0 - 1 / _FS)


def test_gap_message_uses_the_form_the_user_typed(tmp_path):
    src = _write_nwb(tmp_path / "rec.nwb", gap=(10.0, 5.0))
    # Clock input -> clock times on the recording's (UTC) clock.
    clock = _params(start=TimePoint("clock", _START_CLOCK_S + 8), duration_s=10.0, clock_mode=True)
    assert "13:12:44–13:12:54" in str(_expect_gap(src, clock))
    # Numeric input in minutes -> minutes.
    mins = _params(start=TimePoint("relative", 8.0), duration_s=10.0,
                   time_factor=60.0, time_symbol="min")
    assert "0.133333 min–0.3 min" in str(_expect_gap(src, mins))


def test_window_on_one_side_of_a_gap_is_fine(tmp_path):
    src = _write_nwb(tmp_path / "rec.nwb", gap=(10.0, 5.0))
    before = readers.load_signal(src, _params(start=TimePoint("relative", 7.0), duration_s=2.0))
    after = readers.load_signal(src, _params(start=TimePoint("relative", 15.0), duration_s=2.0))
    assert before.t.size == after.t.size == 512
