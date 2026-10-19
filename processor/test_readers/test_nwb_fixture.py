"""
The NWB reader against files the converter actually wrote.

test_nwb_to_signal.py builds its files with h5py in the layout we believe
pynwb uses. These tests read fixtures/mef_continuous.nwb and
fixtures/mef_gapped.nwb, written by processor-plot-prep's NWBWriter (see
fixtures/make_nwb_fixtures.py for how, and for what is in them), so if the
converter's layout and this reader ever drift apart, it shows here and not
on the compute node.

To run:
pytest processor/test_readers/test_nwb_fixture.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from processor import readers
from processor.errors import PlotErrorCode, PlotInvalidInputError
from processor.readers import nwb_to_signal as reader
from processor.templates.ieeg_processed_timeseries import RenderParams
from processor.test_readers.fixtures import make_nwb_fixtures as fx
from processor.tools.ts_dsp import TimePoint

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONTINUOUS = str(_FIXTURES / "mef_continuous.nwb")
GAPPED = str(_FIXTURES / "mef_gapped.nwb")


def _params(channel="LA01", channel2=None, start_s=1.0, duration_s=2.0) -> RenderParams:
    return RenderParams(
        channel=channel, channel2=channel2, montage=channel2 is not None,
        start=TimePoint("relative", start_s), end=None, duration_s=duration_s,
        time_factor=1.0, time_symbol="s", clock_mode=False,
        y_limits=None, volt=None, pipeline=None,
    )


def test_fixtures_are_checked_in():
    assert Path(CONTINUOUS).is_file() and Path(GAPPED).is_file()


def test_continuous_header_matches_what_the_writer_was_given():
    h = reader._read_header(CONTINUOUS)
    assert h.series == "/acquisition/ElectricalSeries"
    assert h.labels == fx.CHANNELS
    assert h.fs == fx.FS and h.n_samples == fx.N_SAMPLES and not h.gapped
    assert h.duration_s == pytest.approx(fx.SECONDS)
    assert h.start == fx.SESSION_START
    assert h.conversion == pytest.approx(1e-6) and h.offset == 0.0
    assert h.channel_conversion == pytest.approx(fx.UV_PER_COUNT)


def test_continuous_samples_come_back_in_microvolts():
    # LA01: 100 counts × 0.5 µV/count = 50 µV; the values are the writer's
    # input converted, sample for sample.
    sig = readers.load_signal(CONTINUOUS, _params(start_s=1.0, duration_s=2.0))
    i0, i1 = int(1.0 * fx.FS), int(3.0 * fx.FS)
    expected_uv = fx.synthetic_counts(i0, i1)[:, 0] * fx.UV_PER_COUNT[0]
    assert sig.y_unit == "µV"
    assert np.allclose(sig.y, expected_uv)
    assert sig.t[0] == pytest.approx(1.0) and sig.t.size == i1 - i0
    assert sig.recording_start == fx.SESSION_START


def test_continuous_montage():
    m = readers.load_signal(CONTINUOUS, _params(channel="LA01", channel2="LA02"))
    a = readers.load_signal(CONTINUOUS, _params(channel="LA01"))
    b = readers.load_signal(CONTINUOUS, _params(channel="LA02"))
    assert m.channel == "LA01-LA02" and np.allclose(m.y, a.y - b.y)


def test_gapped_header():
    h = reader._read_header(GAPPED)
    assert h.gapped and h.fs == pytest.approx(fx.FS)
    last = fx.timestamps_seconds(fx.N_SAMPLES - 1, fx.N_SAMPLES, gapped=True)[0]
    assert h.last_s == pytest.approx(last)
    assert h.duration_s == pytest.approx(fx.SECONDS + fx.GAP_LENGTH_S)


def test_gapped_window_after_the_hole_reads_the_right_samples():
    # The hole is 4–9 s; samples 4 s into the recording resume at 9 s.
    sig = readers.load_signal(GAPPED, _params(start_s=10.0, duration_s=2.0))
    i0, i1 = int(5.0 * fx.FS), int(7.0 * fx.FS)        # sample numbers, not seconds
    assert sig.t[0] == pytest.approx(10.0) and sig.t.size == i1 - i0
    assert np.allclose(sig.y, fx.synthetic_counts(i0, i1)[:, 0] * fx.UV_PER_COUNT[0])


def test_gapped_window_across_the_hole_is_refused():
    with pytest.raises(PlotInvalidInputError) as ei:
        readers.load_signal(GAPPED, _params(start_s=3.0, duration_s=8.0))
    assert ei.value.error_code is PlotErrorCode.WINDOW_SPANS_GAP
    assert ei.value.error_facts["gap_start_s"] == pytest.approx(fx.GAP_AFTER_S - 1 / fx.FS)
    assert ei.value.error_facts["gap_end_s"] == pytest.approx(fx.GAP_AFTER_S + fx.GAP_LENGTH_S)
