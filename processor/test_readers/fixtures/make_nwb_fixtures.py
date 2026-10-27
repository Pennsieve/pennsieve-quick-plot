"""
Regenerate the NWB fixtures next to this file with processor-plot-prep's
own writer, so test_nwb_fixture.py reads exactly what the converter
produces rather than what we believe it produces.

    PLOT_PREP=/path/to/processor-plot-prep \\
        python processor/test_readers/fixtures/make_nwb_fixtures.py

needs pynwb (the writer's dependency) in the current environment, plus the
processor-plot-prep checkout on PLOT_PREP. Run it as a script, as above,
not with -m: the converter's writer lives in a package also called
`processor`, which the script puts first on sys.path. Re-run it when
nwb_writer.py changes, then commit the two files; the tests import only
the constants and `synthetic_counts()` from this module, so they need
neither pynwb nor the converter.

Two recordings, both 2 channels at 128 Hz, 10 s of samples:

  mef_continuous.nwb   one segment: rate + starting_time
  mef_gapped.nwb       two segments with a 5 s hole after 4 s, so the
                       samples run 0–4 s and 9–15 s: per-sample timestamps

Sample values are what `synthetic_counts()` below returns — raw A/D counts,
channel i a sine of amplitude 100·(i+1) at (i+1) Hz — with per-channel
voltage conversion factors of 0.5 and 0.25 µV per count, so LA01 reads as
a 50 µV sine and LA02 as a 50 µV sine at 2 Hz. The session starts at
2020-09-24 13:12:36 UTC.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHANNELS = ("LA01", "LA02")
FS = 128.0
SECONDS = 10.0
N_SAMPLES = int(FS * SECONDS)
UV_PER_COUNT = (0.5, 0.25)
SESSION_START = datetime(2020, 9, 24, 13, 12, 36, tzinfo=timezone.utc)
GAP_AFTER_S = 4.0
GAP_LENGTH_S = 5.0


def synthetic_counts(start: int, end: int) -> np.ndarray:
    """Raw counts for samples [start, end): channel i is amplitude
    100·(i+1) at (i+1) Hz, indexed by sample number (gaps do not shift it)."""
    t = np.arange(start, end) / FS
    return np.stack([(100.0 * (i + 1)) * np.sin(2 * np.pi * (i + 1) * t)
                     for i in range(len(CHANNELS))], axis=1)


def timestamps_seconds(start: int, end: int, *, gapped: bool) -> np.ndarray:
    """Seconds from the session start for samples [start, end)."""
    t = np.arange(start, end, dtype=np.float64) / FS
    if gapped:
        t[t >= GAP_AFTER_S] += GAP_LENGTH_S
    return t


@dataclass
class _SyntheticReader:
    """Stands in for mef2's MultiChannelReader: the attributes NWBWriter reads."""
    gapped: bool
    channel_names: list = field(default_factory=lambda: list(CHANNELS))
    voltage_conversion_factors: list = field(default_factory=lambda: list(UV_PER_COUNT))
    num_samples: int = N_SAMPLES
    sampling_rate: float = FS
    session_start_time: datetime = SESSION_START

    @property
    def num_channels(self) -> int:
        return len(self.channel_names)

    def has_gaps(self) -> bool:
        return self.gapped

    def read_all_channels(self, start: int, end: int) -> np.ndarray:
        return synthetic_counts(start, end)

    def get_timestamps_seconds(self, start: int, end: int) -> np.ndarray:
        return timestamps_seconds(start, end, gapped=True)


def main() -> int:
    plot_prep = os.environ.get("PLOT_PREP")
    if not plot_prep or not Path(plot_prep, "processor", "converters", "mef2", "nwb_writer.py").is_file():
        print("Set PLOT_PREP to a processor-plot-prep checkout "
              "(it must contain processor/converters/mef2/nwb_writer.py).", file=sys.stderr)
        return 2
    sys.path.insert(0, plot_prep)
    from processor.converters.mef2.nwb_writer import NWBWriter  # the converter's writer

    for name, gapped in (("mef_continuous.nwb", False), ("mef_gapped.nwb", True)):
        out = HERE / name
        if out.exists():
            out.unlink()
        NWBWriter(_SyntheticReader(gapped), out, chunk_samples=512).write()
        print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
