"""
processor.readers — file -> Signal.

A reader turns a recording on disk plus the user's validated request into a
raw `Signal` (see tools/ts_dsp/signal_definition.py). It owns everything that
needs the file: reading the header, checking the request against it, reading
the channels, and building the Signal. It never decides how the figure looks.

One module per file format, named by what it converts:

  edf_to_signal.py   EDF / EDF+ via pyedflib
  nwb_to_signal.py   NWB (an ElectricalSeries) via h5py — what
                     processor-plot-prep writes from a MEF 2.0 recording

plus the rules every reader shares:

  window.py          the request's window resolved against the recording
                     (clock/date anchoring, bounds, the clip limits
                     MAX_DURATION_S and MIN_SAMPLES)

`load_signal()` below is the entry point templates call to read a clip;
`channel_names()` lists a recording's channels without reading samples (a
template uses it to tell the modality). Both pick the module by file
extension.
"""

from __future__ import annotations

import os

from processor.errors import PlotDataUnavailableError, PlotErrorCode, PlotErrorStage

from processor.readers import edf_to_signal, nwb_to_signal
from processor.readers.window import MAX_DURATION_S, MIN_SAMPLES

_READERS = {
    ".edf": edf_to_signal,
    ".nwb": nwb_to_signal,
}


def supported_extensions() -> tuple[str, ...]:
    return tuple(sorted(_READERS))


def _reader_for(path: str):
    """The reader module for `path`'s extension; raises when there is none
    or the file is missing."""
    ext = os.path.splitext(path)[1].lower()
    reader = _READERS.get(ext)
    if reader is None:
        raise PlotDataUnavailableError(
            PlotErrorCode.UNSUPPORTED_FILE_FORMAT,
            f"No reader for {ext!r} files. Supported: {', '.join(supported_extensions())}.",
            error_facts={"extension": ext, "supported": list(supported_extensions())},
            source_stage=PlotErrorStage.READER,
        )
    if not os.path.isfile(path):
        raise PlotDataUnavailableError(
            PlotErrorCode.FILE_NOT_FOUND, f"Input file does not exist: {path!r}",
            error_facts={"file": os.path.basename(path)},
            source_stage=PlotErrorStage.READER)
    return reader


def load_signal(path: str, params):
    """Read the clip described by `params` from `path` and return a Signal.

    `params` is the template's validated RenderParams (channel, channel2,
    window, unit choices). Raises RuntimeError on any problem with the file
    or with the request once checked against the file's header (every one
    a PlotError subclass, so `except RuntimeError` still catches them).
    """
    return _reader_for(path).load_signal(path, params)


def channel_names(path: str) -> tuple:
    """The recording's channel names, in file order, from the header alone."""
    return _reader_for(path).channel_names(path)


__all__ = ["load_signal", "channel_names", "supported_extensions", "MAX_DURATION_S", "MIN_SAMPLES"]
