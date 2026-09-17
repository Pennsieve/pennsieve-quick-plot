"""
processor.readers — file -> Signal.

A reader turns a recording on disk plus the user's validated request into a
raw `Signal` (see tools/ts_dsp/signal_definition.py). It owns everything that
needs the file: reading the header, checking the request against it, reading
the channels, and building the Signal. It never decides how the figure looks.

One module per file format, named by what it converts:

  edf_to_signal.py   EDF / EDF+ via pyedflib

`load_signal()` below is the only entry point templates call; it picks the
module by file extension.

Clip limits shared by every reader:
  MAX_DURATION_S  longest window we will read (a longer raw trace is both
                  slow to draw and unreadable at screen resolution)
  MIN_SAMPLES     fewest samples worth plotting (a handful of samples is a
                  dot cloud, not a signal)
"""

from __future__ import annotations

import os

from processor.readers import edf_to_signal

MAX_DURATION_S = 600.0
MIN_SAMPLES = 16

_READERS = {
    ".edf": edf_to_signal,
}


def supported_extensions() -> tuple[str, ...]:
    return tuple(sorted(_READERS))


def load_signal(path: str, params):
    """Read the clip described by `params` from `path` and return a Signal.

    `params` is the template's validated RenderParams (channel, channel2,
    window, unit choices). Raises RuntimeError on any problem with the file
    or with the request once checked against the file's header.
    """
    ext = os.path.splitext(path)[1].lower()
    reader = _READERS.get(ext)
    if reader is None:
        raise RuntimeError(
            f"No reader for {ext!r} files. Supported: {', '.join(supported_extensions())}."
        )
    if not os.path.isfile(path):
        raise RuntimeError(f"Input file does not exist: {path!r}")
    return reader.load_signal(path, params)


__all__ = ["load_signal", "supported_extensions", "MAX_DURATION_S", "MIN_SAMPLES"]
