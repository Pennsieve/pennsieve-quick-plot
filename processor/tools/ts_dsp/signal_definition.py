"""
ts_dsp.signal_definition — what a Signal *is*.

Every other piece of the timeseries code codes against this one definition:
a reader produces a `Signal` from a file, each DSP tool takes a `Signal` and
returns a new one, and the template plots a `Signal`. For them to
interoperate they must agree on (a) the fields and what they mean, and
(b) the vocabulary of legal domain values. Both live here.

Unit tables and conversions live next door in `units.py`. This module
imports only the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass


############# DOMAIN VOCABULARY ##################
# The legal values for a Signal's axis "domain" fields. The DSP type-system
# (tool `requires` / `produces`) is expressed in these exact strings, so a
# reader that sets x_domain="time" and a tool that requires x_domain="time"
# are guaranteed to match. Extend these as new transforms are added.
X_DOMAINS: tuple[str, ...] = ("time", "frequency")
# "amplitude" is the raw trace (the only y_domain a reader produces);
# "magnitude" is a spectrum's per-frequency strength (e.g. fft output). Keeping
# them distinct lets templates tell "still the raw trace" from "same unit but
# transformed" (e.g. the y_range guard), and requires-gates match exactly.
Y_DOMAINS: tuple[str, ...] = (
    "amplitude", "magnitude", "power", "energy",
    # windowed time-domain features (see feature_extraction.py)
    "rms", "zcr", "line_length", "kurtosis", "skewness",
)


############# SIGNAL ##################
@dataclass
class Signal:
    """A signal clip plus the axis metadata the plotter renders from.

    `t` and `y` are already expressed in `x_unit` / `y_unit`. A reader
    always produces time in seconds from the start of the recording
    (x_unit="s"); the plot stage converts to the user's display unit or to
    wall-clock ticks. Processing stages take a Signal and return a Signal
    (via dataclasses.replace), updating the *_domain / *_unit fields as the
    physical meaning changes.
    """

    t: "object"          # np.ndarray of x values, in x_unit
    y: "object"          # np.ndarray of y values, in y_unit
    fs: float            # sampling rate (Hz) of the source recording
    channel: str
    # What `channel` represents: "single_channel" for one trace, "montage"
    # for a bipolar A-B derivation (then `channel` holds "A-B").
    channel_kind: str = "single_channel"
    x_domain: str = "time"
    x_unit: str = "s"
    y_domain: str = "amplitude"
    y_unit: str = "µV"
    # Wall-clock time of the recording's first sample, as seconds since
    # midnight, or None when the file carries no usable start timestamp.
    # Filled by the reader; the plot stage uses it to draw H:M:S ticks when
    # the user gave clock-string times.
    recording_start_clock_s: "float | None" = None

    def x_label(self) -> str:
        return f"{self.x_domain} ({self.x_unit})"

    def y_label(self) -> str:
        # Dimensionless quantities (e.g. kurtosis, skewness) have y_unit "";
        # skip the unit parens rather than rendering "kurtosis ()".
        return f"{self.y_domain} ({self.y_unit})" if self.y_unit else self.y_domain
