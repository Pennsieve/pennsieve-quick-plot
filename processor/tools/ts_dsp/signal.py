"""
ts_dsp.signal — the Signal *contract* for voltage timeseries.

This module is the shared agreement every other piece codes against: the
EDF reader produces a `Signal`, each DSP tool transforms a `Signal`, and the
template plots a `Signal`. For all of them to interoperate they must agree
on (a) the fields and what they mean, and (b) the vocabulary of legal
domain / unit values. That agreement lives here, in one place everyone
imports, so nothing drifts into "time" vs "temporal" vs "t".

It imports only the standard library, so importing it is cheap and it can be
depended on by readers, tools, and templates without any cycle.
"""

from __future__ import annotations

from dataclasses import dataclass


############# DOMAIN VOCABULARY ##################
# The legal values for a Signal's axis "domain" fields. The DSP type-system
# (tool `requires` / `produces`) is expressed in these exact strings, so a
# reader that sets x_domain="time" and a tool that requires x_domain="time"
# are guaranteed to match. Extend these as new transforms are added.
X_DOMAINS: tuple[str, ...] = ("time", "frequency")
# "amplitude" is the raw trace (the only y_domain the reader produces);
# "magnitude" is a spectrum's per-frequency strength (e.g. fft output). Keeping
# them distinct lets templates tell "still the raw trace" from "same unit but
# transformed" (e.g. the y_range guard), and requires-gates match exactly.
Y_DOMAINS: tuple[str, ...] = (
    "amplitude", "magnitude", "power", "energy",
    # windowed time-domain features (see feature_extraction.py)
    "rms", "zcr", "line_length", "kurtosis", "skewness",
)


############# UNIT TABLES ##################
# Time units -> seconds. Keys are lowercased and stripped of a trailing
# "s"-plural at lookup time, so "microseconds"/"microsecond" both hit.
_TIME_TO_S: dict[str, float] = {
    "us": 1e-6, "usec": 1e-6, "µs": 1e-6, "microsecond": 1e-6,
    "ms": 1e-3, "msec": 1e-3, "millisecond": 1e-3,
    "s": 1.0, "sec": 1.0, "second": 1.0,
    "min": 60.0, "minute": 60.0, "m": 60.0,
    "h": 3600.0, "hr": 3600.0, "hour": 3600.0,
}
# Canonical display symbol per resolved factor (so labels read cleanly
# regardless of which alias the user typed).
_TIME_SYMBOL: dict[float, str] = {
    1e-6: "µs", 1e-3: "ms", 1.0: "s", 60.0: "min", 3600.0: "h",
}

# Voltage units -> volts (SI).
_VOLT_TO_V: dict[str, float] = {
    "v": 1.0, "volt": 1.0,
    "mv": 1e-3, "millivolt": 1e-3,
    "uv": 1e-6, "µv": 1e-6, "microvolt": 1e-6,
    "nv": 1e-9, "nanovolt": 1e-9,
}
_VOLT_SYMBOL: dict[float, str] = {1.0: "V", 1e-3: "mV", 1e-6: "µV", 1e-9: "nV"}


############# SIGNAL BUNDLE ##################
@dataclass
class Signal:
    """A signal clip plus the axis metadata the plotter renders from.

    `t` and `y` are already expressed in `x_unit` / `y_unit`. Processing
    stages take a Signal and return a Signal (via dataclasses.replace),
    updating the four *_domain / *_unit fields as the physical meaning
    changes.
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
    # How the x tick LABELS are rendered (the axis data is always numeric):
    #   "numeric" -> plain integers/decimals in x_unit, no 1e9-style offset.
    #   "clock"   -> H:M:S wall clock; `t` then holds seconds-since-midnight.
    x_tick_style: str = "numeric"

    def x_label(self) -> str:
        return f"{self.x_domain} ({self.x_unit})"

    def y_label(self) -> str:
        return f"{self.y_domain} ({self.y_unit})"


############# UNIT RESOLUTION ##################
def _normalise_unit_key(unit: str) -> str:
    """Lowercase, strip whitespace, and drop a trailing plural 's'."""
    key = str(unit).strip().lower()
    # "seconds" -> "second", "microvolts" -> "microvolt", but leave a bare
    # "s"/"ms"/"us"/"mv"/"uv" alone (they are already the short symbols).
    if len(key) > 2 and key.endswith("s"):
        key = key[:-1]
    return key


def resolve_time_unit(unit: str | None, *, required: bool) -> tuple[float, str | None]:
    """(factor-to-seconds, display symbol) for a time unit.

    We do not default the unit. When `required` (a numeric time was given)
    and none is supplied, raise and ask the user for one. A supplied unit
    that is not a time unit always raises.
    """
    if unit is None or str(unit).strip() == "":
        if required:
            raise RuntimeError(
                "A time unit is required for numeric start/end/duration "
                "values — please specify one (us/usec, ms, s/sec, min, h), "
                "or give the times as H:M:S clock values instead."
            )
        return 1.0, None
    factor = _TIME_TO_S.get(_normalise_unit_key(unit))
    if factor is None:
        raise RuntimeError(
            f"Time unit {unit!r} is not a recognised unit of time. "
            "Use one of: us/usec, ms, s/sec, min, h."
        )
    return factor, _TIME_SYMBOL[factor]


def resolve_volt_unit(unit: str | None, *, required: bool = True) -> tuple[float, str]:
    """(factor-to-volts, display symbol) for a voltage unit.

    We do not default the unit. When `required` and none is supplied, raise
    and ask the user for one. A supplied unit that is not a voltage unit
    always raises. (`required=False` is used internally when reading a unit
    declared inside the file, where a sensible fallback is acceptable.)
    """
    if unit is None or str(unit).strip() == "":
        if required:
            raise RuntimeError(
                "A y-axis unit is required — please specify the voltage unit "
                "the range is in (V, mV, uV, or nV)."
            )
        return 1e-6, "µV"
    factor = _VOLT_TO_V.get(_normalise_unit_key(unit))
    if factor is None:
        raise RuntimeError(
            f"y-axis unit {unit!r} is not a recognised unit of voltage. "
            "Use one of: V, mV, uV, nV."
        )
    return factor, _VOLT_SYMBOL[factor]


def seconds_to_clock(seconds: float) -> str:
    """Seconds-since-midnight -> 'H:MM:SS' (or 'H:MM:SS.mmm' with a fraction)."""
    s = float(seconds) % (24 * 3600.0)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s - h * 3600 - m * 60
    if abs(sec - round(sec)) < 1e-6:
        return f"{h:d}:{m:02d}:{int(round(sec)):02d}"
    return f"{h:d}:{m:02d}:{sec:06.3f}"