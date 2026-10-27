"""
ts_dsp.modality — which kind of recording a set of channel names looks like.

Scalp EEG and intracranial iEEG both arrive as EDF (and, from the MEF
converter, as NWB), so the file format says nothing about the modality.
The channel names do: a scalp montage uses the 10-20 / 10-10 system's
names (Fp1, Fz, Cz, O2, …); intracranial electrodes are arrays of
numbered contacts (LA1 … LA8, RAH1 … RAH10, "EEG AD 01-Ref"). A template
declares the modalities it is written for (SUPPORTED_MODALITIES) and
compares them with what the file looks like, so an EEG recording is not
silently drawn with iEEG conventions. The guess can be wrong (a sEEG array
named "F" with twelve contacts looks a lot like a dense scalp montage), so
every template also takes an explicit `modality` argument that overrides
it — the BIDS path or the dataset usually knows.

    detect_modality(labels) -> "ieeg" | "eeg" | "unknown"

Readers know nothing about modality; the template asks the reader for the
channel names and calls this. Standard library only.
"""

from __future__ import annotations

import re

IEEG = "ieeg"
EEG = "eeg"
UNKNOWN = "unknown"
MODALITIES: tuple[str, ...] = (IEEG, EEG)

# Scalp positions of the 10-20 and 10-10 systems, plus the usual
# references (A1/A2 earlobes, M1/M2 mastoids). Lowercase for matching.
_SCALP_NAMES: frozenset[str] = frozenset(
    name.lower() for name in (
        "Fp1", "Fp2", "Fpz", "AF3", "AF4", "AF7", "AF8", "AFz",
        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "Fz",
        "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FCz", "FT7", "FT8", "FT9", "FT10",
        "C1", "C2", "C3", "C4", "C5", "C6", "Cz", "T3", "T4", "T7", "T8",
        "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz", "TP7", "TP8", "TP9", "TP10",
        "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "Pz", "T5", "T6",
        "PO3", "PO4", "PO7", "PO8", "POz", "O1", "O2", "Oz", "Iz", "Nz",
        "A1", "A2", "M1", "M2",
    )
)
# Scalp names with no contact number: a sure sign of a scalp montage, since
# an intracranial array always numbers its contacts.
_SCALP_MARKERS: frozenset[str] = frozenset(n for n in _SCALP_NAMES if not n[-1].isdigit())

# Leading tokens that name the signal type rather than the electrode.
_TYPE_PREFIXES = ("eeg", "ieeg", "seeg", "ecog", "ref")
# An intracranial contact: an array name of letters, then its number.
_CONTACT = re.compile(r"^[a-z]{1,8}\d{1,3}$")


def normalise_label(label: str) -> str:
    """The electrode name inside a channel label, lowercase, no spaces:
    "EEG Fp1-Ref" -> "fp1", "EEG AD 01-Ref" -> "ad01", "LA 3" -> "la3"."""
    text = str(label).strip().lower()
    text = text.split("-", 1)[0]                 # drop a reference / montage suffix
    tokens = text.split()
    while len(tokens) > 1 and tokens[0] in _TYPE_PREFIXES:
        tokens = tokens[1:]
    return "".join(tokens)


def detect_modality(labels) -> str:
    """"eeg" when the channel names are scalp positions, "ieeg" when they
    are numbered intracranial contacts, "unknown" when neither dominates
    (no recognisable names, or EKG/EMG-only files)."""
    scalp = array = markers = 0
    for label in labels:
        name = normalise_label(label)
        if not name:
            continue
        if name in _SCALP_NAMES:
            scalp += 1
            markers += name in _SCALP_MARKERS
        elif _CONTACT.match(name):
            array += 1
    if scalp == 0 and array == 0:
        return UNKNOWN
    # Unnumbered scalp names (Cz, Fz, …) settle it for EEG unless the
    # numbered contacts clearly outnumber everything scalp-like; otherwise
    # the larger group wins, ties going to iEEG because dense sEEG arrays
    # collide with 10-10 names (F1 … F10) while scalp montages never
    # produce names like LA7 or RAH3.
    if markers and scalp >= array:
        return EEG
    if array >= scalp:
        return IEEG
    return EEG


def describe(labels, limit: int = 4) -> str:
    """A few of the channel names for an error message: "Fp1, Cz, O2, …"."""
    names = [str(x).strip() for x in labels if str(x).strip()]
    shown = ", ".join(names[:limit])
    return shown + (", …" if len(names) > limit else "")


__all__ = ["IEEG", "EEG", "UNKNOWN", "MODALITIES", "detect_modality", "normalise_label", "describe"]
