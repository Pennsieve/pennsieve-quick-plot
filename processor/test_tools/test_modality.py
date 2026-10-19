"""
Unit tests for ts_dsp/modality.py — telling scalp EEG from intracranial
iEEG by channel names.

The detector is a heuristic the template consults before drawing, and the
user's explicit `modality` argument overrides it, so what these tests pin
is that the clear cases come out right and the unclear ones say "unknown"
rather than guessing.

To run:
pytest processor/test_tools/test_modality.py
"""

from __future__ import annotations

import pytest

from processor.tools.ts_dsp import MODALITIES, detect_modality
from processor.tools.ts_dsp.modality import describe, normalise_label

SCALP_10_20 = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4",
               "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2")
SEEG_ARRAYS = tuple(f"{array}{i}" for array in ("LA", "LAH", "RPH", "ROF") for i in range(1, 9))


def test_modalities_vocabulary():
    assert MODALITIES == ("ieeg", "eeg")


@pytest.mark.parametrize("label, expected", [
    ("Fp1", "fp1"),
    ("EEG Fp1-Ref", "fp1"),           # signal-type prefix and reference suffix dropped
    ("EEG AD 01-Ref", "ad01"),        # the names mef2streamer sees on Pennsieve
    ("LA 3", "la3"),
    ("Fp1-F7", "fp1"),                # a bipolar label keeps its first electrode
    ("  Cz ", "cz"),
])
def test_normalise_label(label, expected):
    assert normalise_label(label) == expected


def test_scalp_montage_is_eeg():
    assert detect_modality(SCALP_10_20) == "eeg"
    assert detect_modality([f"EEG {n}-Ref" for n in SCALP_10_20]) == "eeg"
    # A few scalp names are enough; EKG/EMG channels do not count either way.
    assert detect_modality(("F7", "F8", "FP1", "EKG2")) == "eeg"


def test_intracranial_arrays_are_ieeg():
    assert detect_modality(SEEG_ARRAYS) == "ieeg"
    assert detect_modality(("EEG AD 01-Ref", "EEG PT 01-Ref")) == "ieeg"
    assert detect_modality(("LA01", "LA02")) == "ieeg"
    # A single array whose name collides with a 10-10 position (F1 … F12)
    # still has contacts no scalp montage has, and the tie goes to iEEG.
    assert detect_modality([f"F{i}" for i in range(1, 13)] + list(SEEG_ARRAYS[:8])) == "ieeg"


def test_unnumbered_scalp_names_settle_a_mixed_file_for_eeg():
    # Cz/Fz/Pz never appear as intracranial contacts.
    assert detect_modality(("Fz", "Cz", "Pz", "F1", "F2", "C1", "C2")) == "eeg"


def test_nothing_recognisable_is_unknown():
    assert detect_modality(()) == "unknown"
    assert detect_modality(("EKG", "EMG", "Pulse", "Event")) == "unknown"
    assert detect_modality(("", "   ")) == "unknown"


def test_describe_lists_a_few_names():
    assert describe(("Fp1", "Cz", "O2")) == "Fp1, Cz, O2"
    assert describe(SCALP_10_20) == "Fp1, Fp2, F7, F3, …"
