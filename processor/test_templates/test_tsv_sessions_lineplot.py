"""
Tests for the tsv_sessions_lineplot template.

To run:
Requires pandas + matplotlib + pytest to be importable 
pip install -r requirements-dev.txt  
pytest processor/test_templates/test_tsv_sessions_lineplot.py
"""

# SET UP 
from __future__ import annotations
import pytest
import matplotlib.axes
from processor.templates import tsv_sessions_lineplot as template

def _write_tsv(path, text: str) -> str:
    """Write a TSV to a path`, return its path in string."""
    path.write_text(text)
    return str(path)

# TEST FOR GOOD PATH AND INPUT 
# 1. produce a PNG
def test_render_writes_a_png(tmp_path):
    src = _write_tsv(tmp_path / "sessions.tsv",
                     "session_id\tsubject_age\nses-visit01m\t1.5\nses-visit04.5m\t4.7\nses-visit10.5m\t11\n")
    out = tmp_path / "figure.png"

    template.render(src, str(out))

    assert out.is_file()
    assert out.read_bytes().startswith(b"\x89PNG")  # PNG file signature

# 2. test that a non-TSV input is rejected
def test_render_raises_on_non_tsv_input(tmp_path):
    src = _write_tsv(tmp_path / "sessions.csv",
                     "session_id,subject_age\nses-visit01m,1.5\nses-visit04.5m,4.7\n")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"))

# 3.1 test if error is raised when there are too few plottable sessions (a line
# needs at least MIN_SESSIONS=2 points). Here only 1 row has a usable age.
def test_render_raises_when_too_few_sessions(tmp_path):
    src = _write_tsv(tmp_path / "sessions.tsv", "session_id\tsubject_age\nses-visit01m\t1.5\n")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"))

# 3.2, test if error is raised when there are too many sessions (> MAX_SESSIONS):
# build MAX_SESSIONS + 1 rows with unique labels and unique ages.
def test_render_raises_when_too_many_sessions(tmp_path):
    n = template.MAX_SESSIONS + 1
    rows = "".join(f"ses-{i}\t{i}\n" for i in range(n))
    src = _write_tsv(tmp_path / "sessions.tsv", "session_id\tsubject_age\n" + rows)
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"))

# TEST FOR BAD INPUT TSV DATA
# 1. test if error is raised when no appropriate columns is given
def test_render_raises_on_wrong_columns(tmp_path):
    src = _write_tsv(tmp_path / "sessions.tsv", "eeg_channel\teeg_sampling_frequency\nA1\t1024\nA2\t1024\n")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"))

# 2. test if error is raised when the input tsv is empty 
def test_render_raises_on_empty_tsv(tmp_path):
    src = _write_tsv(tmp_path / "sessions.tsv", "session_id\tsubject_age\n")  # header only
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"))

# 3.1. test if error is rasied when the session id column is missing
def test_render_raises_when_no_session_column(tmp_path):
    src = _write_tsv(tmp_path / "sessions.tsv", "subject_age\tnotes\n10.5\tfoo\n")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"))

# 3.2. test if error is raised when the age column is missing
def test_render_raises_when_no_age_column(tmp_path):
    src = _write_tsv(tmp_path / "sessions.tsv", "session_id\tnotes\nses-visit01m\tfoo\n")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"))

# 4.1. test if error is raised when all session id values are empty
def test_render_raises_when_all_session_id_empty(tmp_path):
    src = _write_tsv(tmp_path / "sessions.tsv", "session_id\tsubject_age\n\t1.5\n\t4.7\n")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"))

# 4.2. test if error is raised when all age values are empty
def test_render_raises_when_all_age_empty(tmp_path):
    src = _write_tsv(tmp_path / "sessions.tsv", "session_id\tsubject_age\nses-visit01m\t\nses-visit04.5m\t\n")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"))

# 5.1. test if error is rasied when session id is non-unique
def test_render_raises_when_session_id_non_unique(tmp_path):
    src = _write_tsv(tmp_path / "sessions.tsv",
                     "session_id\tsubject_age\nses-visit01m\t1.5\nses-visit01m\t4.7\n")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png")) 

# 5.2. test if error is raised when age is non-unique
def test_render_raises_when_age_non_unique(tmp_path):
    src = _write_tsv(tmp_path / "sessions.tsv",
                     "session_id\tsubject_age\nses-visit01m\t1.5\nses-visit04.5m\t1.5\n")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"))

# 6. test if error is raised when all ages are non-numeric
def test_render_raises_when_all_ages_non_numeric(tmp_path):
    src = _write_tsv(tmp_path / "sessions.tsv",
                     "session_id\tsubject_age\npreimplant\tn/a\npostimplant\tunknown\n")
    with pytest.raises(RuntimeError):
        template.render(src, str(tmp_path / "figure.png"))

# PLOTTING CHECK
# 1. test that the line is plotted in ascending age order, no matter the row order in the file. 
def test_render_plots_ages_in_ascending_order(monkeypatch, tmp_path):
    captured = {}
    original_plot = matplotlib.axes.Axes.plot

    def spy_plot(self, *args, **kwargs):
        # render() calls ax.plot(x_positions, y_values, ...); grab y_values.
        if len(args) >= 2:
            captured["ages"] = list(args[1])
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy_plot)

    # age order in the file (60, 25, 40).
    src = _write_tsv(tmp_path / "sessions.tsv",
                     "session_id\tsubject_age\nses-old\t60\nses-young\t25\nses-mid\t40\n")
    template.render(src, str(tmp_path / "figure.png"))

    ages = [float(a) for a in captured["ages"]]
    assert ages == sorted(ages)         # ascending
    assert ages == [25.0, 40.0, 60.0]   # sorted, not the file order [60, 25, 40]

# 2. test that a row with a MISSING session id (but valid age) is dropped, not
# plotted with the literal label "nan". 
def test_render_drops_rows_with_missing_session(monkeypatch, tmp_path):
    captured = {}
    original_plot = matplotlib.axes.Axes.plot

    def spy_plot(self, *args, **kwargs):
        if len(args) >= 2:
            captured["ages"] = [float(a) for a in args[1]]
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy_plot)

    src = _write_tsv(tmp_path / "sessions.tsv",
                     "session_id\tsubject_age\nses-a\t1\nses-b\t2\n\t3\n")  # 3rd session empty
    template.render(src, str(tmp_path / "figure.png"))

    assert captured["ages"] == [1.0, 2.0]  # age 3 (missing session) was dropped