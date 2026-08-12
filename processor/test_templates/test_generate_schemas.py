"""
Tests for the generated catalogs (templates.json / <family>_tools.json)
that pennsieve-mcp embeds — see generate_template_schema.py and
processor/tools/generate_tools_schema.py.

These pin the *contract* of the generated JSON: entry shape, required-arg
derivation, family join keys, and — most importantly — that every declared
template arg is a real keyword of the template's render() (the check that
keeps ARGS_SPEC from drifting from the code it describes).
"""

from __future__ import annotations

import inspect

from processor import templates
from processor.templates.generate_template_schema import build_catalog as build_template_catalog
from processor.tools.generate_tools_schema import build_catalogs as build_tool_catalogs


############# templates.json ##################

def test_every_registered_template_is_in_catalog():
    names = [t["name"] for t in build_template_catalog()["templates"]]
    assert names == templates.known_names()


def test_entry_shape_and_needs_args_derivation():
    by_name = {t["name"]: t for t in build_template_catalog()["templates"]}

    edf = by_name["edf_processed_timeseries"]
    assert edf["needs_args"] is True
    assert edf["pipeline_tools"] == "ts_dsp"
    assert edf["extensions"] == [".edf"]
    required = [a["name"] for a in edf["args"] if a["required"]]
    assert required == ["channel", "start_time", "y_range", "y_unit"]
    assert set(edf["example_args"]) >= set(required)

    csv = by_name["csv_column_distributions"]
    assert csv["needs_args"] is False
    assert csv["pipeline_tools"] is None
    assert csv["extensions"] == [".csv", ".tsv"]
    assert csv["args"] == []


def test_every_template_has_a_summary():
    for t in build_template_catalog()["templates"]:
        assert t["summary"].strip(), f"{t['name']} has an empty SUMMARY"


def test_args_spec_matches_render_signature():
    """Every declared arg (and `pipeline`, when a tool family is declared)
    must be an actual keyword parameter of render() — the catalog may not
    advertise arguments the code doesn't accept."""
    for entry in build_template_catalog()["templates"]:
        mod = templates.get(entry["name"])
        render_params = set(inspect.signature(mod.render).parameters)
        declared = {a["name"] for a in entry["args"]}
        missing = declared - render_params
        assert not missing, (
            f"{entry['name']}: ARGS_SPEC declares {sorted(missing)} "
            f"but render() has no such keyword(s)"
        )
        if entry["pipeline_tools"]:
            assert "pipeline" in render_params, (
                f"{entry['name']} declares PIPELINE_TOOLS but render() "
                f"has no `pipeline` kwarg"
            )


############# <family>_tools.json ##################

def test_tool_catalogs_one_file_per_registry():
    catalogs = build_tool_catalogs()
    assert "ts_tools.json" in catalogs
    ts = catalogs["ts_tools.json"]
    assert ts["family"] == "ts_dsp"
    names = [t["name"] for t in ts["tools"]]
    assert "highpass_filter" in names and "energy" in names


def test_pipeline_tools_references_resolve():
    """The family join key every template declares must exist among the
    emitted tool catalogs (the cross-file invariant the Go side panics on)."""
    families = {c["family"] for c in build_tool_catalogs().values()}
    for t in build_template_catalog()["templates"]:
        if t["pipeline_tools"] is not None:
            assert t["pipeline_tools"] in families, (
                f"{t['name']} references unknown tool family {t['pipeline_tools']!r}"
            )
