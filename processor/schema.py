"""
processor.schema — emit the template and tool catalogs as JSON.

pennsieve-mcp embeds these two files (`internal/tools/schemas/`) to build
the `plot_file` tool's template enum and description text, so the two repos
can't drift on template names, arguments, file-type coverage or the DSP
tool list. The single source of truth is the code:

  templates_schema()  <- processor.templates._REGISTRY and each module's
                         declarative contract (NAME, SUPPORTED_EXTENSIONS,
                         SUMMARY, ARGS_SPEC, EXAMPLE_ARGS, ARGS_NOTES,
                         PIPELINE_TOOLS; see templates/contract.py)
  tools_schema()      <- processor.tools.ts_dsp.specs(), i.e. the ToolSpec
                         each @dsp_tool attaches and ALL_TOOLS lists

`templates.json` entries name the tool family they accept in `pipeline_tools`
("ts_dsp"); `ts_tools.json` carries that same name in its `family` field.
That string, not the filename, is the join key the Go side uses.

Usage:
  python -m processor.schema              # print both documents to stdout
  python -m processor.schema --out DIR    # write DIR/templates.json + DIR/ts_tools.json

`make schemas` runs the second form into schema/; tests in
test_templates/test_generate_schemas.py assert the checked-in files match
a fresh regeneration.
"""

from __future__ import annotations

import argparse
import json
import os

from processor import templates
from processor.tools import ts_dsp

TEMPLATES_BASENAME = "templates.json"
TOOLS_BASENAME = "ts_tools.json"


def templates_schema() -> dict:
    """{"templates": [ {name, extensions, summary, needs_args, pipeline_tools,
    args, example_args, notes}, ... ]} — one entry per registered template.
    `needs_args` is derived (any required arg), not declared, so it can't lie."""
    entries = []
    for name in templates.known_names():
        mod = templates.get(name)
        args_spec = getattr(mod, "ARGS_SPEC", ())
        entries.append({
            "name": mod.NAME,
            "extensions": list(mod.SUPPORTED_EXTENSIONS),
            "summary": getattr(mod, "SUMMARY", ""),
            "needs_args": any(a.required for a in args_spec),
            "pipeline_tools": getattr(mod, "PIPELINE_TOOLS", None),
            "args": [
                {
                    "name": a.name,
                    "type": a.type,
                    "required": a.required,
                    "unit": a.unit,
                    "description": a.description,
                }
                for a in args_spec
            ],
            "example_args": getattr(mod, "EXAMPLE_ARGS", None),
            "notes": getattr(mod, "ARGS_NOTES", ""),
        })
    return {"templates": entries}


def tools_schema() -> dict:
    """{"family": "ts_dsp", "tools": [ {name, description, requires, produces,
    params}, ... ]} — one entry per tool in ts_dsp.ALL_TOOLS, name-sorted."""
    tools = []
    for s in ts_dsp.specs():
        tools.append({
            "name": s.name,
            "description": s.description,
            "requires": dict(s.requires),
            "produces": dict(s.produces),
            "params": [
                {
                    "name": p.name,
                    "required": p.required,
                    "unit": p.unit,
                    "description": p.description,
                }
                for p in s.params
            ],
        })
    return {"family": "ts_dsp", "tools": tools}


def all_schemas() -> dict[str, dict]:
    """{output basename -> document}."""
    return {TEMPLATES_BASENAME: templates_schema(), TOOLS_BASENAME: tools_schema()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", metavar="DIR",
                        help=f"write {TEMPLATES_BASENAME} and {TOOLS_BASENAME} into DIR "
                             "instead of printing to stdout")
    ns = parser.parse_args()

    schemas = all_schemas()
    if ns.out:
        os.makedirs(ns.out, exist_ok=True)
        for basename, doc in schemas.items():
            path = os.path.join(ns.out, basename)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(doc, indent=2) + "\n")
            print(f"wrote {path}")
    else:
        print(json.dumps(schemas, indent=2))


if __name__ == "__main__":
    main()
