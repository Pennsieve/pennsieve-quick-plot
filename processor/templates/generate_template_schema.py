"""
Emit the template schema as JSON — the single source of truth the
pennsieve-mcp `plot_file` tool consumes to build its `template` enum and
the `template`/`template_args` description text (so the two repos can't
drift on template names, arguments, or file-type coverage).

Every module registered in `processor.templates._REGISTRY` contributes one
entry, built from its declarative contract (see templates/contract.py):
NAME, SUPPORTED_EXTENSIONS, SUMMARY, and — for parameterised templates —
ARGS_SPEC / EXAMPLE_ARGS / ARGS_NOTES / PIPELINE_TOOLS. `needs_args` is
derived (any required arg), not declared, so it can't lie.

`pipeline_tools` names the tool-registry family (e.g. "ts_dsp") whose
tools the template accepts in its `pipeline` arg; the matching family
schema is emitted by `processor.tools.generate_tools_schema`.

Usage:
  python -m processor.templates.generate_template_schema              # stdout
  python -m processor.templates.generate_template_schema --out DIR    # DIR/templates.json

To regenerate the copies pennsieve-mcp embeds (or run `make schemas`):
  python -m processor.templates.generate_template_schema \\
      --out ../pennsieve-mcp/internal/tools/schemas
"""

from __future__ import annotations

import argparse
import json
import os

from processor import templates


OUTPUT_BASENAME = "templates.json"


def build_schema() -> dict:
    """Return {"templates": [ {name, extensions, summary, needs_args, ...} ]}."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", metavar="DIR",
                        help=f"write {OUTPUT_BASENAME} into DIR instead of stdout")
    ns = parser.parse_args()

    doc = json.dumps(build_schema(), indent=2) + "\n"
    if ns.out:
        path = os.path.join(ns.out, OUTPUT_BASENAME)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(doc)
        print(f"wrote {path}")
    else:
        print(doc, end="")


if __name__ == "__main__":
    main()
