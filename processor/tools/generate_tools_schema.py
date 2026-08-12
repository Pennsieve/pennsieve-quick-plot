"""
Emit every tool-registry family's catalog as JSON — one file per registry —
for pennsieve-mcp to embed (see also templates/generate_template_schema.py,
which emits the template catalog that references these families via its
`pipeline_tools` field).

A "registry" is any subpackage of `processor.tools` that exposes a
`specs() -> list[ToolSpec]` callable (today: ts_dsp). Each one becomes
`<short>_tools.json`, where <short> is the subpackage name with a trailing
"_dsp" stripped — so `ts_dsp` -> `ts_tools.json`. The JSON's `family` field
keeps the FULL subpackage name ("ts_dsp"); that field — not the filename —
is the join key `templates.json` entries use, so the filename is cosmetic.

Adding a new tool family (e.g. img_dsp for image ops) requires no change
here: give the subpackage a specs() and it is discovered and emitted.

Usage:
  python -m processor.tools.generate_tools_schema              # stdout
  python -m processor.tools.generate_tools_schema --out DIR    # DIR/<short>_tools.json per registry

To regenerate the copies pennsieve-mcp embeds (or run `make schemas`):
  python -m processor.tools.generate_tools_schema \\
      --out ../pennsieve-mcp/internal/tools/schemas
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pkgutil

import processor.tools


def _discover_registries() -> dict[str, "object"]:
    """{subpackage name -> imported module} for every processor.tools
    subpackage exposing a callable specs()."""
    registries = {}
    for info in pkgutil.iter_modules(processor.tools.__path__):
        if not info.ispkg:
            continue  # loose helper modules (like this one) are not registries
        mod = importlib.import_module(f"processor.tools.{info.name}")
        if callable(getattr(mod, "specs", None)):
            registries[info.name] = mod
    return registries


def _basename_for(family: str) -> str:
    short = family[: -len("_dsp")] if family.endswith("_dsp") else family
    return f"{short}_tools.json"


def build_catalog(family: str, mod: "object") -> dict:
    """Return {"family": ..., "tools": [ {name, description, requires,
    produces, params}, ... ]} for one registry."""
    tools = []
    for s in mod.specs():
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
    return {"family": family, "tools": tools}


def build_catalogs() -> dict[str, dict]:
    """{output basename -> catalog} for every discovered registry."""
    return {
        _basename_for(family): build_catalog(family, mod)
        for family, mod in sorted(_discover_registries().items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", metavar="DIR",
                        help="write one <short>_tools.json per registry into "
                             "DIR instead of printing a combined map to stdout")
    ns = parser.parse_args()

    catalogs = build_catalogs()
    if ns.out:
        for basename, catalog in catalogs.items():
            path = os.path.join(ns.out, basename)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(catalog, indent=2) + "\n")
            print(f"wrote {path}")
    else:
        print(json.dumps(catalogs, indent=2))


if __name__ == "__main__":
    main()
