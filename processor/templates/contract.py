"""
templates.contract — the declarative half of a template's contract.

Alongside NAME / SUPPORTED_EXTENSIONS / render(), a template module may
declare (all optional; a plain summary-only template omits the rest):

    SUMMARY: str            one-liner the MCP `plot_file` tool shows the model
                            in its template enum (include example user
                            phrasings — that text is what routes requests).
    ARGS_SPEC:              tuple[TemplateArg, ...] — the render() kwargs the
                            caller must/can supply via `template_args`.
                            The generated schema (and therefore the MCP tool
                            description) is built from this, so it is the
                            single source of truth for a template's inputs.
    EXAMPLE_ARGS: dict      a valid `template_args` example, shown verbatim
                            in the MCP tool description.
    ARGS_NOTES: str         cross-argument rules a per-arg description can't
                            express (e.g. "exactly one of end_time/duration").
    PIPELINE_TOOLS: str     name of the tool registry family (e.g. "ts_dsp")
                            whose registered DSP tools this template accepts
                            in its `pipeline` arg. Absent/None = no pipeline.

`generate_template_schema.py` reads these into `templates.json`, which
pennsieve-mcp embeds — see that script's docstring for the regeneration
workflow. Keep this module stdlib-only: every template imports it at
module top, which happens at processor startup.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemplateArg:
    """One `template_args` key a template's render() accepts.

    `type` is a human/model-facing hint (e.g. 'string', 'number',
    'number or "HH:MM:SS" string'), not a JSON-Schema type — it is rendered
    into description text, so plain language beats formality.
    """

    name: str
    type: str
    required: bool = True
    unit: str = ""
    description: str = ""
