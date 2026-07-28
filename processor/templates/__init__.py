"""
Per-format template registry — canned plotting paths the processor tries
BEFORE the LLM agent loop.

Each module under `processor.templates` exposes:

    NAME: str                     # template identifier the MCP tool ships
    SUPPORTED_EXTENSIONS:         # tuple of file suffixes (lowercase, with dot)
        tuple[str, ...]
    def render(target_file_path: str, output_path: str) -> None
        # Reads from target_file_path, writes a PNG to output_path.
        # Raises on failure. Should be deterministic given the same input.

The dispatcher in `processor.main.run()` looks up the chosen template by
NAME and calls its render() function. Adding a new template = one new
module + one import line in `_REGISTRY` below; no MCP, workflow, or
Lambda config change.

Canned templates run in-process and import their format-specific deps
from the shared EFS layer (`quick-plot-stack`), the same layer the agent
loop uses. Heavy imports happen lazily inside each module's render()
function so an unrelated template choice doesn't pay the
matplotlib/fcsparser cold-start tax.

If a template's render() raises or no NAME matches, the caller's
expected behavior is to fall through to the agent loop. See main.py.
"""

from __future__ import annotations

# Import each template module and register it. Keep this list short and
# curated — every entry contributes to the catalog the MCP `plot_file`
# tool advertises in its template-argument enum, so the model's choice
# space stays narrow and predictable.
from processor.templates import csv_column_distributions  # noqa: E402
from processor.templates import edf_processed_timeseries  # noqa: E402
from processor.templates import fcs_channel_histograms  # noqa: E402
from processor.templates import fcs_compensation_heatmap  # noqa: E402
from processor.templates import fcs_fsc_ssc_scatter  # noqa: E402
from processor.templates import fcs_time_diagnostics  # noqa: E402
from processor.templates import tsv_sessions_lineplot  # noqa: E402


_REGISTRY: dict[str, object] = {
    csv_column_distributions.NAME: csv_column_distributions,
    edf_processed_timeseries.NAME: edf_processed_timeseries,
    fcs_channel_histograms.NAME: fcs_channel_histograms,
    fcs_compensation_heatmap.NAME: fcs_compensation_heatmap,
    fcs_fsc_ssc_scatter.NAME: fcs_fsc_ssc_scatter,
    fcs_time_diagnostics.NAME: fcs_time_diagnostics,
    tsv_sessions_lineplot.NAME: tsv_sessions_lineplot,
}


def get(name: str):
    """Return the template module for `name`, or None when unknown."""
    return _REGISTRY.get(name)


def known_names() -> list[str]:
    return sorted(_REGISTRY.keys())


__all__ = ["get", "known_names"]
