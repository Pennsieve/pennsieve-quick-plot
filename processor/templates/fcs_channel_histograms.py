"""
fcs_channel_histograms — Flow Cytometry Standard (.fcs) summary plot.

Reads an .fcs file with `flowkit` and renders a grid of histograms, one
per fluorescence/scatter channel. The grid is sized by the channel count
(square-ish layout) and each subplot is labeled with the channel's $PnS
(antibody / marker) short name when present, falling back to $PnN
(parameter name).

Use this when the user asks for a quick overview of an FCS file — typical
phrasing: "summarize this FCS file", "show me histograms", "what does
this flow data look like". The output is a single PNG suitable for inline
display in chat (matches the `figure.png` convention the data-target
harvests).

Why flowkit and not fcsparser: the layer's pinned fcsparser calls
`numpy.ndarray.newbyteorder()`, which NumPy 2.0 removed (AttributeError
at parse time). flowkit is also in the layer (REQUIREMENTS includes it
alongside fcsparser), actively maintained, and NumPy-2-compatible.

Deterministic: same input → same output bytes (matplotlib version
pinned via the EFS layer). No randomness in bin edges; subsampling
uses a content-derived seed so re-renders of the same file produce
the same plot.
"""

from __future__ import annotations

import math

NAME = "fcs_channel_histograms"
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".fcs",)

# Subsampling threshold. FCS files can carry hundreds of thousands of
# events; we only need a representative slice for a visual summary.
# 50_000 keeps the histograms statistically meaningful while bounding
# render time on any plausible file.
MAX_EVENTS_FOR_PLOT = 50_000


def render(target_file_path: str, output_path: str) -> None:
    # Heavy imports happen here, not at module-import time, so a
    # mis-routed dispatch doesn't pay the matplotlib cold-start.
    import flowkit as fk
    import matplotlib

    matplotlib.use("Agg")  # headless — no display in Lambda or batch ECS
    import matplotlib.pyplot as plt
    import numpy as np

    sample = fk.Sample(target_file_path)
    # get_events returns a (n_events, n_channels) array. Use "raw" so we
    # see the file's stored values without any auto-compensation applied
    # — for a summary plot we want what's actually in the file.
    events = sample.get_events(source="raw")
    if events is None or len(events) == 0:
        raise RuntimeError("FCS file parsed but contained no events")

    channels = list(sample.pnn_labels)
    if not channels:
        raise RuntimeError("FCS file has no channels to plot")

    if len(events) > MAX_EVENTS_FOR_PLOT:
        # Deterministic subsample — seed from event count so re-renders
        # of the same file produce the same plot.
        rng = np.random.default_rng(seed=len(events))
        idx = rng.choice(len(events), size=MAX_EVENTS_FOR_PLOT, replace=False)
        events = events[idx]

    # Prefer the $PnS short name (typically the antibody / marker label)
    # when set; fall back to $PnN (parameter name).
    pns_labels = list(sample.pns_labels)
    display_labels = [
        (pns_labels[i].strip() if i < len(pns_labels) and pns_labels[i] and pns_labels[i].strip() else channels[i])
        for i in range(len(channels))
    ]

    cols = math.ceil(math.sqrt(len(channels)))
    rows = math.ceil(len(channels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes_flat = [axes] if rows == cols == 1 else list(axes.flat)

    for i, ax in enumerate(axes_flat[: len(channels)]):
        values = events[:, i]
        # `auto` chooses a sensible bin count; safe across log/linear ranges.
        ax.hist(values, bins="auto", color="#2a6c97", alpha=0.85)
        ax.set_title(display_labels[i], fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)

    # Hide unused axes when channel count doesn't fill the grid.
    for ax in axes_flat[len(channels):]:
        ax.set_visible(False)

    plotted_count = len(events)
    total_count = sample.event_count
    fig.suptitle(
        f"FCS channel histograms — {plotted_count:,} events"
        + (f" (subsampled from {total_count:,})" if plotted_count < total_count else ""),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
