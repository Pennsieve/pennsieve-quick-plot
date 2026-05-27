"""
fcs_channel_histograms — Flow Cytometry Standard (.fcs) summary plot.

Reads an .fcs file with `flowio` (the lightweight underlying parser that
flowkit wraps) and renders a grid of histograms, one per fluorescence/
scatter channel. Each subplot is labeled with the channel's $PnS
(antibody / marker) short name when present, falling back to $PnN
(parameter name).

Use this when the user asks for a quick overview of an FCS file — typical
phrasing: "summarize this FCS file", "show me histograms", "what does
this flow data look like". The output is a single PNG suitable for inline
display in chat (matches the `figure.png` convention the data-target
harvests).

Why flowio and not flowkit / fcsparser:

  - `fcsparser` (the original choice) calls `numpy.ndarray.newbyteorder()`,
    which NumPy 2.0 removed. The layer's pinned version raises
    AttributeError at parse time.
  - `flowkit` works on NumPy 2 but its top-level `__init__` eagerly
    imports gates / Session / Workspace / bokeh — ~20 seconds of import
    cost on the EFS-mounted layer that we don't need for histograms.
  - `flowio` is the lightweight reader that flowkit wraps internally
    (already on the layer as flowkit's own dep). Imports in <1s; same
    NumPy-2 compatibility. Exposes everything we need:
    FlowData.as_array(), .pnn_labels, .pns_labels, .event_count.

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
    import flowio
    import matplotlib

    matplotlib.use("Agg")  # headless — no display in Lambda or batch ECS
    import matplotlib.pyplot as plt
    import numpy as np

    fd = flowio.FlowData(target_file_path)
    # as_array returns the events reshaped to (n_events, n_channels).
    # preprocess=False keeps the values as stored in the file (no gain,
    # log, or time-channel scaling applied) — for a summary plot we want
    # what's actually in the file, not a derived transformation.
    events = fd.as_array(preprocess=False)
    if events is None or events.size == 0:
        raise RuntimeError("FCS file parsed but contained no events")

    channels = list(fd.pnn_labels)
    if not channels:
        raise RuntimeError("FCS file has no channels to plot")

    if len(events) > MAX_EVENTS_FOR_PLOT:
        # Deterministic subsample — seed from event count so re-renders
        # of the same file produce the same plot.
        rng = np.random.default_rng(seed=len(events))
        idx = rng.choice(len(events), size=MAX_EVENTS_FOR_PLOT, replace=False)
        events = events[idx]

    # Prefer the $PnS short name (typically the antibody / marker label)
    # when set; fall back to $PnN (parameter name). flowio stores them
    # as parallel lists already.
    pns_labels = list(fd.pns_labels)
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
    total_count = fd.event_count
    fig.suptitle(
        f"FCS channel histograms — {plotted_count:,} events"
        + (f" (subsampled from {total_count:,})" if plotted_count < total_count else ""),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
