"""
fcs_time_diagnostics — acquisition-quality QC plot for an FCS file.

For each non-time channel, plots the channel's value distribution as a
function of acquisition time. Healthy acquisitions show horizontal
bands (stable signal throughout). Clogs, flow-rate drift, laser power
fluctuation, and panel-related issues all show up here even when the
single-channel histograms look fine — the marginal distributions hide
acquisition drift.

Use this when the user asks "did the acquisition run cleanly", "any
issues with this file", "QC this flow run", or before drilling into
biology questions on a file they don't trust.

Channel selection: scans `pnn_labels` for the time channel — case-
insensitive match against {"Time", "time", "TIME"}. If absent, raises
and the caller falls through to the agent. All other channels are
plotted in a square-ish grid, hexbin density per panel.

Caps:
  - 100_000 event subsample (FCS files can be huge; binned density
    plots saturate quickly above this).
  - 24 non-time channels max (beyond that the grid stops being
    legible). Truncation annotated in the suptitle.

Deterministic: same input → same output bytes. Subsampling uses a
content-derived seed so re-renders of the same file produce the same
plot.
"""

from __future__ import annotations

import math

NAME = "fcs_time_diagnostics"
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".fcs",)
SUMMARY = "channel-vs-time hexbin grid; \"is acquisition healthy / any drift\" (QC view)"

MAX_EVENTS_FOR_PLOT = 100_000
MAX_CHANNELS_FOR_PLOT = 24


def _find_time_channel(labels: list[str]) -> int | None:
    """Return the index of the time channel, or None if no channel
    label matches a case-insensitive 'time' prefix."""
    for i, n in enumerate(labels):
        if isinstance(n, str) and n.strip().lower() == "time":
            return i
    return None


def render(target_file_path: str, output_path: str) -> None:
    import flowio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fd = flowio.FlowData(target_file_path)
    events = fd.as_array(preprocess=False)
    if events is None or events.size == 0:
        raise RuntimeError("FCS file parsed but contained no events")

    labels = list(fd.pnn_labels)
    pns_labels = list(fd.pns_labels)
    time_idx = _find_time_channel(labels)
    if time_idx is None:
        raise RuntimeError(
            "FCS file has no Time channel — fcs_time_diagnostics needs one. "
            "Channels: " + ", ".join(labels)
        )

    # Subsample for the plot (bin density saturates above ~100k anyway).
    if len(events) > MAX_EVENTS_FOR_PLOT:
        rng = np.random.default_rng(seed=len(events))
        idx = rng.choice(len(events), size=MAX_EVENTS_FOR_PLOT, replace=False)
        # Preserve time order in the subsample so the hexbin x-axis still
        # makes visual sense as "left=earlier, right=later".
        idx.sort()
        events = events[idx]

    time_values = events[:, time_idx]

    # Build the list of channels to plot — everything except the time
    # channel, in the order they appear in the file. Truncate to fit
    # the grid; annotate in the title if we drop some.
    plot_indices = [i for i in range(len(labels)) if i != time_idx]
    truncated = len(plot_indices) > MAX_CHANNELS_FOR_PLOT
    if truncated:
        plot_indices = plot_indices[:MAX_CHANNELS_FOR_PLOT]

    cols = math.ceil(math.sqrt(len(plot_indices)))
    rows = math.ceil(len(plot_indices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes_flat = [axes] if rows == cols == 1 else list(axes.flat)

    # Display labels: $PnS short name when present, else $PnN.
    def display_label(i: int) -> str:
        if i < len(pns_labels) and isinstance(pns_labels[i], str) and pns_labels[i].strip():
            return pns_labels[i].strip()
        return labels[i]

    for panel_i, channel_i in enumerate(plot_indices):
        ax = axes_flat[panel_i]
        values = events[:, channel_i]
        # Hexbin time-vs-channel. mincnt=1 keeps empty regions clean.
        # gridsize tuned for legibility in small panels.
        ax.hexbin(time_values, values, gridsize=40, cmap="viridis", mincnt=1)
        ax.set_title(display_label(channel_i), fontsize=10)
        ax.set_xlabel("time", fontsize=8)
        ax.set_ylabel("value", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

    # Hide unused axes.
    for ax in axes_flat[len(plot_indices):]:
        ax.set_visible(False)

    plotted_count = len(events)
    suffix_bits = []
    if plotted_count < fd.event_count:
        suffix_bits.append(f"subsampled from {fd.event_count:,}")
    if truncated:
        suffix_bits.append(
            f"first {MAX_CHANNELS_FOR_PLOT} of {len(labels) - 1} non-time channels"
        )
    suffix = f" ({'; '.join(suffix_bits)})" if suffix_bits else ""

    fig.suptitle(
        f"FCS time diagnostics — {plotted_count:,} events vs time{suffix}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
