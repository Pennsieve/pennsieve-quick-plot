"""
fcs_channel_histograms — Flow Cytometry Standard (.fcs) summary plot.

Reads an .fcs file with `fcsparser` and renders a grid of histograms, one
per fluorescence/scatter channel. The grid is sized by the channel count
(square-ish layout) and each subplot is labeled with the channel's $PnN
parameter name from the FCS metadata.

Use this when the user asks for a quick overview of an FCS file — typical
phrasing: "summarize this FCS file", "show me histograms", "what does
this flow data look like". The output is a single PNG suitable for inline
display in chat (matches the `figure.png` convention the data-target
harvests).

Deterministic: same input → same output bytes (assuming matplotlib
version pinned via the EFS layer). No randomness in bin edges; we use
`auto` with a deterministic seed for any subsampling.
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
    import fcsparser
    import matplotlib

    matplotlib.use("Agg")  # headless — no display in Lambda or batch ECS
    import matplotlib.pyplot as plt
    import numpy as np

    meta, df = fcsparser.parse(target_file_path, reformat_meta=True)
    if df is None or df.empty:
        raise RuntimeError("FCS file parsed but contained no events")

    if len(df) > MAX_EVENTS_FOR_PLOT:
        # Deterministic subsample — seed from event count so re-renders
        # of the same file produce the same plot.
        rng = np.random.default_rng(seed=len(df))
        idx = rng.choice(len(df), size=MAX_EVENTS_FOR_PLOT, replace=False)
        df = df.iloc[idx]

    channels = list(df.columns)
    if not channels:
        raise RuntimeError("FCS file has no channels to plot")

    cols = math.ceil(math.sqrt(len(channels)))
    rows = math.ceil(len(channels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes_flat = [axes] if rows == cols == 1 else list(axes.flat)

    # Pull friendly names from the FCS metadata when available
    # ($PnS = "short name", typically the antibody / marker label).
    short_names = meta.get("_channels_", None)
    label_for = {}
    if short_names is not None:
        for _, row in short_names.iterrows():
            n = row.get("$PnN")
            s = row.get("$PnS")
            if n and isinstance(n, str):
                label_for[n] = s.strip() if isinstance(s, str) and s.strip() else n

    for ax, ch in zip(axes_flat, channels):
        values = df[ch].to_numpy()
        # `auto` chooses a sensible bin count; safe across log/linear ranges.
        ax.hist(values, bins="auto", color="#2a6c97", alpha=0.85)
        ax.set_title(label_for.get(ch, ch), fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)

    # Hide unused axes when channel count doesn't fill the grid.
    for ax in axes_flat[len(channels):]:
        ax.set_visible(False)

    event_count = len(df)
    fig.suptitle(
        f"FCS channel histograms — {event_count:,} events"
        + (f" (subsampled from {len(df):,})" if event_count == MAX_EVENTS_FOR_PLOT else ""),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
