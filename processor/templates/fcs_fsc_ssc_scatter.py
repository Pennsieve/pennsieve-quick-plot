"""
fcs_fsc_ssc_scatter — FSC vs SSC population view for an FCS file.

The single most universally useful flow-cytometry plot: forward scatter
(roughly cell size) vs side scatter (roughly internal complexity / granularity).
Every flow analyst's first plot. Reveals population structure at a glance —
live cells, debris, doublets, aggregates — without any gating.

Use this when the user asks "what's in this flow file", "show me the
populations", "what does it look like" — anywhere they want orientation
before drilling into specific markers. The histogram template
(`fcs_channel_histograms`) shows per-channel distributions; this template
shows the joint distribution of the two universal scatter channels.

Renders as a hexbin density plot when the event count is large
(>SCATTER_THRESHOLD), and a true scatter plot otherwise. Hexbin reads
much better than overlapping dots at 100k+ events; under that threshold
the scatter form preserves the "individual event" feel flow analysts
expect.

Channel selection: scans `pnn_labels` for the first label starting with
"FSC" (any of FSC, FSC-A, FSC-H, FSC-W) and the first starting with
"SSC". Case-insensitive. If either is missing, raises and the caller
falls through to the agent loop.

Deterministic: same input → same output bytes (matplotlib version pinned
via the EFS layer). No randomness in bin edges or hexbin tessellation.
"""

from __future__ import annotations

NAME = "fcs_fsc_ssc_scatter"
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".fcs",)

# Above this event count, hexbin density beats a scatter plot for
# readability. Below it, individual dots are still useful.
SCATTER_THRESHOLD = 20_000


def _find_channel(labels: list[str], prefix: str) -> int | None:
    """Return the index of the first label starting with `prefix`
    (case-insensitive), or None if not found."""
    p = prefix.upper()
    for i, n in enumerate(labels):
        if isinstance(n, str) and n.upper().startswith(p):
            return i
    return None


def render(target_file_path: str, output_path: str) -> None:
    # Heavy imports inside render() — see processor/templates/__init__.py
    # for why (registry imports every template; module-top heavy imports
    # would cost every cold start regardless of which template was chosen).
    import flowio
    import matplotlib

    matplotlib.use("Agg")  # headless — no display in Lambda or batch ECS
    import matplotlib.pyplot as plt

    fd = flowio.FlowData(target_file_path)
    events = fd.as_array(preprocess=False)
    if events is None or events.size == 0:
        raise RuntimeError("FCS file parsed but contained no events")

    labels = list(fd.pnn_labels)
    fsc_idx = _find_channel(labels, "FSC")
    ssc_idx = _find_channel(labels, "SSC")
    if fsc_idx is None or ssc_idx is None:
        raise RuntimeError(
            "FCS file is missing FSC and/or SSC channels — fcs_fsc_ssc_scatter "
            "needs both. Found channels: " + ", ".join(labels)
        )

    fsc = events[:, fsc_idx]
    ssc = events[:, ssc_idx]
    n_events = len(events)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    if n_events > SCATTER_THRESHOLD:
        # Hexbin density: tessellate the plane, color by event count.
        # `mincnt=1` skips empty bins so the background stays clean.
        # `gridsize=80` gives ~6400 hex cells — plenty of resolution for
        # population structure without being noisy.
        hb = ax.hexbin(fsc, ssc, gridsize=80, cmap="viridis", mincnt=1)
        cbar = fig.colorbar(hb, ax=ax, label="event count", shrink=0.85)
        cbar.ax.tick_params(labelsize=9)
        plot_type = "density"
    else:
        # Direct scatter: small dots, low alpha for overlap legibility.
        ax.scatter(fsc, ssc, s=2, alpha=0.4, color="#2a6c97", edgecolors="none")
        plot_type = "scatter"

    ax.set_xlabel(labels[fsc_idx], fontsize=11)
    ax.set_ylabel(labels[ssc_idx], fontsize=11)
    ax.tick_params(axis="both", labelsize=9)

    fig.suptitle(
        f"FCS scatter — {labels[fsc_idx]} vs {labels[ssc_idx]} "
        f"({n_events:,} events, {plot_type})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
