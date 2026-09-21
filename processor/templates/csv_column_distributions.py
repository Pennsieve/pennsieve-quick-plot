"""
csv_column_distributions — CSV / TSV summary plot.

Reads the file with pandas, picks the numeric columns, and renders a
grid of histograms — one per column. The grid is sized by the column
count (square-ish layout) and each subplot is labeled with the column
name from the header row.

Use this when the user asks for a quick overview of a tabular file —
typical phrasing: "show me distributions in this CSV", "summarize this
file", "what's the data look like". The output is a single PNG suitable
for inline display in chat.

Non-numeric columns (strings, datetimes, mixed) are skipped — they don't
histogram cleanly and a quick summary needs to stay quick. If the file
has NO numeric columns, the template raises and the workflow falls
through to the agent (which can build something custom).

Defaults:
  - row subsample at 100_000 (CSVs can be very large; we don't need
    every row for a visual summary)
  - column cap at 24 (beyond that the grid stops being readable;
    user can ask for a custom plot via the agent for specific columns)
  - separator inferred from extension (.tsv → tab, .csv → comma)

Deterministic: same input → same output bytes. No randomness in bin
edges; row subsampling uses a content-derived seed so re-renders of
the same file produce the same plot.
"""

from __future__ import annotations

import math

NAME = "csv_column_distributions"
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".csv", ".tsv")
SUMMARY = "histogram grid, one per numeric column (text columns skipped)"

# Caps. CSVs vary wildly in shape — we want the template to run in
# bounded time / produce a legible plot regardless of input size.
MAX_ROWS_FOR_PLOT = 100_000
MAX_COLUMNS_FOR_PLOT = 24


def render(target_file_path: str, output_path: str) -> None:
    # All heavy imports inside render() — see processor/templates/__init__.py
    # for why (registry imports every template module at processor startup;
    # module-top heavy imports would cost every cold start regardless of
    # which template was chosen).
    import matplotlib

    matplotlib.use("Agg")  # headless — no display in Lambda or batch ECS
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    sep = "\t" if target_file_path.lower().endswith(".tsv") else ","
    # low_memory=False forces pandas to use a single chunk for type
    # inference. The default warns + can mis-detect column types on
    # large files. For a summary we want clean dtypes; the cost is
    # higher transient memory, which is fine on a Lambda with 3 GB.
    df = pd.read_csv(target_file_path, sep=sep, low_memory=False)

    if df.empty:
        raise RuntimeError("CSV parsed but contained no rows")

    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        raise RuntimeError(
            "CSV has no numeric columns to histogram — every column was "
            "string / categorical. Ask the user to refine via a custom prompt."
        )

    if len(numeric) > MAX_ROWS_FOR_PLOT:
        # Deterministic subsample — seed from row count so re-renders
        # of the same file produce the same plot.
        rng = np.random.default_rng(seed=len(numeric))
        idx = rng.choice(len(numeric), size=MAX_ROWS_FOR_PLOT, replace=False)
        numeric = numeric.iloc[idx]

    columns = list(numeric.columns)
    truncated_columns = len(columns) > MAX_COLUMNS_FOR_PLOT
    if truncated_columns:
        columns = columns[:MAX_COLUMNS_FOR_PLOT]

    cols = math.ceil(math.sqrt(len(columns)))
    rows = math.ceil(len(columns) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes_flat = [axes] if rows == cols == 1 else list(axes.flat)

    for i, ax in enumerate(axes_flat[: len(columns)]):
        # Drop NaN per-column (different columns may have different
        # nullable patterns; pd.DataFrame.dropna across all columns
        # would over-trim).
        values = numeric[columns[i]].dropna().to_numpy()
        if values.size == 0:
            ax.set_visible(False)
            continue
        # `auto` chooses a sensible bin count; safe across log/linear ranges.
        ax.hist(values, bins="auto", color="#2a6c97", alpha=0.85)
        ax.set_title(columns[i], fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)

    # Hide unused axes when column count doesn't fill the grid.
    for ax in axes_flat[len(columns):]:
        ax.set_visible(False)

    plotted_rows = len(numeric)
    total_rows = len(df)
    suffix_bits = []
    if plotted_rows < total_rows:
        suffix_bits.append(f"subsampled from {total_rows:,}")
    if truncated_columns:
        suffix_bits.append(f"first {MAX_COLUMNS_FOR_PLOT} of {numeric.shape[1]} numeric columns")
    suffix = f" ({'; '.join(suffix_bits)})" if suffix_bits else ""

    fig.suptitle(
        f"CSV column distributions — {plotted_rows:,} rows{suffix}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
