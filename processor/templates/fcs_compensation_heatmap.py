"""
fcs_compensation_heatmap — visualize an FCS file's spillover matrix.

Modern flow files often embed a compensation / spillover matrix in the
`$SPILL` or `$SPILLOVER` FCS keyword. This template parses that matrix
and renders it as a labeled heatmap, with rows/columns labeled by the
fluorescence channels and cells annotated with the spillover coefficient.

The diagonal is always 1.0 (each channel "spills" 100% of itself into
itself). Off-diagonal values are the fraction of channel-i signal that
appears in channel-j's detector — typical values are 0.01–0.30. Reading
this plot tells the user how much spectral overlap exists between their
fluorophores, which informs whether the experiment will need
compensation correction and how much.

Use this when the user asks "what's the spillover", "show me
compensation", "how clean is my panel".

If the file has no `$SPILL` / `$SPILLOVER` keyword (some files don't —
especially older formats or files that lost compensation metadata in
transit), the template raises and the caller falls through to the
agent loop.

`$SPILL` format (per FCS 3.1 spec):
    "N,p1,p2,...,pN,a11,a12,...,aNN"

  - N: integer count of fluorescence channels in the matrix
  - pi: parameter name ($PnN value) for column/row i
  - aij: matrix entry at row i, column j (row-major)

Some vendor files use ";" instead of "," — we accept either.
"""

from __future__ import annotations

NAME = "fcs_compensation_heatmap"
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".fcs",)


def _parse_spill(spill_text: str) -> tuple[list[str], list[list[float]]]:
    """Parse $SPILL string → (channel_names, matrix). Raises ValueError
    on malformed input — caller treats that as "no usable spillover"
    and falls through to the agent."""
    # Some vendors use ";" as separator; normalize.
    raw = spill_text.replace(";", ",").strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("$SPILL is empty")
    try:
        n = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"$SPILL leading count is not an integer: {parts[0]!r}") from exc
    expected_len = 1 + n + n * n
    if len(parts) < expected_len:
        raise ValueError(
            f"$SPILL truncated: N={n} expects {expected_len} fields, got {len(parts)}"
        )
    channels = parts[1 : 1 + n]
    flat = parts[1 + n : 1 + n + n * n]
    matrix = []
    for r in range(n):
        row = []
        for c in range(n):
            try:
                row.append(float(flat[r * n + c]))
            except ValueError as exc:
                raise ValueError(
                    f"$SPILL[{r},{c}] is not a number: {flat[r * n + c]!r}"
                ) from exc
        matrix.append(row)
    return channels, matrix


def render(target_file_path: str, output_path: str) -> None:
    import flowio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fd = flowio.FlowData(target_file_path)
    # flowio normalizes FCS keywords to lowercase keys.
    text = fd.text or {}
    spill_raw = text.get("spill") or text.get("spillover")
    if not spill_raw:
        raise RuntimeError(
            "FCS file has no $SPILL or $SPILLOVER keyword — no embedded "
            "compensation matrix to visualize."
        )

    channels, matrix = _parse_spill(spill_raw)
    arr = np.array(matrix, dtype=float)
    n = arr.shape[0]
    if n == 0:
        raise RuntimeError("$SPILL parsed but contained zero channels")

    # Heatmap scaling: spillover values are typically 0–1 with diagonal=1.
    # Use a colormap that pops on the off-diagonal — viridis works fine
    # because the diagonal saturates and the spillover bleeds stand out.
    fig, ax = plt.subplots(figsize=(0.6 * n + 3, 0.6 * n + 3))
    im = ax.imshow(arr, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="spillover fraction")
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(range(n))
    ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(channels, fontsize=9)
    ax.set_xlabel("detected in (column)", fontsize=10)
    ax.set_ylabel("from (row)", fontsize=10)

    # Annotate each cell with the coefficient. Skip diagonals (always 1.0)
    # to reduce visual noise; readers infer it from the diagonal saturation.
    for r in range(n):
        for c in range(n):
            v = arr[r, c]
            if r == c:
                continue
            # Choose text color for contrast against the cell color.
            txt_color = "white" if v < 0.5 else "black"
            if abs(v) >= 0.005:
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=txt_color)

    fig.suptitle(f"FCS spillover matrix — {n} fluorescence channels", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
