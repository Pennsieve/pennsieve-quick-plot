"""
tsv_sessions_lineplot — patient age across sessions, as a line plot.

Reads a sessions TSV, pulls the session identifier and a subject-age
column, sorts the sessions by age (ascending), and draws a single line
plot of age per session in an academic style (black axes, ticks + tick
labels, no gridlines, no background fill).

Use this when the user asks to "plot age across sessions", "show patient
age per session", "how does age change over sessions".

Column discovery, for both x and y: first an exact preferred name
(case-insensitive), else a name-substring match whose values are all
unique. Requiring uniqueness on the fallback skips lookalike columns like
"session_status" that repeat and aren't identifiers.
  - x: "session_id" / "session_name" / "session", else a unique "session"* col
  - y: "subject_age",                              else a unique "age"* col

If there is no age column, or every age value is missing / non-numeric,
the template raises and the caller falls through to the agent loop. Same
for a missing session column, or when the number of plottable sessions is
outside [MIN_SESSIONS, MAX_SESSIONS] (too few to form a line, or too many
to label readably).

Sessions files are always TSV, so the separator is fixed (tab) — unlike
the CSV template there's no extension sniffing.

Deterministic: same input → same output. Rows are sorted by age (ties
broken by session label) so re-renders of the same file are identical.
"""


############# SET UP ##################
from __future__ import annotations

NAME = "tsv_sessions_lineplot"
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".tsv",)
SUMMARY = "subject age across visit sessions as a line plot; \"plot age across sessions\", \"how does age change over sessions\""

# Bounds on the number of plottable sessions (checked on the cleaned data,
# i.e. after dropping rows with no usable age). Below MIN there's no line to
# draw; above MAX the x-axis becomes an unreadable pile of labels. Both raise
# so the caller falls through to the agent loop.
MIN_SESSIONS = 2
MAX_SESSIONS = 20



############# HELPERS ##################
def _find_column(df, *, prefer: list[str], contains: str) -> str | None:
    """Return the best-matching column name, or None.

    Two-step match:
      1. `prefer` is a list of exact (case-insensitive) names tried in
         priority order — the first column that equal to one of them AND the column that has all-unique values
         wins. An exact-named column is trusted as-is.
      2. `contains` is a vague match. If none of the columns match the ones in the 
         `prefer` list, then the code would fall back to any column whose name contains
         `contains` AND whose values are all unique. 

    Examples (columns shown in order):
      prefer=["session_id","session_name","session"], 
      contains="session"
        columns = ["session_id", "session_status"]  -> return "session_id"    (exact match AND unique id)
        columns = ["session_if_sleep_detected"]     -> return None            (non-unique, not an id)
        columns = ["visit_session"]                 -> return "visit_session" (unique fallback)
    """
    columns = list(df.columns)
    for name in prefer:
        for c in columns:
            if c.lower() == name and df[c].dropna().is_unique:
                return c
    for c in columns:
        if contains in c.lower() and df[c].dropna().is_unique:
            return c
    return None



############### PLOTTING ##################
def render(target_file_path: str, output_path: str) -> None:


####### IMPORTS
    import matplotlib
    matplotlib.use("Agg")  
    import matplotlib.pyplot as plt
    import pandas as pd


####### LOAD DATA
    df = pd.read_csv(target_file_path, sep="\t", low_memory=False)
    if df.empty:
        raise RuntimeError("TSV parsed but contained no rows")


####### CLEAN DATA
    # find the session column: exact 'session_id'/'session_name'/'session',
    # else a 'session'-named column whose values are unique (an identifier).
    session_col = _find_column(
        df, prefer=["session_id", "session_name", "session"], contains="session"
    )
    if session_col is None:
        raise RuntimeError(
            "No session column found — expected a column named 'session_id', "
            "'session_name', or 'session', or a 'session'-named column whose "
            "values are unique enough to act as the identifier."
        )
    # find the age column: find 'subject_age', "session_age", "visit_age", else a unique 'age'-named column.
    age_col = _find_column(df, prefer=["subject_age", "session_age", "visit_age"], contains="age")
    if age_col is None:
        raise RuntimeError(
            "No age column found — expected a column whose name contains "
            "'age' (e.g. 'subject_age')."
        )

    # Coerce age to numeric; anything non-numeric (e.g. "n/a") becomes NaN.
    # Drop any row missing either session or age
    # stringify the surviving session.
    ages = pd.to_numeric(df[age_col], errors="coerce")
    data = pd.DataFrame({"session": df[session_col], "age": ages})
    data = data.dropna(subset=["session", "age"])
    data["session"] = data["session"].astype(str) # needs to dropna first then stringfy 
    if data.empty:
        raise RuntimeError(
            f"Age column {age_col!r} has no usable numeric values — every "
            "row was missing / NaN / non-numeric."
        )

    # Bound the number of plottable sessions. Below MIN there's no line;
    # above MAX the x-axis is an unreadable pile of labels.
    if len(data) < MIN_SESSIONS:
        raise RuntimeError(
            f"Only {len(data)} session has a usable age — need at least "
            f"{MIN_SESSIONS} to draw a line plot."
        )
    if len(data) > MAX_SESSIONS:
        raise RuntimeError(
            f"{len(data)} sessions with usable ages exceeds the max of "
            f"{MAX_SESSIONS} — too many for a readable line plot."
        )

    # Rank sessions by age, small → large (ties broken by label so the
    # order is fully deterministic).
    data = data.sort_values(["age", "session"], kind="stable").reset_index(drop=True)


####### DRAW LINEPLOT
    x_labels = data["session"].tolist()
    y_values = data["age"].to_numpy()
    x_positions = list(range(len(x_labels)))

    # Draw at a provisional size, then resize to the real tick labels
    #   We can't know how wide the tick labels render until matplotlib lays
    #   them out, so: draw once, measure the actual label extents, then set
    #   the figure size from those physical measurements.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_positions, y_values, color="#2a6c97", marker="o", linewidth=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(session_col)
    ax.set_ylabel(age_col)
    ax.set_title("Patient age for each session")

    # Academic styling: black left/bottom axes, no top/right frame, ticks
    # pointing out, no gridlines, white background.
    ax.grid(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(axis="both", direction="out", color="black", labelsize=9)

    # Measure the rendered tick labels (in inches) so the axis physical
    # length scales with label size: short labels ("1", "2") → compact;
    # long labels ("preimplant") → wider so they don't collide.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi

    x_lbls = ax.get_xticklabels()
    max_x_w_in = max((t.get_window_extent(renderer).width for t in x_lbls), default=0) / dpi
    # Each x slot must be at least as wide as the widest label, plus a gap.
    per_x_in = max_x_w_in + 0.15
    fig_w = max(4.0, min(30.0, len(x_positions) * per_x_in + 1.4))

    y_lbls = ax.get_yticklabels()
    max_y_h_in = max((t.get_window_extent(renderer).height for t in y_lbls), default=10) / dpi
    # Give each y tick ~2.2x its label height so values aren't cramped.
    per_y_in = max_y_h_in * 2.2
    fig_h = max(3.0, min(20.0, len(y_lbls) * per_y_in + 1.4))

    fig.set_size_inches(fig_w, fig_h)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
