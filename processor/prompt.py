"""
System + user prompt templates for the quick-plot LLM call.

The model is asked to return ONLY a complete Python script. No prose, no
markdown fences, no explanation. The script must:
  - read from the file path provided
  - produce a matplotlib figure saved to OUTPUT_DIR/figure.png
  - use libraries available on the EFS layer `quick-plot-stack`
"""

SYSTEM_PROMPT = """\
You write Python scripts that generate matplotlib figures from scientific data files.

Constraints:
- Output ONLY a complete, self-contained Python script. No markdown fences, no commentary.
- The script reads input from the file at TARGET_FILE_PATH (provided in the user message).
- The script saves a single matplotlib figure to OUTPUT_PATH (provided in the user message).
- Use `matplotlib.use("Agg")` before importing pyplot — there is no display.
- Save with `plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")`.
- The following libraries are available: matplotlib, pandas, numpy, scipy, seaborn,
  flowkit (FCS), fcsparser, anndata, scanpy (h5ad / Seurat-via-h5ad), nibabel (NIfTI),
  pillow, tifffile, pyreadr (RDS), h5py, pyarrow.
- Pick the right reader for the file extension. If you can't read the file with any
  available library, print a clear error and exit non-zero.
- If the user's request needs specific columns / markers / genes that you can't see in
  the data, print what you see (column names, shape) and exit non-zero rather than guess.
- Keep the figure simple and labeled. Title, axis labels, legend if applicable.
- Do not use `plt.show()`.
"""


def build_user_message(prompt: str, target_file_path: str, target_file_name: str, output_path: str) -> str:
    """Build the user message that ships with the file as an efs_document content block."""
    return (
        f"User request: {prompt}\n\n"
        f"TARGET_FILE_PATH = {target_file_path}\n"
        f"TARGET_FILE_NAME = {target_file_name}\n"
        f"OUTPUT_PATH = {output_path}\n\n"
        f"The file is also attached as an efs_document below — you can inspect its "
        f"structure (headers, shape, column names) before writing the script."
    )


def build_retry_message(error_text: str) -> str:
    """Build a follow-up message asking the model to fix a script that failed to run."""
    return (
        "The script you returned failed when executed. The error was:\n\n"
        f"```\n{error_text}\n```\n\n"
        "Return a corrected complete script. Same output constraints as before "
        "(no markdown fences, save to OUTPUT_PATH, use Agg backend)."
    )
