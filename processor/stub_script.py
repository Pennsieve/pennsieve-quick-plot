"""
Stub script used when STUB_MODE=1 to smoke-test the processor end-to-end
without calling the LLM. Reads the target file, makes a simple plot,
saves it. Format-agnostic enough to work on CSV / TSV / FCS / h5ad.

Used during early dev to validate the EFS-layer mount + viewer-asset
attachment chain before the LLM integration is exercised.
"""

STUB_SCRIPT_TEMPLATE = '''\
"""Auto-generated stub script. Plots the first numeric column as a histogram."""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET = os.environ.get("STUB_TARGET_FILE", "{target_file_path}")
OUT = os.environ.get("STUB_OUTPUT_PATH", "{output_path}")

def load_dataframe(path):
    """Try a few readers depending on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv", ".txt"):
        import pandas as pd
        sep = "," if ext == ".csv" else "\\t"
        return pd.read_csv(path, sep=sep, on_bad_lines="skip", nrows=100000)
    if ext == ".parquet":
        import pandas as pd
        return pd.read_parquet(path)
    if ext == ".fcs":
        import flowkit
        sample = flowkit.Sample(path)
        return sample.as_dataframe()
    if ext in (".h5ad",):
        import anndata as ad
        adata = ad.read_h5ad(path)
        # Use obs as a tabular view for a quick stub plot
        return adata.obs.copy()
    raise ValueError(f"Unsupported file extension: {{ext}}")


def main():
    print(f"STUB: target={{TARGET}}, output={{OUT}}", flush=True)
    df = load_dataframe(TARGET)
    print(f"STUB: loaded shape={{df.shape}}, cols={{list(df.columns)[:10]}}", flush=True)

    # First numeric column
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] == 0:
        # Fallback: row-count bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(["rows"], [len(df)])
        ax.set_title(f"{{os.path.basename(TARGET)}} \\u2014 row count (no numeric columns)")
        ax.set_ylabel("rows")
    else:
        col = numeric.columns[0]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(numeric[col].dropna(), bins=40, color="#4C78A8", edgecolor="white")
        ax.set_title(f"{{os.path.basename(TARGET)}} \\u2014 {{col}} (stub histogram)")
        ax.set_xlabel(col)
        ax.set_ylabel("count")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"STUB: wrote {{OUT}} ({{os.path.getsize(OUT)}} bytes)", flush=True)


if __name__ == "__main__":
    main()
'''


def build_stub_script(target_file_path: str, output_path: str) -> str:
    """Return a ready-to-execute Python script with paths substituted in."""
    return STUB_SCRIPT_TEMPLATE.format(
        target_file_path=target_file_path,
        output_path=output_path,
    )
