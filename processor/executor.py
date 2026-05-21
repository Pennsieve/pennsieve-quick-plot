"""
Executes the LLM-generated Python script in a subprocess.

Why subprocess (not exec):
  - Isolation: a buggy script can't corrupt the main process state
  - Easy timeout: subprocess.run(..., timeout=N) is straightforward
  - Easy capture of stderr separately from stdout for error feedback to the LLM

The EFS layer `quick-plot-stack` is mounted read-only at /mnt/layers/quick-plot-stack/.
Its Python site-packages are injected into the child process's PYTHONPATH so the
script can import matplotlib, pandas, flowkit, etc.
"""

import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# Location of the EFS layer's site-packages.
#
# Production: the platform injects `layersDir` in the event payload (bridged
# to LAYERS_DIR by handler.py); it points to the per-execution layers root,
# e.g. /mnt/efs/<computeNodeId>/<runId>/layers/. The actual site-packages
# path is constructed as: $LAYERS_DIR/<layerName>/lib/python<version>/site-packages
#
# Local dev: point QUICK_PLOT_STACK_SITE_PACKAGES at a venv path directly.
LAYER_NAME = os.environ.get("QUICK_PLOT_LAYER_NAME", "quick-plot-stack")
LAYER_PYTHON_VERSION = os.environ.get("QUICK_PLOT_LAYER_PYTHON_VERSION", "3.12")


def _resolve_layer_site_packages() -> str:
    """Compute the absolute path to the layer's site-packages directory."""
    # Explicit override wins
    direct = os.environ.get("QUICK_PLOT_STACK_SITE_PACKAGES", "").strip()
    if direct:
        return direct
    # Platform-injected per-run layers dir
    layers_dir = os.environ.get("LAYERS_DIR", "").strip()
    if layers_dir:
        return os.path.join(layers_dir, LAYER_NAME, "lib", f"python{LAYER_PYTHON_VERSION}", "site-packages")
    # Last-resort fallback for older deployments
    return f"/mnt/layers/{LAYER_NAME}/lib/python{LAYER_PYTHON_VERSION}/site-packages"


EFS_LAYER_SITE_PACKAGES = _resolve_layer_site_packages()

EXECUTION_TIMEOUT_SECONDS = int(os.environ.get("SCRIPT_TIMEOUT_SECONDS", "120"))


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    output_exists: bool
    output_size: int


def execute_script(script: str, output_path: str) -> ExecutionResult:
    """
    Run the generated script in a subprocess. Verify the figure file exists afterward.

    Returns ExecutionResult — caller decides whether to retry.
    """
    # Write the script to a temp file (subprocess -c is fine for short scripts,
    # but a file makes tracebacks readable and supports larger scripts).
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    # Compose PYTHONPATH: EFS layer first, then existing path
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    if os.path.isdir(EFS_LAYER_SITE_PACKAGES):
        env["PYTHONPATH"] = f"{EFS_LAYER_SITE_PACKAGES}:{existing_pythonpath}".rstrip(":")
    else:
        log.warning(
            "EFS layer site-packages not found at %s — falling back to ambient Python. "
            "Production runs will fail without the layer mounted.",
            EFS_LAYER_SITE_PACKAGES,
        )

    log.info("Executing generated script (timeout=%ds, output=%s)", EXECUTION_TIMEOUT_SECONDS, output_path)
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT_SECONDS,
            check=False,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        ok = proc.returncode == 0
        if stdout:
            log.info("Script stdout:\n%s", stdout)
        if stderr:
            log.info("Script stderr:\n%s", stderr)
    except subprocess.TimeoutExpired:
        log.error("Script exceeded %ds timeout", EXECUTION_TIMEOUT_SECONDS)
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"Script execution exceeded {EXECUTION_TIMEOUT_SECONDS}-second timeout.",
            output_exists=False,
            output_size=0,
        )
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    output_exists = os.path.isfile(output_path)
    output_size = os.path.getsize(output_path) if output_exists else 0

    if ok and not output_exists:
        # Script exited cleanly but didn't produce the figure — treat as failure
        stderr = (stderr + "\n\nScript completed but did not write the expected output file: " + output_path).strip()
        ok = False

    return ExecutionResult(
        success=ok,
        stdout=stdout,
        stderr=stderr,
        output_exists=output_exists,
        output_size=output_size,
    )
