"""
Quick-plot processor — ECS/local entry point.

Reads the target file path + user prompt from env vars (or, in Lambda mode,
from event-payload-bridged env vars set by handler.py), asks the LLM Governor
for a Python plotting script, runs the script in a subprocess against the EFS
scientific-Python layer, and writes /OUTPUT_DIR/figure.png.

On script error, retries ONCE with the traceback fed back to the LLM.
"""

import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("quick-plot")

MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "1"))
FIGURE_FILENAME = "figure.png"
SCRIPT_FILENAME = "script.py"


def get_config():
    return {
        "input_dir": os.environ.get("INPUT_DIR", ""),
        "output_dir": os.environ.get("OUTPUT_DIR", ""),
        "prompt": os.environ.get("PROMPT", ""),
        "target_file_name": os.environ.get("TARGET_FILE_NAME", ""),
        "execution_run_id": os.environ.get("EXECUTION_RUN_ID", ""),
        "llm_governor_url": os.environ.get("LLM_GOVERNOR_URL", ""),
    }


def resolve_target_file(input_dir: str, hint: str) -> str:
    """
    Find the target file. If TARGET_FILE_NAME is set, prefer that. Otherwise pick
    the first file in INPUT_DIR (data-source typically stages only one).
    """
    if hint:
        candidate = os.path.join(input_dir, hint)
        if os.path.isfile(candidate):
            return candidate
        log.warning("TARGET_FILE_NAME=%s not found in INPUT_DIR; falling back to first file", hint)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"INPUT_DIR does not exist: {input_dir}")

    entries = sorted(
        os.path.join(input_dir, e) for e in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, e)) or os.path.islink(os.path.join(input_dir, e))
    )
    if not entries:
        raise FileNotFoundError(f"No files in INPUT_DIR: {input_dir}")
    return entries[0]


def run():
    start = time.time()
    config = get_config()

    log.info("=" * 60)
    log.info("Quick-plot processor")
    log.info("Run ID: %s", config["execution_run_id"] or "(not set)")
    log.info("Runtime: %s", "Lambda" if os.environ.get("AWS_LAMBDA_RUNTIME_API") else "ECS/Local")
    log.info("=" * 60)

    # Validate config
    if not config["input_dir"] or not config["output_dir"]:
        log.error("INPUT_DIR and OUTPUT_DIR are required")
        sys.exit(1)
    if not config["prompt"]:
        log.error("PROMPT is required (the user's plain-English request)")
        sys.exit(1)

    os.makedirs(config["output_dir"], exist_ok=True)

    # Resolve input file
    target_file_path = resolve_target_file(config["input_dir"], config["target_file_name"])
    target_file_name = os.path.basename(target_file_path)
    output_path = os.path.join(config["output_dir"], FIGURE_FILENAME)
    script_out_path = os.path.join(config["output_dir"], SCRIPT_FILENAME)

    log.info("Target file: %s", target_file_path)
    log.info("Prompt: %s", config["prompt"])
    log.info("Figure → %s", output_path)

    # Lazy import so the module loads even without pennsieve_llm installed in dev contexts
    from processor.llm import generate_script
    from processor.executor import execute_script

    previous_error = None
    previous_script = None

    for attempt in range(MAX_RETRIES + 1):
        log.info("--- Attempt %d / %d ---", attempt + 1, MAX_RETRIES + 1)

        try:
            script = generate_script(
                user_prompt=config["prompt"],
                target_file_path=target_file_path,
                target_file_name=target_file_name,
                output_path=output_path,
                previous_error=previous_error,
                previous_script=previous_script,
            )
        except Exception as e:
            log.error("LLM call failed: %s", e)
            sys.exit(1)

        # Persist the script for transparency / debugging — overwritten each attempt
        with open(script_out_path, "w") as f:
            f.write(script)

        result = execute_script(script, output_path)
        if result.success:
            log.info(
                "Figure produced: %s (%d bytes, %.2fs total)",
                output_path, result.output_size, time.time() - start
            )
            log.info("Script saved: %s", script_out_path)
            return

        log.warning("Attempt %d failed", attempt + 1)
        previous_error = result.stderr
        previous_script = script

    log.error("All %d attempts failed. Last error:\n%s", MAX_RETRIES + 1, previous_error)
    sys.exit(1)


if __name__ == "__main__":
    run()
