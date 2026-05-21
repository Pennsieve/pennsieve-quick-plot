"""
Quick-plot processor — ECS/local entry point.

Reads the target file path + user prompt from env vars (or, in Lambda mode,
from event-payload-bridged env vars set by handler.py), runs an agent loop
that uses bash/read_file/write_file tools (via the LLM Governor) to inspect
the input file and generate a matplotlib figure at /OUTPUT_DIR/figure.png.

The agent loop replaces the previous single-shot script-generation backend
(see processor/agent.py for the full rationale). In short: single-shot
generation hallucinated columns / schemas on novel files. An agent that can
`head file.csv` or `python -c "..."` before writing the plot grounds every
assumption in real data and generalizes to any supported file type without
per-type preview extractors.
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

FIGURE_FILENAME = "figure.png"

# When set to "1", bypass the LLM entirely and use the built-in stub script
# in processor/stub_script.py. Used for smoke-tests of the EFS layer mount +
# viewer-asset data-target chain without burning LLM tokens.
STUB_MODE = os.environ.get("STUB_MODE", "") == "1"


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
    if not config["prompt"] and not STUB_MODE:
        log.error("PROMPT is required (the user's plain-English request)")
        sys.exit(1)

    os.makedirs(config["output_dir"], exist_ok=True)

    # Resolve input file
    target_file_path = resolve_target_file(config["input_dir"], config["target_file_name"])
    output_path = os.path.join(config["output_dir"], FIGURE_FILENAME)

    log.info("Target file: %s", target_file_path)
    log.info("Prompt: %s", config["prompt"])
    log.info("Figure → %s", output_path)

    # Stub mode short-circuits the agent loop. Used to smoke-test the
    # EFS-layer mount + viewer-asset attachment without the LLM in the loop.
    if STUB_MODE:
        log.info("STUB_MODE=1 — bypassing agent, using built-in stub script")
        from processor.executor import execute_script
        from processor.stub_script import build_stub_script

        script = build_stub_script(target_file_path, output_path)
        result = execute_script(script, output_path)
        if result.success:
            log.info(
                "Figure produced: %s (%d bytes, %.2fs total)",
                output_path, result.output_size, time.time() - start,
            )
            return
        log.error("Stub script failed:\n%s", result.stderr)
        sys.exit(1)

    # Agent loop — inspect + plot via tool calls.
    from processor.agent import run_agent
    from processor.executor import resolve_layer_site_packages  # reuse path resolver

    # Workdir = OUTPUT_DIR. The agent saves the final figure there and may
    # write intermediate scripts / artifacts alongside it. Everything in the
    # workdir is harvested by the figure-asset data-target after we exit.
    workdir = config["output_dir"]

    result = run_agent(
        user_prompt=config["prompt"],
        target_file_path=target_file_path,
        output_path=output_path,
        workdir=workdir,
        layer_site_packages=resolve_layer_site_packages(),
    )

    if result.success:
        log.info(
            "Figure produced: %s (%d bytes, %d iterations, %.2fs total)",
            output_path, result.output_size, result.iterations, time.time() - start,
        )
        if result.final_text:
            log.info("Agent summary: %s", result.final_text)
        return

    log.error(
        "Agent loop failed after %d iterations: %s",
        result.iterations, result.error or "(no error message)",
    )
    sys.exit(1)


if __name__ == "__main__":
    run()
