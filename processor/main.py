"""
Quick-plot processor — ECS/local entry point.

Reads the target file path + optional template + optional user prompt
from env vars (or, in Lambda mode, from event-payload-bridged env vars
set by handler.py) and produces a matplotlib figure at
`/OUTPUT_DIR/figure.png`.

Two render paths share this single processor:

  1. **Canned template** (cheap, deterministic). When TEMPLATE is set to
     a known identifier (e.g. `fcs_channel_histograms`), the processor
     calls the matching template's render() function. No LLM, ~5s.

  2. **LLM agent loop** (flexible, expensive). Falls back when TEMPLATE
     is unset, unknown, or the canned render raised. Uses bash /
     read_file / write_file tools via the LLM Governor to inspect the
     file and write matplotlib code. ~30s, ~$0.15-0.19.

History: an earlier design split these two paths across two separate
processor stages (pennsieve-plot-templates + pennsieve-quick-plot) with
an inter-stage data-flow contract on `OUTPUT_DIR/figure.png`. That
turned out fragile (workdirs are per-stage, input passthrough is
implicit, several production failures from the data-flow protocol). The
single-stage in-process dispatch removes the protocol entirely — both
paths see the same INPUT_DIR (the original file) and write to the same
OUTPUT_DIR. Adding a new template = one new module in
`processor/templates/`; no workflow or MCP change.

The agent loop itself was already a hard rewrite of an earlier
single-shot script-generation backend (see processor/agent.py for the
full rationale). In short: single-shot generation hallucinated columns
/ schemas on novel files. An agent that can `head file.csv` or
`python -c "..."` before writing the plot grounds every assumption in
real data and generalizes to any supported file type without per-type
preview extractors.
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
        "template": os.environ.get("TEMPLATE", ""),
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


def try_canned_template(config: dict, target_file_path: str, output_path: str) -> bool:
    """
    Run the canned template named in config["template"], if any.

    Returns True when a template was selected AND it produced figure.png.
    Returns False when:
      - TEMPLATE wasn't set (caller didn't ask for a template),
      - TEMPLATE was set to an unknown name (caller asked for a template
        we don't have),
      - TEMPLATE's render() raised (caller asked for one we have but it
        broke on this file).

    All "no canned figure" outcomes are reported via False rather than an
    exception — the caller's contract is to fall through to the agent
    loop on any of them. Exceptions are logged loudly for CloudWatch
    discoverability. Any partial figure.png left behind by a failing
    render is cleaned up so the data-target stage doesn't see a
    half-baked file.
    """
    template_name = config["template"]
    if not template_name:
        return False

    # Put the EFS layer's site-packages on sys.path so the template's
    # imports (fcsparser, matplotlib, etc.) resolve. The agent loop
    # doesn't need this because it runs its generated scripts as a
    # subprocess with PYTHONPATH set on the child env; the templates
    # import in-process and need the path on the parent interpreter.
    setup_layer_python_path()

    from processor.templates import get as get_template, known_names

    template = get_template(template_name)
    if template is None:
        log.warning(
            "Unknown template %r. Known templates: %s. Falling back to agent loop.",
            template_name, ", ".join(known_names()),
        )
        return False

    log.info("Trying canned template: %s", template.NAME)
    ext = os.path.splitext(target_file_path)[1].lower()
    if ext and ext not in template.SUPPORTED_EXTENSIONS:
        log.warning(
            "File extension %s isn't in template %s's declared supported set %s — attempting anyway",
            ext, template.NAME, template.SUPPORTED_EXTENSIONS,
        )

    try:
        template.render(target_file_path, output_path)
    except Exception as exc:  # noqa: BLE001
        log.exception(
            "Template %s raised during render: %s — falling back to agent loop.",
            template.NAME, exc,
        )
        if os.path.isfile(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return False

    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        return True

    # render() didn't raise but also didn't write anything — treat the
    # same as a soft failure; agent fallback.
    log.warning(
        "Template %s completed without writing %s — falling back to agent loop.",
        template.NAME, output_path,
    )
    return False


def setup_layer_python_path() -> None:
    """
    Prepend the shared EFS layer's site-packages to sys.path so canned
    templates can `import fcsparser` / `import matplotlib` directly.
    Idempotent. Mirrors processor/executor.py's resolve_layer_site_packages
    so both the agent's child PYTHONPATH and the canned in-process path
    look at the same place on the same volume.
    """
    from processor.executor import resolve_layer_site_packages

    sp = resolve_layer_site_packages()
    if not os.path.isdir(sp):
        log.warning(
            "Layer site-packages not found at %s — canned templates that "
            "need extra deps will fail on import (agent fallback unaffected).",
            sp,
        )
        return
    if sp not in sys.path:
        sys.path.insert(0, sp)
        log.info("Added layer site-packages to sys.path: %s", sp)


def run():
    start = time.time()
    config = get_config()

    log.info("=" * 60)
    log.info("Quick-plot processor")
    log.info("Run ID:  %s", config["execution_run_id"] or "(not set)")
    log.info("Runtime: %s", "Lambda" if os.environ.get("AWS_LAMBDA_RUNTIME_API") else "ECS/Local")
    log.info("Template: %s", config["template"] or "(unset — agent path)")
    log.info("=" * 60)

    # Validate config
    if not config["input_dir"] or not config["output_dir"]:
        log.error("INPUT_DIR and OUTPUT_DIR are required")
        sys.exit(1)
    if not config["template"] and not config["prompt"] and not STUB_MODE:
        log.error(
            "At least one of TEMPLATE or PROMPT is required — TEMPLATE picks "
            "a canned per-format script, PROMPT drives the LLM agent loop."
        )
        sys.exit(1)

    os.makedirs(config["output_dir"], exist_ok=True)

    # Resolve input file (used by both paths)
    target_file_path = resolve_target_file(config["input_dir"], config["target_file_name"])
    output_path = os.path.join(config["output_dir"], FIGURE_FILENAME)

    log.info("Target file: %s", target_file_path)
    log.info("Figure → %s", output_path)

    # Path 1: try the canned template first if one was requested.
    if try_canned_template(config, target_file_path, output_path):
        size = os.path.getsize(output_path)
        log.info(
            "Figure produced by canned template '%s': %s (%d bytes, %.2fs total)",
            config["template"], output_path, size, time.time() - start,
        )
        return

    # Path 2: agent loop fallback. Requires PROMPT. When a TEMPLATE was
    # selected but failed, MCP synthesizes a generic prompt so the agent
    # has something to act on; pure-template invocations without a prompt
    # caught at the validate-config step above.
    if not config["prompt"] and not STUB_MODE:
        log.error(
            "Canned template didn't produce a figure and no PROMPT was provided — "
            "nothing for the agent to do."
        )
        sys.exit(1)

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
    from processor.executor import resolve_layer_site_packages

    log.info("Prompt: %s", config["prompt"])

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
            "Figure produced by agent: %s (%d bytes, %d iterations, %.2fs total)",
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
