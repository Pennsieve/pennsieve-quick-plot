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

  2. **LLM agent loop** (flexible, expensive). Runs when TEMPLATE is
     unset, or when the canned render failed for a reason the user can't
     fix (environment / internal — see FAIL_FAST below). Template failures
     the user can act on (invalid_input, data_unavailable, resource_limit)
     stop the run with the template's own message instead. Uses bash /
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

import json
import logging
import os
import sys
import time
from dataclasses import dataclass

from processor.errors import (
    GENERIC_INTERNAL_MESSAGE,
    PlotDataUnavailableError,
    PlotEnvironmentError,
    PlotError,
    PlotErrorCategory,
    PlotErrorCode,
    PlotErrorStage,
    PlotInternalError,
    PlotInvalidInputError,
    wrap_unexpected_exception,
)
from processor.report import PlotRunReport, push_report_to_workflow_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("quick-plot")

FIGURE_FILENAME = "figure.png"

# The gate's policy: when the canned template fails with
# one of these categories, stop and show the template's own message. The
# user can act on it (fix the request, pick other data, ask for less), and
# a ~30 s / ~$0.15 agent detour would only hide it. environment / internal
# failures fall back to the agent, which may still rescue the request.
FAIL_FAST = frozenset({
    PlotErrorCategory.INVALID_INPUT,
    PlotErrorCategory.DATA_UNAVAILABLE,
    PlotErrorCategory.RESOURCE_LIMIT,
})

# When set to "1", bypass the LLM entirely and use the built-in stub script
# in processor/stub_script.py. Used for smoke-tests of the EFS layer mount +
# viewer-asset data-target chain without burning LLM tokens.
STUB_MODE = os.environ.get("STUB_MODE", "") == "1"


def get_config():
    return {
        "input_dir": os.environ.get("INPUT_DIR", ""),
        "output_dir": os.environ.get("OUTPUT_DIR", ""),
        "template": os.environ.get("TEMPLATE", ""),
        # Per-template render args as a JSON object string (or "" for none).
        # Parsed + splatted into the template's render() by try_canned_template.
        "template_args": os.environ.get("TEMPLATE_ARGS", ""),
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
        # The provisioner stages the input dir; its absence is a platform problem.
        raise PlotEnvironmentError(
            PlotErrorCode.CONFIG_MISSING_DIRS, f"INPUT_DIR does not exist: {input_dir}",
            source_stage=PlotErrorStage.CONFIG)

    entries = sorted(
        os.path.join(input_dir, e) for e in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, e)) or os.path.islink(os.path.join(input_dir, e))
    )
    if not entries:
        raise PlotDataUnavailableError(
            PlotErrorCode.NO_INPUT_FILES, "No input file was staged for this run.",
            error_facts={"input_dir": input_dir}, source_stage=PlotErrorStage.CONFIG)
    return entries[0]


@dataclass
class CannedOutcome:
    """What try_canned_template() came back with.

    produced   figure.png exists and is non-empty
    error      why not (None when produced, or when no TEMPLATE was asked for)
    """

    produced: bool = False
    error: PlotError | None = None


def try_canned_template(config: dict, target_file_path: str, output_path: str) -> CannedOutcome:
    """
    Run the canned template named in config["template"], if any.

    Returns CannedOutcome(produced=True) when a template was selected AND it
    produced figure.png. Otherwise `error` says why, as a PlotError, and
    the caller (run) decides whether the agent loop gets a turn: FAIL_FAST
    categories do not fall back — the user gets the template's own message
    instead of a 30 s detour — everything else does, with the template
    error recorded on the report as `fallback_trigger_error`.

    TEMPLATE unset → CannedOutcome() with no error: the caller didn't ask
    for a template. Any partial figure.png left behind by a failing render
    is removed so the data-target stage doesn't see a half-baked file.
    """
    template_name = config["template"]
    if not template_name:
        return CannedOutcome()

    # Put the EFS layer's site-packages on sys.path so the template's
    # imports (fcsparser, matplotlib, etc.) resolve. The agent loop
    # doesn't need this because it runs its generated scripts as a
    # subprocess with PYTHONPATH set on the child env; the templates
    # import in-process and need the path on the parent interpreter.
    setup_layer_python_path()

    from processor.templates import get as get_template, known_names

    template = get_template(template_name)
    if template is None:
        return CannedOutcome(error=PlotInvalidInputError(
            PlotErrorCode.UNKNOWN_TEMPLATE,
            f"Unknown plot template {template_name!r}. Known templates: {', '.join(known_names())}.",
            error_facts={"known_templates": list(known_names())},
            source_stage=PlotErrorStage.TEMPLATE,
        ))

    log.info("Trying canned template: %s", template.NAME)
    ext = os.path.splitext(target_file_path)[1].lower()
    if ext and ext not in template.SUPPORTED_EXTENSIONS:
        log.warning(
            "File extension %s isn't in template %s's declared supported set %s — attempting anyway",
            ext, template.NAME, template.SUPPORTED_EXTENSIONS,
        )

    # Per-template render args arrive as a JSON object string (TEMPLATE_ARGS,
    # set by MCP's plot_file from its `template_args` param). Templates that
    # take no extra args get an empty dict and render(path, out) as before.
    render_kwargs = {}
    template_args = config.get("template_args", "")
    if template_args:
        try:
            parsed = json.loads(template_args)
        except (ValueError, TypeError) as exc:
            return CannedOutcome(error=PlotInvalidInputError(
                PlotErrorCode.TEMPLATE_ARGS_INVALID_JSON,
                f"The settings for template {template_name!r} are not valid JSON: {exc}",
                source_stage=PlotErrorStage.TEMPLATE,
            ))
        if not isinstance(parsed, dict):
            return CannedOutcome(error=PlotInvalidInputError(
                PlotErrorCode.TEMPLATE_ARGS_NOT_OBJECT,
                f"The settings for template {template_name!r} must be a JSON object, "
                f"got {type(parsed).__name__}.",
                source_stage=PlotErrorStage.TEMPLATE,
            ))
        render_kwargs = parsed

    try:
        template.render(target_file_path, output_path, **render_kwargs)
    except Exception as exc:  # noqa: BLE001
        err = wrap_unexpected_exception(exc)      # first of the two catch sites
        err.source_stage = err.source_stage or PlotErrorStage.TEMPLATE
        log.warning("Template %s raised during render [%s/%s]: %s",
                    template.NAME, err.error_category.value, err.error_code.value, exc,
                    exc_info=True)
        if os.path.isfile(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return CannedOutcome(error=err)

    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        return CannedOutcome(produced=True)

    # render() didn't raise but also didn't write anything — an internal
    # template bug; the agent may still rescue the request.
    return CannedOutcome(error=PlotInternalError(
        PlotErrorCode.TEMPLATE_NO_OUTPUT,
        f"The {template.NAME} template finished without producing a figure.",
        source_stage=PlotErrorStage.TEMPLATE,
    ))


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


def run(report: PlotRunReport | None = None):
    """Produce OUTPUT_DIR/figure.png or raise PlotError.

    `report` is filled in as the run progresses (which route drew the
    figure, its size); main() owns pushing it out.
    """
    report = report if report is not None else PlotRunReport()
    start = time.time()
    config = get_config()
    report.requested_template = config["template"]

    log.info("=" * 60)
    log.info("Quick-plot processor")
    log.info("Run ID:  %s", config["execution_run_id"] or "(not set)")
    log.info("Runtime: %s", "Lambda" if os.environ.get("AWS_LAMBDA_RUNTIME_API") else "ECS/Local")
    log.info("Template: %s", config["template"] or "(unset — agent path)")
    log.info("=" * 60)

    # Validate config. Both are platform problems, not user ones: the
    # provisioner sets the dirs; MCP's plot_file guarantees template/prompt.
    if not config["input_dir"] or not config["output_dir"]:
        raise PlotEnvironmentError(
            PlotErrorCode.CONFIG_MISSING_DIRS,
            "The plot runner was started without INPUT_DIR/OUTPUT_DIR.",
            source_stage=PlotErrorStage.CONFIG)
    if not config["template"] and not config["prompt"] and not STUB_MODE:
        raise PlotInvalidInputError(
            PlotErrorCode.CONFIG_NO_REQUEST,
            "Nothing to plot: neither a template nor a prompt was given.",
            source_stage=PlotErrorStage.CONFIG)

    os.makedirs(config["output_dir"], exist_ok=True)

    # Resolve input file (used by both paths)
    target_file_path = resolve_target_file(config["input_dir"], config["target_file_name"])
    output_path = os.path.join(config["output_dir"], FIGURE_FILENAME)

    log.info("Target file: %s", target_file_path)
    log.info("Figure → %s", output_path)

    # Path 1: try the canned template first if one was requested.
    canned = try_canned_template(config, target_file_path, output_path)
    if canned.produced:
        size = os.path.getsize(output_path)
        log.info(
            "Figure produced by canned template '%s': %s (%d bytes, %.2fs total)",
            config["template"], output_path, size, time.time() - start,
        )
        report.figure_generated_by, report.figure_bytes = "template", size
        return

    # The gate. A template failure the *user* can fix —
    # wrong channel, window outside the recording, bad settings — stops
    # here with the template's own message. Spending ~30 s / ~$0.15 on the
    # agent would only hide that message behind a different plot, or a
    # second failure for an unrelated reason.
    if canned.error is not None:
        if canned.error.error_category in FAIL_FAST:
            raise canned.error
        # environment / internal: the agent may still rescue the request.
        # Keep the reason so the user learns why the canned plot didn't
        # happen even when the agent succeeds.
        report.fallback_trigger_error = canned.error
        log.warning("Template failed [%s/%s]; falling back to agent loop.",
                    canned.error.error_category.value, canned.error.error_code.value)

    # Path 2: agent loop fallback. Requires PROMPT. When a TEMPLATE was
    # selected but failed, MCP synthesizes a generic prompt so the agent
    # has something to act on; pure-template invocations without a prompt
    # caught at the validate-config step above.
    if not config["prompt"] and not STUB_MODE:
        if canned.error is not None:
            # The real reason beats "no prompt". No fallback was attempted,
            # so it is the run's failure, not a fallback trigger.
            report.fallback_trigger_error = None
            raise canned.error
        raise PlotInvalidInputError(
            PlotErrorCode.TEMPLATE_FAILED_NO_PROMPT,
            f"The '{config['template']}' template could not produce a figure "
            "and no prompt was given for the agent to try instead.",
            source_stage=PlotErrorStage.TEMPLATE)

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
            report.figure_generated_by, report.figure_bytes = "stub", result.output_size
            return
        raise PlotInternalError(
            PlotErrorCode.STUB_FAILED, "The stub plot script failed.",
            error_facts={"stderr_tail": (result.stderr or "")[-500:]},
            source_stage=PlotErrorStage.STUB)

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
        report.figure_generated_by, report.figure_bytes = "agent", result.output_size
        return

    raise PlotInternalError(
        PlotErrorCode.AGENT_FAILED,
        "The plot agent could not produce a figure for this request.",
        error_facts={"iterations": result.iterations,
                     "agent_error": (result.error or "")[:500]},
        source_stage=PlotErrorStage.AGENT)


def main() -> int:
    """Single choke point for every failure. Returns the process exit code.

    `run()` raises a PlotError subclass for the failures it recognises;
    anything else that escapes is wrapped here. Either way the category,
    code and user message are logged once, in one shape, and the process
    exits 1. The report is pushed to workflow-service from the `finally`
    on every path, success included.
    """
    started = time.time()
    report = PlotRunReport(run_id=os.environ.get("EXECUTION_RUN_ID", ""))
    code = 0                      # assume success until a failure is caught
    try:
        run(report)               # run() fills the report as it goes
        report.run_status = "succeeded"
    # Known failure: run() raised a PlotError for something it recognises
    except PlotError as err:
        err.source_stage = err.source_stage or PlotErrorStage.RUN
        report.run_failure_error = err
        code = 1
    # Unknown failure: anything else that escaped run(). Second catch site.
    except Exception as exc:  # noqa: BLE001
        err = wrap_unexpected_exception(exc)
        err.source_stage = err.source_stage or PlotErrorStage.RUN
        log.exception("Unexpected exception escaped run(): %s", exc)
        report.run_failure_error = err
        code = 1
    finally:
        # Reaches here when something other than Exception got out of run()
        # (SystemExit, KeyboardInterrupt) or when a handler above raised.
        # The report is still "failed" and no error was recorded, so give it
        # a generic one; the real traceback is in the log.
        if report.run_status != "succeeded" and report.run_failure_error is None:
            report.run_failure_error = PlotInternalError(
                PlotErrorCode.ABORTED, GENERIC_INTERNAL_MESSAGE, source_stage=PlotErrorStage.RUN)
        if report.run_failure_error is not None:
            _log_failure(report.run_failure_error)
        report.run_duration_seconds = time.time() - started
        # Push the report to workflow-service (no-op off the platform)
        push_report_to_workflow_service(report)
    return code


def _log_failure(err: PlotError) -> None:
    log.error(
        "Quick-plot failed [%s/%s at %s]: %s%s",
        err.error_category.value, err.error_code.value,
        err.source_stage.value if err.source_stage else "?", err.user_message,
        f" (cause: {err.__cause__!r})" if err.__cause__ else "",
    )


if __name__ == "__main__":
    sys.exit(main())
