"""
processor.report — the run report: what one quick-plot run produced, or
why it did not, pushed to workflow-service as run outputs.

Built on *every* exit path, success included, so the contract is
uniform: a reader never has to infer "no report means it worked".

The run record (`GET /runs/{id}`) is what pennsieve-mcp polls and what
the completion webhook carries to compute-node-chat, so both consumers
see the report without new plumbing. Outside the platform (local runs,
tests) there is nothing to push to; the log line written by
main._log_failure() is the report's content in one line.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from processor.errors import PlotError

log = logging.getLogger("quick-plot")

# Every run-output key this processor writes starts with this prefix so it
# can never collide with keys other steps report (data-target-assets writes
# `assetId`). Consumers look for `<prefix>.run_failure_error.user_message` first.
OUTPUTS_PREFIX = "quickplot"
OUTPUTS_TIMEOUT_S = 5
OUTPUTS_VALUE_MAX = 2000      # workflow-service stores a map[string]string; keep values bounded


@dataclass
class PlotRunReport:
    """What one run produced, or why it did not. Fields are in the order
    they are filled during a run:

      run_id                  main(), at creation
      requested_template      run(), from config, before anything is tried
      fallback_trigger_error  run(), when the canned template failed for a
                              non-user reason and the agent was given the
                              request instead (checklist item 7)
      figure_generated_by     run(), on success: which route drew the figure
      figure_bytes            run(), on success
      run_status              main(), after run() returned or raised
      run_failure_error       main(), when run() raised
      run_duration_seconds    main(), in the finally, just before the push

    The saved keys mirror these names one-to-one under the "quickplot."
    prefix (see to_outputs), so a reader of the run record and a reader of
    this file use the same words.
    """

    run_id: str = ""                   # The run's identifier, copied from the EXECUTION_RUN_ID environment variable
    requested_template: str = ""       # The template name as the user asked for it, even when the agent drew it
    fallback_trigger_error: PlotError | None = None   # Why the canned template failed for a non-user reason, if it did; the agent was given the request instead
    figure_generated_by: str = ""      # Which drawing route actually produced figure.png: "template" | "agent" | "stub" | "" (never got that far)
    figure_bytes: int = 0              # How many bytes were written to figure.png (0 if no file was produced)
    run_status: str = "failed"         # "succeeded" | "failed" — same words as workflow-service's run status
    run_failure_error: PlotError | None = None   # Filled when run_status == "failed". The full PlotError (category, code, stage, user_message, next step, facts), pushed as run outputs so the consumer reads it without parsing a log line
    run_duration_seconds: float = 0.0  # Seconds from main's first line to just before the push, so it includes error handling time. Useful for spotting slow runs


def to_outputs(report: PlotRunReport) -> dict[str, str]:
    """Flatten the report into the string→string map workflow-service stores.

    Keys are stable API: pennsieve-mcp and compute-node-chat read them by
    name. Add keys freely; never rename one. Each key is the report field
    name under the prefix; an error field expands into its pockets:

      quickplot.run_status, .requested_template, .figure_generated_by,
      .figure_bytes, .run_duration_seconds
      quickplot.run_failure_error.{error_category, error_code, source_stage,
                                   user_message, user_next_step, error_facts}
      quickplot.fallback_trigger_error.{same pockets}
    """
    out: dict[str, str] = {
        f"{OUTPUTS_PREFIX}.run_status": report.run_status,
        f"{OUTPUTS_PREFIX}.requested_template": report.requested_template,
        f"{OUTPUTS_PREFIX}.figure_generated_by": report.figure_generated_by,
        f"{OUTPUTS_PREFIX}.figure_bytes": str(report.figure_bytes),
        f"{OUTPUTS_PREFIX}.run_duration_seconds": f"{report.run_duration_seconds:.2f}",
    }
    if report.run_failure_error is not None:
        out.update(_flatten_error(f"{OUTPUTS_PREFIX}.run_failure_error", report.run_failure_error))
    if report.fallback_trigger_error is not None:
        out.update(_flatten_error(f"{OUTPUTS_PREFIX}.fallback_trigger_error",
                                  report.fallback_trigger_error))
    return {k: _clip(v) for k, v in out.items()}


def _flatten_error(prefix: str, err: PlotError) -> dict[str, str]:
    d = {
        f"{prefix}.error_category": err.error_category.value,
        f"{prefix}.error_code": err.error_code.value,
        f"{prefix}.source_stage": err.source_stage.value if err.source_stage else "",
        f"{prefix}.user_message": public_message(err.user_message),
        f"{prefix}.user_next_step": err.user_next_step,
    }
    if err.error_facts:
        d[f"{prefix}.error_facts"] = json.dumps(err.error_facts, sort_keys=True, default=str)
    return d


def _clip(value: str) -> str:
    value = str(value)
    return value if len(value) <= OUTPUTS_VALUE_MAX else value[: OUTPUTS_VALUE_MAX - 1] + "…"


# An absolute path with at least one directory component. The reader's
# messages quote the staged file's full EFS path ("/mnt/efs/.../rec.edf");
# the user only knows the file by name, and the rest is our plumbing.
_ABS_PATH = re.compile(r"(?<![\w/])/(?:[\w.\-@+]+/)+([\w.\-@+]+)")


def public_message(text: str) -> str:
    """Strip filesystem paths down to their basename for user-facing text.

    In the chat app the message is pushed verbatim with no model in
    between, so this is the last filter before the user. The full text
    stays in the log (main._log_failure logs the original exception).
    """
    return _ABS_PATH.sub(r"\1", text)


def push_report_to_workflow_service(report: PlotRunReport, env: dict[str, str] | None = None) -> bool:
    """PUT the flattened report to workflow-service's run outputs. Best effort.

    Endpoint: {host}/compute/workflows/runs/{EXECUTION_RUN_ID}/outputs

    Auth, in order of preference:
      X-Callback-Token       if the runtime forwarded CALLBACK_TOKEN (not today)
      Bearer SESSION_TOKEN   the user's token; accepted because the run was
                             created by the same user (authenticateRunAccess
                             creator check in workflow-service)

    Host: QUICKPLOT_API_HOST (explicit override, e.g. passed as a
    processorParam) > PENNSIEVE_API_HOST2 > PENNSIEVE_API_HOST. The route
    lives on the api2 services gateway.

    Returns True on 2xx. Never raises: reporting can not fail the run.
    """
    env = os.environ if env is None else env
    run_id = env.get("EXECUTION_RUN_ID", "")
    if not run_id:
        # Local run / unit test: there is no run record to attach to.
        log.info("No EXECUTION_RUN_ID; not on the platform, report stays in the log.")
        return False

    host = (env.get("QUICKPLOT_API_HOST") or env.get("PENNSIEVE_API_HOST2")
            or env.get("PENNSIEVE_API_HOST") or "").rstrip("/")
    callback_token = env.get("CALLBACK_TOKEN", "")
    session_token = env.get("SESSION_TOKEN", "")
    if not host or not (callback_token or session_token):
        # On the platform but missing a prerequisite: that is a real
        # misconfiguration, worth a warning.
        log.warning("Cannot push report for run %s: host=%s token=%s",
                    run_id, host or "(none)",
                    "callback" if callback_token else ("session" if session_token else "none"))
        return False

    headers = {"Content-Type": "application/json"}
    if callback_token:
        headers["X-Callback-Token"] = callback_token
        auth = "callback_token"
    else:
        headers["Authorization"] = f"Bearer {session_token}"
        auth = "session_token"

    body = to_outputs(report)
    # Tell the reader which auth/host worked — answers "what did the Lambda
    # have in its pocket" without dumping the environment.
    body[f"{OUTPUTS_PREFIX}.reporter.auth"] = auth
    body[f"{OUTPUTS_PREFIX}.reporter.host"] = host

    url = f"{host}/compute/workflows/runs/{run_id}/outputs"
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 method="PUT", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=OUTPUTS_TIMEOUT_S) as resp:
            log.info("Pushed %d report keys to %s (HTTP %s)", len(body), url, resp.status)
            return True
    except urllib.error.HTTPError as exc:
        log.warning("PUT %s failed: HTTP %s %s", url, exc.code, (exc.read() or b"")[:300])
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log.warning("PUT %s failed: %s", url, exc)
    return False
