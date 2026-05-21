"""
Agent-loop plotting backend.

Replaces the previous single-shot "ask the LLM for one Python script and run it"
approach (`processor/llm.py` + `processor/executor.py`) with a proper agent
loop. The model gets `bash` / `read_file` / `write_file` tools, inspects the
target file with real commands, writes plotting code that's grounded in the
actual schema, runs it, and iterates if it errors.

Why the change:
  Single-shot generation was fragile. The LLM guessed column names from prose
  in the system prompt and hallucinated schemas. Even with `efs_document`
  attached, the LLM couldn't execute code interactively to verify its
  assumptions — so a first attempt that referenced `df['Date']` against a CSV
  that didn't have a `Date` column would fail and burn a retry. Worse, on
  novel file types (FCS, h5ad, NIfTI, RDS) there's no schema available without
  actually reading the file.

  An agent loop lets the model run `head target.csv` (or
  `python -c "import fcsparser; print(fcsparser.parse(...).columns)"`) before
  writing the plot. That eliminates the entire class of "wrong column name"
  failures and generalizes to any supported file type without us building a
  per-type preview extractor.

Transport: uses `pennsieve_llm.Governor` → `gov.client()` to get a real
`anthropic.Anthropic` instance pointed at the LLM Governor. Tool-use blocks
are passed straight through; the Governor's Anthropic-API translation layer
already supports the protocol (see compute-node-aws-provisioner-v2's
cmd/llm-governor/anthropic.go).

Sandboxing: all tool effects are constrained to the per-run workdir and
read-only access to the staged input directory. `bash` strips AWS_* env vars
so generated commands can't call out to AWS APIs using the processor role's
credentials.
"""

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Cap the number of model ↔ tool round-trips per run. Plotting workflows
# typically converge in 3–6 iterations (inspect file → write script → run →
# fix / done). 20 leaves comfortable headroom for retries without letting a
# pathological loop eat the Lambda's 15-minute ceiling.
MAX_ITERATIONS = int(os.environ.get("AGENT_MAX_ITERATIONS", "20"))

# Per-`bash` call wall clock cap. Generous enough for non-trivial plots
# (FCS files with hundreds of thousands of events, h5ad embeddings, etc.)
# but short enough that a hung command doesn't drain the Lambda budget.
BASH_TIMEOUT_SECONDS = int(os.environ.get("AGENT_BASH_TIMEOUT_SECONDS", "60"))

# Truncate tool output we feed back to the model. Otherwise a stray
# `head -n 100000 huge.csv` would blow up context and cost. 16KB is enough
# for the model to see column lists, sample rows, and stack traces without
# dragging in the full data.
TOOL_OUTPUT_MAX_BYTES = int(os.environ.get("AGENT_TOOL_OUTPUT_MAX_BYTES", "16384"))


@dataclass
class AgentResult:
    """What the agent loop produced."""

    success: bool
    final_text: str = ""
    iterations: int = 0
    output_exists: bool = False
    output_size: int = 0
    transcript: list[dict] = field(default_factory=list)
    error: str = ""


def run_agent(
    user_prompt: str,
    target_file_path: str,
    output_path: str,
    workdir: str,
    layer_site_packages: str,
) -> AgentResult:
    """
    Run the agent loop until it produces `output_path` or hits the iteration cap.

    Args:
      user_prompt: the user's plain-English request.
      target_file_path: absolute path to the input file on EFS.
      output_path: where the agent must save the figure (`workdir/figure.png`
        in practice — passed explicitly so this module isn't path-aware).
      workdir: absolute path to a per-run scratch directory. The agent's bash
        tool runs with cwd=workdir; write_file accepts paths under here only.
      layer_site_packages: absolute path to the EFS layer's `site-packages`
        directory (matplotlib, pandas, fcsparser, etc.). Injected into the
        bash tool's PYTHONPATH so the agent can `import` the scientific stack.

    Returns AgentResult. `success=True` requires both `stop_reason=end_turn`
    AND `output_path` existing on disk.
    """
    # Import inside the function so import-time failures (missing
    # pennsieve_llm in a partial install) don't break the module.
    from pennsieve_llm import Governor, MODEL_SONNET_45  # type: ignore

    from processor.prompt import AGENT_SYSTEM_PROMPT, build_agent_user_message

    gov = Governor()
    client = gov.client()

    system_prompt = AGENT_SYSTEM_PROMPT.format(
        target_file_path=target_file_path,
        target_file_name=os.path.basename(target_file_path),
        output_path=output_path,
        workdir=workdir,
        bash_max=BASH_TIMEOUT_SECONDS,
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": build_agent_user_message(
                user_prompt=user_prompt,
                target_file_path=target_file_path,
                output_path=output_path,
                workdir=workdir,
            ),
        }
    ]

    tools = _tool_schemas()

    log.info("Agent loop start: cap=%d iterations, model=%s", MAX_ITERATIONS, MODEL_SONNET_45)
    started = time.time()
    final_text = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        log.info("--- Iteration %d / %d (elapsed %.1fs) ---", iteration, MAX_ITERATIONS, time.time() - started)

        try:
            resp = client.messages.create(
                model=MODEL_SONNET_45,
                max_tokens=4096,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )
        except Exception as e:  # noqa: BLE001 — surface the underlying error
            log.error("LLM call failed: %s", e)
            return AgentResult(
                success=False,
                iterations=iteration - 1,
                error=f"LLM call failed: {e}",
                transcript=messages,
            )

        # Persist the assistant turn (text + tool_use blocks) so the next
        # request can reference the same tool_use_ids in tool_result.
        messages.append({"role": "assistant", "content": _normalize_assistant_content(resp.content)})

        # If the model emitted any text, log it for transparency.
        for block in resp.content:
            if block.type == "text" and block.text.strip():
                log.info("Agent: %s", _truncate_for_log(block.text.strip(), 500))
                final_text = block.text.strip()

        if resp.stop_reason == "end_turn":
            # Model thinks it's done. Verify the figure exists; if not, the
            # caller surfaces the discrepancy. We don't loop further — the
            # model has already declared completion.
            log.info("Agent reported end_turn after %d iterations", iteration)
            break

        if resp.stop_reason != "tool_use":
            log.warning("Unexpected stop_reason=%s; terminating loop", resp.stop_reason)
            break

        # Run every tool_use block in the assistant turn and collect results.
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            log.info("Tool call: %s(%s)", block.name, _truncate_for_log(json.dumps(block.input), 200))
            result_text, is_error = _run_tool(
                name=block.name,
                inputs=block.input,
                workdir=workdir,
                target_file_path=target_file_path,
                layer_site_packages=layer_site_packages,
            )
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
                "is_error": is_error,
            })

        if not tool_results:
            # Stop reason was tool_use but no tool_use blocks present — protocol
            # bug or refusal-shaped response. Bail.
            log.warning("stop_reason=tool_use but no tool_use blocks in response; terminating loop")
            break

        messages.append({"role": "user", "content": tool_results})

    else:
        # Loop fell off the end without break — hit the iteration cap.
        log.warning("Hit MAX_ITERATIONS=%d without end_turn", MAX_ITERATIONS)

    output_exists = os.path.isfile(output_path)
    output_size = os.path.getsize(output_path) if output_exists else 0
    iterations_used = min(iteration, MAX_ITERATIONS)  # noqa: F821 — `iteration` defined in the for-loop above

    return AgentResult(
        success=output_exists,
        final_text=final_text,
        iterations=iterations_used,
        output_exists=output_exists,
        output_size=output_size,
        transcript=messages,
        error="" if output_exists else "Agent stopped before writing the figure file.",
    )


# --------------------------------------------------------------------------- #
# Tool definitions + dispatch
# --------------------------------------------------------------------------- #


def _tool_schemas() -> list[dict[str, Any]]:
    """JSON Schemas Anthropic Messages API expects for each available tool."""
    return [
        {
            "name": "bash",
            "description": (
                "Run a shell command. Use this to inspect files (e.g. `head -n 5 file.csv`, "
                "`wc -l file.csv`, `python -c \"...\"`), run scripts, and create figures. "
                "Working directory is set to WORKDIR. Commands have a wall-clock timeout "
                "of {bash_timeout}s. Output is truncated to {max_bytes} bytes; if you need "
                "more, pipe through grep/head/etc. AWS credentials are not available; "
                "network calls beyond the local filesystem are not supported."
            ).format(bash_timeout=BASH_TIMEOUT_SECONDS, max_bytes=TOOL_OUTPUT_MAX_BYTES),
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute. Multi-line scripts are fine; the shell is /bin/bash.",
                    },
                },
                "required": ["command"],
            },
        },
        {
            "name": "read_file",
            "description": (
                "Read the first N bytes of a file. Use to inspect text-formatted files "
                "(CSV, TSV, JSON, etc.). Only paths under WORKDIR or the staged input "
                "directory containing TARGET_FILE are allowed."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path."},
                    "max_bytes": {
                        "type": "integer",
                        "description": f"Bytes to read. Capped at {TOOL_OUTPUT_MAX_BYTES}. Default 4096.",
                        "default": 4096,
                    },
                },
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": (
                "Write text to a file. Only paths under WORKDIR are allowed (use this for "
                "plotting scripts you want to persist or for the final figure if it's "
                "produced by a separate command). The figure must end up at OUTPUT_PATH."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path under WORKDIR."},
                    "content": {"type": "string", "description": "File contents (UTF-8 text)."},
                },
                "required": ["path", "content"],
            },
        },
    ]


def _run_tool(
    name: str,
    inputs: dict[str, Any],
    workdir: str,
    target_file_path: str,
    layer_site_packages: str,
) -> tuple[str, bool]:
    """Dispatch one tool_use block. Returns (result_text, is_error)."""
    try:
        if name == "bash":
            return _tool_bash(
                command=inputs.get("command", ""),
                workdir=workdir,
                layer_site_packages=layer_site_packages,
            )
        if name == "read_file":
            return _tool_read_file(
                path=inputs.get("path", ""),
                max_bytes=int(inputs.get("max_bytes", 4096)),
                workdir=workdir,
                target_file_path=target_file_path,
            )
        if name == "write_file":
            return _tool_write_file(
                path=inputs.get("path", ""),
                content=inputs.get("content", ""),
                workdir=workdir,
            )
        return f"Unknown tool: {name}", True
    except Exception as e:  # noqa: BLE001 — defensive: never crash the agent loop
        log.exception("Tool %s crashed", name)
        return f"Tool {name} raised: {type(e).__name__}: {e}", True


def _tool_bash(command: str, workdir: str, layer_site_packages: str) -> tuple[str, bool]:
    """Run a shell command in the sandbox."""
    if not command.strip():
        return "Empty command.", True

    # Build a stripped env: keep PATH + Python config, drop AWS credentials so
    # generated commands can't reach AWS APIs with the processor role's
    # identity. Inject PYTHONPATH so `import matplotlib` etc. resolve.
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": workdir,  # Some libraries (matplotlib) want a writable HOME.
        "MPLCONFIGDIR": os.path.join(workdir, ".mplconfig"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    # PYTHONPATH = EFS layer + caller's existing PYTHONPATH (if any).
    pythonpath_parts = []
    if os.path.isdir(layer_site_packages):
        pythonpath_parts.append(layer_site_packages)
    if os.environ.get("PYTHONPATH"):
        pythonpath_parts.append(os.environ["PYTHONPATH"])
    if pythonpath_parts:
        env["PYTHONPATH"] = ":".join(pythonpath_parts)

    os.makedirs(env["MPLCONFIGDIR"], exist_ok=True)

    try:
        proc = subprocess.run(
            ["/bin/bash", "-c", command],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"Command exceeded {BASH_TIMEOUT_SECONDS}-second timeout.", True

    # Compose a single result string. Include exit code on non-zero.
    parts = []
    if proc.stdout:
        parts.append("STDOUT:\n" + _truncate(proc.stdout))
    if proc.stderr:
        parts.append("STDERR:\n" + _truncate(proc.stderr))
    if proc.returncode != 0:
        parts.append(f"EXIT CODE: {proc.returncode}")
    if not parts:
        parts.append("(no output)")
    body = "\n\n".join(parts)
    return body, proc.returncode != 0


def _tool_read_file(path: str, max_bytes: int, workdir: str, target_file_path: str) -> tuple[str, bool]:
    """Read the head of a file under workdir or alongside the target file."""
    real = os.path.realpath(path)
    if not (real.startswith(os.path.realpath(workdir) + os.sep)
            or real.startswith(os.path.realpath(os.path.dirname(target_file_path)) + os.sep)
            or real == os.path.realpath(target_file_path)):
        return (
            f"Access denied: {path} is outside the allowed roots (WORKDIR + the staged "
            f"input directory).",
            True,
        )
    if not os.path.isfile(real):
        return f"Not a file or does not exist: {path}", True

    cap = min(max(max_bytes, 1), TOOL_OUTPUT_MAX_BYTES)
    with open(real, "rb") as f:
        raw = f.read(cap)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return (
            f"Binary file (first {cap} bytes are not valid UTF-8). Use bash with an "
            f"appropriate reader (e.g. fcsparser, h5py, nibabel).",
            True,
        )
    suffix = "" if os.path.getsize(real) <= cap else f"\n\n(truncated; full size {os.path.getsize(real)} bytes)"
    return text + suffix, False


def _tool_write_file(path: str, content: str, workdir: str) -> tuple[str, bool]:
    """Write a text file under workdir."""
    real = os.path.realpath(path)
    workdir_real = os.path.realpath(workdir)
    if not real.startswith(workdir_real + os.sep) and real != workdir_real:
        return f"Access denied: {path} is outside WORKDIR ({workdir}).", True

    os.makedirs(os.path.dirname(real), exist_ok=True)
    with open(real, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} bytes to {path}.", False


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _normalize_assistant_content(blocks) -> list[dict[str, Any]]:
    """
    Convert Anthropic SDK content block objects to the dict shape the next
    `messages.create` expects. The SDK accepts both shapes on input but
    returning plain dicts keeps the transcript JSON-serializable for logs.
    """
    out = []
    for b in blocks:
        if b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return out


def _truncate(s: str) -> str:
    """Truncate a tool-output string to TOOL_OUTPUT_MAX_BYTES."""
    if len(s) <= TOOL_OUTPUT_MAX_BYTES:
        return s
    return s[:TOOL_OUTPUT_MAX_BYTES] + f"\n\n(truncated; full output {len(s)} bytes)"


def _truncate_for_log(s: str, n: int) -> str:
    """One-line log-friendly truncation."""
    s = s.replace("\n", " ⏎ ")
    if len(s) <= n:
        return s
    return s[:n] + "...(truncated)"
