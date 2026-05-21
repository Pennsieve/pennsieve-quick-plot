"""
Thin wrapper around the Pennsieve LLM Governor.

Uses `pennsieve-llm`'s `Governor` to get a pre-configured `anthropic.Anthropic`
client. The SDK auto-selects the right transport from env:

  - `LLM_GOVERNOR_FUNCTION_NAME` (preferred — direct Lambda invoke against the
    compute-node-local governor; no Function URL needed; private AWS APIs only)
  - `LLM_GOVERNOR_URL` (legacy HTTPS + SigV4 path)
  - `PENNSIEVE_LLM_MOCK=1` (opt-in mock for local dev)

If none are set the SDK raises GovernorError at construction time — that's the
right behavior. There's no offline LLM fallback in this processor by design.
"""

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)


def generate_script(
    user_prompt: str,
    target_file_path: str,
    target_file_name: str,
    output_path: str,
    previous_error: Optional[str] = None,
    previous_script: Optional[str] = None,
) -> str:
    """
    Call the LLM via the Pennsieve Governor and return a Python script.

    Args:
      user_prompt: The user's plain-English request ("plot CD3 vs CD19").
      target_file_path: Absolute path to the input file on EFS.
      target_file_name: Basename of the input file (for nicer labels in the figure).
      output_path: Where the script must save the figure (typically OUTPUT_DIR/figure.png).
      previous_error: If retrying after a failure, the error text from the prior run.
      previous_script: The script that failed (for the retry context).

    Returns:
      A Python script as a string. Caller is responsible for executing it.

    Raises:
      RuntimeError: if the governor SDK can't configure a backend, or the call fails.
    """
    # Import inside the function so the module loads even when pennsieve_llm isn't installed
    # (the requirements.txt installs it; this guard helps with linting / partial installs).
    from pennsieve_llm import Governor, MODEL_SONNET_45  # type: ignore

    from processor.prompt import SYSTEM_PROMPT, build_user_message, build_retry_message

    gov = Governor()  # auto-selects LLM_GOVERNOR_FUNCTION_NAME or LLM_GOVERNOR_URL

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_user_message(
                    user_prompt, target_file_path, target_file_name, output_path
                )},
                gov.efs_document(target_file_path),
            ],
        }
    ]

    if previous_error and previous_script:
        # Add the prior failed attempt and the retry instruction
        messages.append({"role": "assistant", "content": previous_script})
        messages.append({"role": "user", "content": build_retry_message(previous_error)})

    log.info(
        "Calling LLM Governor (run_id=%s, function=%s, url=%s)",
        os.environ.get("EXECUTION_RUN_ID", "?"),
        os.environ.get("LLM_GOVERNOR_FUNCTION_NAME", ""),
        os.environ.get("LLM_GOVERNOR_URL", ""),
    )
    resp = gov.client().messages.create(
        model=MODEL_SONNET_45,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    # Extract the text. The model is instructed not to wrap in code fences, but
    # strip them defensively in case it does anyway.
    script = "".join(block.text for block in resp.content if block.type == "text")
    return _strip_markdown_fences(script)


def _strip_markdown_fences(text: str) -> str:
    """If the model wrapped the script in ```python ... ``` fences, strip them."""
    text = text.strip()
    if text.startswith("```"):
        # Drop the opening fence (with optional language tag)
        first_newline = text.find("\n")
        if first_newline >= 0:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()
