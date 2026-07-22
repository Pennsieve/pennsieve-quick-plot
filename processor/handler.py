"""
Lambda handler. Bridges the per-invocation event payload to environment variables,
then runs the same processing logic as ECS mode.

The platform passes these payload keys (subset relevant to quick-plot):
  inputDir              → INPUT_DIR
  outputDir             → OUTPUT_DIR
  integrationId         → INTEGRATION_ID
  executionRunId        → EXECUTION_RUN_ID
  sessionToken          → SESSION_TOKEN
  refreshToken          → REFRESH_TOKEN
  llmGovernorFunction   → LLM_GOVERNOR_URL

processorParams (per-node config, surfaced by the provisioner):
  prompt                → PROMPT
  template              → TEMPLATE (optional)
  template_args         → TEMPLATE_ARGS (optional; JSON object as a string)
  target_file_name      → TARGET_FILE_NAME (optional)

Static env vars (PENNSIEVE_API_HOST, ENVIRONMENT, REGION, DEPLOYMENT_MODE)
are set on the Lambda function configuration, not per-invocation.
"""

import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
log = logging.getLogger("quick-plot")


# Payload keys we recognize and bridge to specific env-var names.
# Other string-valued keys are passed through verbatim (caps and all).
_PAYLOAD_TO_ENV = {
    "inputDir": "INPUT_DIR",
    "outputDir": "OUTPUT_DIR",
    "integrationId": "INTEGRATION_ID",
    "executionRunId": "EXECUTION_RUN_ID",
    "sessionToken": "SESSION_TOKEN",
    "refreshToken": "REFRESH_TOKEN",
    "llmGovernorFunction": "LLM_GOVERNOR_URL",
    "layersDir": "LAYERS_DIR",
    "template": "TEMPLATE",
    "template_args": "TEMPLATE_ARGS",
    "prompt": "PROMPT",
    "target_file_name": "TARGET_FILE_NAME",
    "stub_mode": "STUB_MODE",
}


def handler(event, context):
    log.info("Lambda handler invoked. Event keys: %s", list(event.keys()))

    # Bridge known keys
    for payload_key, env_key in _PAYLOAD_TO_ENV.items():
        if payload_key in event and event[payload_key] is not None:
            os.environ[env_key] = str(event[payload_key])

    # Bridge any unrecognized string-valued keys (e.g. injected secrets)
    for key, value in event.items():
        if key in _PAYLOAD_TO_ENV:
            continue
        if isinstance(value, str):
            os.environ[key] = value

    # Run the same logic as ECS mode
    from processor.main import run
    run()

    return {
        "status": "success",
        "executionRunId": event.get("executionRunId", ""),
        "outputDir": event.get("outputDir", ""),
    }
