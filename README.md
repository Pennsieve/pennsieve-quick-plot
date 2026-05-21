# pennsieve-quick-plot

Lambda + Fargate dual-mode processor for the `quick-plot` workflow. Given a target data file and a plain-English prompt, asks the Pennsieve LLM Governor for a Python plotting script, runs it against the scientific-Python stack on the `quick-plot-stack` EFS layer, and writes a PNG figure to `OUTPUT_DIR`.

This is the v1 workflow described in [`compute-node-chat/docs/developer/plan-notebook-workflows.md`](../compute-node-chat/docs/developer/plan-notebook-workflows.md) §Part I.

## What it does

```
prompt + file ──▶ LLM generates Python ──▶ subprocess runs against EFS layer ──▶ figure.png
                  (single Claude call,                                            (picked up by
                   1 retry on error,                                              viewer-asset
                   ~$0.10-0.50)                                                   data-target)
```

End-to-end: ~10 seconds. The user sees the figure inline in chat via the `plot_file` MCP tool.

## Dual-mode architecture

Same image runs as Lambda or ECS Fargate. The runtime is detected by `entrypoint.sh` checking `AWS_LAMBDA_RUNTIME_API`:

- **Lambda**: `awslambdaric` invokes `processor.handler.handler`, which bridges the event payload to env vars, then calls `main.run()`.
- **ECS / local**: `python -m processor.main` reads env vars directly.

The processor logic in `main.py` is identical in both modes.

## Project layout

```
processor/
  main.py           # ECS/local entry — orchestrates the flow
  handler.py        # Lambda handler — event → env vars → main.run()
  prompt.py         # System prompt + user-message builder for the LLM
  llm.py            # Pennsieve LLM Governor wrapper (calls anthropic SDK via SigV4 to Function URL)
  executor.py       # subprocess that runs the LLM-generated script with EFS layer on PYTHONPATH
  requirements.txt  # awslambdaric, pennsieve-llm, boto3 (heavy deps live on the EFS layer)
workflows/
  quick-plot.json                    # main workflow definition (Lambda; the v1 product)
  populate-quick-plot-stack.json     # one-time layer-population workflow (Fargate; admin-triggered)
  README.md                          # deployment notes for the JSONs
entrypoint.sh       # Runtime detection (Lambda vs ECS)
Dockerfile          # Slim python 3.12 image with Lambda RIC
docker-compose.yml  # Local dev runner
dev.env             # Local env vars
Makefile            # make build / run / clean
data/
  input/            # Sample input files for local testing
  output/           # Output dir (gitignored except .gitkeep)
```

## Workflow definitions

The workflow definitions consumed by `workflow-service` live in `workflows/`:

- **`quick-plot.json`** — the main workflow this repo's processor implements. Lambda + viewer-asset data-target.
- **`populate-quick-plot-stack.json`** — one-time layer-population workflow. Uses `processor-build-python-layer` (a separate repo) + `persistent-layer` data-target to install the scientific Python stack into the `quick-plot-stack` EFS layer.

Workflow definition registration is API-only in workflow-service today — the JSONs are POSTed to `/definitions` at deploy time. See [`workflows/README.md`](workflows/README.md) for details and open questions.

## Local development

```sh
make build    # build the Docker image
make run      # run via docker-compose (ECS mode), reads dev.env
make clean    # remove output files
```

For local LLM calls, set `LLM_GOVERNOR_URL` in `dev.env` to a real Function URL and configure AWS creds with `bedrock:InvokeModel` access via the governor. Otherwise the processor will exit with a clear error — there is no offline LLM fallback by design.

## Environment variables

| Variable | Source (ECS) | Source (Lambda) | Purpose |
|----------|--------------|-----------------|---------|
| `INPUT_DIR` | container override | event payload `inputDir` | Path to staged input file(s) on EFS |
| `OUTPUT_DIR` | container override | event payload `outputDir` | Where to write `figure.png` and `script.py` |
| `PROMPT` | container override | event payload `prompt` | User's plain-English plotting request |
| `TARGET_FILE_NAME` | container override | event payload `target_file_name` | Optional — name of specific file in INPUT_DIR to use |
| `EXECUTION_RUN_ID` | container override | event payload `executionRunId` | Cost attribution; passed to governor as `x-execution-run-id` |
| `LLM_GOVERNOR_URL` | container override | event payload `llmGovernorFunction` | Lambda Function URL of the LLM Governor |
| `INTEGRATION_ID` | container override | event payload `integrationId` | Workflow run identifier |
| `SESSION_TOKEN` | container override | event payload `sessionToken` | API session token |
| `AGENT_MAX_ITERATIONS` | container override | function config | Cap on model ↔ tool round-trips per run. Default 20. |
| `AGENT_BASH_TIMEOUT_SECONDS` | container override | function config | Per-`bash` tool call timeout. Default 60. |
| `AGENT_TOOL_OUTPUT_MAX_BYTES` | container override | function config | Truncate tool output before feeding back to the model. Default 16384. |
| `SCRIPT_TIMEOUT_SECONDS` | container override | function config | Per-execution timeout for STUB_MODE scripts only. Default 120. |
| `QUICK_PLOT_STACK_SITE_PACKAGES` | container override | function config | Override the EFS layer mount path (defaults to `/mnt/layers/quick-plot-stack/lib/python3.12/site-packages`). |
| `STUB_MODE` | container override | function config | When `1`, bypass the agent loop and run a built-in stub script (used to smoke-test the EFS-layer + viewer-asset chain). |
| `PENNSIEVE_API_HOST`, `PENNSIEVE_API_HOST2`, `ENVIRONMENT`, `REGION`, `DEPLOYMENT_MODE` | container override | function config (static) | Platform standards |

## Outputs

After a successful run, `OUTPUT_DIR/` contains:

- `figure.png` — the matplotlib figure (required)
- Any intermediate scripts / files the agent chose to persist via `write_file` (kept for transparency / debugging; not required)

The `viewer-asset` data-target node downstream picks up `figure.png` and attaches it as a viewer-asset on the target file.

## EFS layer dependency

The processor *will not run* without the `quick-plot-stack` EFS layer mounted. The layer provides the scientific-Python stack:

- Plotting: `matplotlib`, `seaborn`
- Tabular: `pandas`, `numpy`, `scipy`
- Single-cell: `anndata`, `scanpy`
- Flow cytometry: `flowkit`, `fcsparser`
- Imaging: `nibabel`, `tifffile`, `pillow`
- R interop: `pyreadr`
- General: `pyarrow`, `h5py`

The layer is provisioned and populated separately (see workflow-service docs on EFS layers). At run time, the workflow request includes `"layers": ["quick-plot-stack"]` which causes the provisioner to mount the layer at `/mnt/layers/quick-plot-stack/`. The executor injects the layer's site-packages into the script subprocess's `PYTHONPATH`.

## LLM Governor integration

The processor uses `pennsieve-llm` (the Python client SDK) which returns a real `anthropic.Anthropic` client pointed at the governor's Lambda Function URL with SigV4 auth wired up. The agent loop (see `processor/agent.py`) calls `client.messages.create` with `tools=[...]` per the Anthropic Messages API — the governor passes tool-use / tool-result blocks through unchanged (see `compute-node-aws-provisioner-v2/cmd/llm-governor/anthropic.go`).

```python
from pennsieve_llm import Governor, MODEL_SONNET_45

gov = Governor()
resp = gov.client().messages.create(
    model=MODEL_SONNET_45,
    max_tokens=4096,
    system=AGENT_SYSTEM_PROMPT,
    tools=[bash_tool, read_file_tool, write_file_tool],
    messages=[...],  # accumulated transcript (user prompt → tool_use / tool_result loop)
)
```

The agent inspects the target file with shell + Python first, then writes plotting code grounded in actual data. Single-shot script generation (v0.x) was retired because it hallucinated columns / schemas on novel file types — see commit message and `processor/agent.py` docstring for the full rationale.

## Failure modes

| Failure | Exit code | Behavior |
|---|---|---|
| `LLM_GOVERNOR_URL` not set | 1 | Clear error from `pennsieve-llm`; no LLM call attempted |
| LLM call fails mid-loop (budget exhausted, network, etc.) | 1 | Error logged with iteration count; loop terminates |
| Agent calls a tool with bad input (e.g. `read_file` outside WORKDIR) | continues | Tool returns an `is_error: true` result; agent can recover |
| Agent script crashes inside `bash` | continues | Non-zero exit + stderr fed back to the model |
| Agent emits `end_turn` without `figure.png` | 1 | Treated as failure; no further retries — model declared completion |
| Agent hits `AGENT_MAX_ITERATIONS` | 1 | Loop terminates without `end_turn`; failure logged |

## TODO (scaffold-only items)

- [ ] Add tests/ — unit tests for prompt builders, executor result parsing; integration test against a stub LLM client
- [ ] Add Jenkinsfile when CI location is confirmed (probably mirrors other processor repos)
- [ ] Confirm the exact payload key name the platform uses for `prompt` (lowercase vs camelCase) and update handler.py accordingly
- [ ] Confirm whether `processorParams` arrive as top-level event keys or nested under a `processorParams` field (this affects how handler.py reads `prompt` and `target_file_name`)
- [ ] Validate the EFS layer mount path matches what the platform produces in production
- [ ] Smoke-test against a real Pennsieve LLM Governor Function URL
- [ ] Add file-extension allowlist with friendly errors (per plan §I.10 q4)
