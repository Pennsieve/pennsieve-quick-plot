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
entrypoint.sh       # Runtime detection (Lambda vs ECS)
Dockerfile          # Slim python 3.12 image with Lambda RIC
docker-compose.yml  # Local dev runner
dev.env             # Local env vars
Makefile            # make build / run / clean
data/
  input/            # Sample input files for local testing
  output/           # Output dir (gitignored except .gitkeep)
```

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
| `MAX_RETRIES` | container override | function config | Number of LLM retries on script failure. Default 1. |
| `SCRIPT_TIMEOUT_SECONDS` | container override | function config | Per-execution timeout for the generated script. Default 120. |
| `QUICK_PLOT_STACK_SITE_PACKAGES` | container override | function config | Override the EFS layer mount path (defaults to `/mnt/layers/quick-plot-stack/lib/python3.12/site-packages`). |
| `PENNSIEVE_API_HOST`, `PENNSIEVE_API_HOST2`, `ENVIRONMENT`, `REGION`, `DEPLOYMENT_MODE` | container override | function config (static) | Platform standards |

## Outputs

After a successful run, `OUTPUT_DIR/` contains:

- `figure.png` — the matplotlib figure (required)
- `script.py` — the LLM-generated Python script that produced it (kept for transparency / debugging)

The `viewer-asset` data-target node downstream picks up `figure.png` and attaches it as a viewer-asset on the target file. The script is also attached so users can see how the figure was made.

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

The processor uses `pennsieve-llm` (the Python client SDK) which returns a pre-configured `anthropic.Anthropic` client pointed at the governor's Lambda Function URL with SigV4 auth wired up.

Key fact: **the governor reads files from EFS server-side via `efs_document(path)` content blocks**, so the processor passes the target file path to the LLM directly — no base64 round-tripping. The LLM can inspect the file's structure (headers, shape) before writing the script.

```python
from pennsieve_llm import Governor, MODEL_SONNET_45

gov = Governor()  # auto-configures from $LLM_GOVERNOR_URL + AWS creds
resp = gov.client().messages.create(
    model=MODEL_SONNET_45,
    max_tokens=4096,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": [
        {"type": "text", "text": user_message},
        gov.efs_document(target_file_path),
    ]}],
)
```

## Failure modes

| Failure | Exit code | Behavior |
|---|---|---|
| `LLM_GOVERNOR_URL` not set | 1 | Clear error; no LLM call attempted |
| LLM call fails (budget exhausted, network, etc.) | 1 | Error from `pennsieve-llm`; logged and bubbled up |
| Script raises an exception | retries up to `MAX_RETRIES` | Traceback fed back to LLM via retry message |
| Script doesn't produce `figure.png` despite exiting 0 | retries up to `MAX_RETRIES` | Treated as failure; LLM asked to fix |
| Script exceeds `SCRIPT_TIMEOUT_SECONDS` | retries up to `MAX_RETRIES` | Timeout error fed back to LLM |
| All retries fail | 1 | Last-attempt script and error are in `OUTPUT_DIR/script.py` and stderr |

## TODO (scaffold-only items)

- [ ] Add tests/ — unit tests for prompt builders, executor result parsing; integration test against a stub LLM client
- [ ] Add Jenkinsfile when CI location is confirmed (probably mirrors other processor repos)
- [ ] Confirm the exact payload key name the platform uses for `prompt` (lowercase vs camelCase) and update handler.py accordingly
- [ ] Confirm whether `processorParams` arrive as top-level event keys or nested under a `processorParams` field (this affects how handler.py reads `prompt` and `target_file_name`)
- [ ] Validate the EFS layer mount path matches what the platform produces in production
- [ ] Smoke-test against a real Pennsieve LLM Governor Function URL
- [ ] Add file-extension allowlist with friendly errors (per plan §I.10 q4)
