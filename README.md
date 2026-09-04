# pennsieve-quick-plot

Lambda + Fargate dual-mode processor for the `quick-plot` workflow. Given a target data file and a plain-English prompt, asks the Pennsieve LLM Governor for a Python plotting script, runs it against the scientific-Python stack on the `quick-plot-stack` EFS layer, and writes a PNG figure to `OUTPUT_DIR`.

This is the v1 workflow described in [`compute-node-chat/docs/developer/plan-notebook-workflows.md`](../compute-node-chat/docs/developer/plan-notebook-workflows.md) §Part I.

## What it does

Two render paths share this single processor; the canned path is tried first when a `TEMPLATE` env var is set, and falls back to the LLM agent loop on any failure:

```
                ┌──── TEMPLATE set + known ────▶  canned template  ──┐
file + args ────┤                                 (~5–10s, ~$0.001) │
                └──── TEMPLATE unset / unknown ──▶  LLM agent loop ──┴──▶ figure.png
                                                  (~30s, ~$0.15)         │
                                                                         ▼
                                                                  (picked up by
                                                                   viewer-asset
                                                                   data-target)
```

End-to-end: **~10 seconds** when a template covers the request, **~30 seconds** when the agent runs. The user sees the figure inline in chat via the `plot_file` MCP tool.

History: this processor started as agent-only (single-shot Claude call → ran the script). [v0.6.0](https://github.com/Pennsieve/pennsieve-quick-plot/releases/tag/v0.6.0) added the canned-template path. Templates live in [`processor/templates/`](processor/templates/) — see [Adding a canned template](#adding-a-canned-template).

## Dual-mode architecture

Same image runs as Lambda or ECS Fargate. The runtime is detected by `entrypoint.sh` checking `AWS_LAMBDA_RUNTIME_API`:

- **Lambda**: `awslambdaric` invokes `processor.handler.handler`, which bridges the event payload to env vars, then calls `main.run()`.
- **ECS / local**: `python -m processor.main` reads env vars directly.

The processor logic in `main.py` is identical in both modes.

## Project layout

```
processor/
  main.py           # ECS/local entry — try canned, else fall through to agent
  handler.py        # Lambda handler — event → env vars → main.run()
  prompt.py         # Agent system prompt + user-message builder
  llm.py            # Pennsieve LLM Governor wrapper (anthropic SDK + SigV4)
  executor.py       # subprocess that runs the agent's generated script with EFS layer on PYTHONPATH
  requirements.txt  # awslambdaric, pennsieve-llm, boto3 (heavy deps live on the EFS layer)
  templates/        # canned per-format render functions (no LLM)
    __init__.py     # registry — { NAME → module }
    contract.py     # declarative half of the template contract (TemplateArg / ARGS_SPEC / …)
    generate_template_schema.py   # emits schema/templates.json (see Generated schemas)
    csv_column_distributions.py
    edf_processed_timeseries.py
    fcs_channel_histograms.py
    fcs_compensation_heatmap.py
    fcs_fsc_ssc_scatter.py
    fcs_time_diagnostics.py
    tsv_sessions_lineplot.py
  tools/            # DSP tool registries templates accept via their `pipeline` arg
    generate_tools_schema.py      # emits schema/<family>_tools.json per registry
    ts_dsp/         # time-series family (17 tools): filters, smoothers, spectral
                    # transforms, windowed features, EDF io — see "DSP pipeline tools"
schema/             # generated schemas (`make schemas`) — pennsieve-mcp vendors copies
  templates.json
  ts_tools.json
workflows/
  plot-templates.json                # main workflow (single-stage; Lambda; the v0.6+ product)
  quick-plot.json                    # legacy agent-only workflow (kept for backwards compat)
  populate-quick-plot-stack.json     # one-time layer-population workflow (Fargate; admin-triggered)
  README.md                          # deployment notes for the JSONs
entrypoint.sh       # Runtime detection (Lambda vs ECS)
Dockerfile          # Slim python 3.12 image with Lambda RIC + MPLCONFIGDIR=/tmp/matplotlib
docker-compose.yml  # Local dev runner
dev.env             # Local env vars
Makefile            # make build / run / clean / schemas
data/
  input/            # Sample input files for local testing
  output/           # Output dir (gitignored except .gitkeep)
```

## Adding a canned template

A "canned template" is a per-format Python function that renders a summary plot deterministically — no LLM in the loop. The MCP `plot_file` tool dispatches to one when its caller sets `template: <name>`; failures fall through to the agent loop automatically.

Adding one is three small steps. Total work: ~50 lines of Python + a schema regeneration — no MCP code changes.

### 1. Write the template

Drop a new module in `processor/templates/`. The contract:

```python
# processor/templates/your_template_name.py
"""
your_template_name — one-line summary, when to use, output shape.
"""
from processor.templates.contract import TemplateArg

NAME = "your_template_name"               # the identifier the MCP enum advertises
SUPPORTED_EXTENSIONS = (".ext1", ".ext2") # advisory; warned-not-blocked at runtime
SUMMARY = "What it plots + example user phrasings (this text routes requests)."

# Declarative contract — only for parameterised templates; a summary-only
# template stops at SUMMARY. The generated schema (templates.json), and
# therefore the MCP tool description, is built from these, so ARGS_SPEC
# must mirror render()'s keyword arguments (a test enforces this).
ARGS_SPEC = (
    TemplateArg("channel", "string", required=False,
                description="Channel to plot; defaults to the first."),
)
EXAMPLE_ARGS = {"channel": "EEG Fpz-Cz"}
ARGS_NOTES = ""        # cross-arg rules, e.g. "exactly one of end_time/duration"
PIPELINE_TOOLS = None  # or a tool-family name like "ts_dsp" to accept a `pipeline` arg

def render(target_file_path: str, output_path: str, *, channel: str | None = None) -> None:
    # ALL heavy imports inside render() — see "Lazy imports" below.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import your_format_parser

    data = your_format_parser.load(target_file_path)
    # ... build figure ...
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
```

Raise on failure (missing data, parse error, format mismatch). The dispatcher catches and falls through to the agent — that's the safety net.

### 2. Register in the package

Add one import + one map entry in `processor/templates/__init__.py`:

```python
from processor.templates import your_template_name  # add

_REGISTRY = {
    csv_column_distributions.NAME: csv_column_distributions,
    fcs_channel_histograms.NAME: fcs_channel_histograms,
    your_template_name.NAME: your_template_name,     # add
}
```

### 3. Regenerate the schemas and vendor them into pennsieve-mcp

The MCP `plot_file` tool hardcodes nothing about templates — it embeds the generated JSON schemas (`//go:embed schemas/*.json` in [`plot_file_schema.go`](https://github.com/Pennsieve/pennsieve-mcp/blob/main/internal/tools/plot_file_schema.go)) and derives its `template` enum, argument docs, and pipeline-tool docs from them:

```sh
make schemas                                              # regenerate schema/*.json from the registries
cp schema/*.json ../pennsieve-mcp/internal/tools/schemas/ # manual vendoring step (by design)
```

Ship a `pennsieve-mcp` PR with the updated JSONs alongside this repo's PR. After tagging the quick-plot release, bump the processor `tag` in each org's registered workflow definition — that's where the version is pinned; no MCP code change.

### Lazy imports — strictly required

`processor/templates/__init__.py` imports **every** template module at processor startup so it can build the registry. If any template did `import matplotlib` (or any other heavy import) at module top level, **every plot run** — even ones that pick a *different* template — would pay that cost.

The rule:
- Module top: `math`, stdlib, `NAME`, `SUPPORTED_EXTENSIONS`. Nothing else.
- Inside `render()`: matplotlib, format-specific parsers, pandas, numpy, anything else.

Prefer narrow import paths when a package has a heavy `__init__`. For example, `import flowkit` triggers eager imports of bokeh / gates / Session (~30s cold-start on EFS); `import flowio` (the lightweight parser flowkit wraps) imports in <1s. The v0.6.2 perf gain came from honoring this rule.

If you're adding a dep that isn't already on the EFS layer, also append it to [`workflows/populate-quick-plot-stack.json`](workflows/populate-quick-plot-stack.json)'s `REQUIREMENTS` field and re-run that workflow once per compute node. Test cold-start time after — if `import yourdep` takes more than 2–3 seconds on the layer-mounted EFS, look for a lighter alternative.

## DSP pipeline tools (`processor/tools/ts_dsp/`)

A template that declares `PIPELINE_TOOLS = "ts_dsp"` (today: `edf_processed_timeseries`) accepts an optional `pipeline` argument — an ordered list of `{tool, params}` steps run on the Signal between reading and plotting:

```json
[{"tool": "notch_filter", "params": {"w0": 60}},
 {"tool": "psd",          "params": {"win_size": 2}}]
```

Every tool is a `Signal -> Signal` function registered via the `@dsp_tool` decorator ([`ts_dsp/registry.py`](processor/tools/ts_dsp/registry.py)), declaring its parameters plus two domain dicts that form a small type system over the Signal's axis metadata:

- **`requires`** — axis domains the input must have. Checked against the running Signal *before* each step executes, so an illegal order (`psd` after `energy`, `fft` after `fft`) fails with a specific `ToolInputError` instead of plotting a wrong-axis figure.
- **`produces`** — a **delta**: lists only the axes the tool *changes*; omitted axes pass through untouched (mirroring how tools use `dataclasses.replace`). A tool that changes nothing declares `produces={}`.

Tool families by module:

| module | tools | domains |
|---|---|---|
| [`bandpass.py`](processor/tools/ts_dsp/bandpass.py) | `highpass_filter`, `lowpass_filter`, `bandpass_filter`, `notch_filter` | time/amplitude → unchanged |
| [`smoothing.py`](processor/tools/ts_dsp/smoothing.py) | `moving_average`, `moving_median`, `savgol_filter`, `gaussian_filter1d` | any → unchanged |
| [`frequency.py`](processor/tools/ts_dsp/frequency.py) | `fft`, `psd` | time/amplitude → frequency spectrum |
| [`feature_extraction.py`](processor/tools/ts_dsp/feature_extraction.py) | `energy`, `power`, `rms`, `zcr`, `line_length`, `kurtosis`, `skewness` | time/amplitude → windowed feature series |

The gating rule behind the `requires` column: **any tool whose math converts seconds or Hz via the Signal's `fs` requires `y_domain: "amplitude"`** — the raw trace is the only domain where `fs` is guaranteed truthful (windowed features resample the series without updating `fs`). Pure sample-arithmetic tools (the smoothers) require nothing and run on traces, feature series, and spectra alike. Follow the same rule when adding a tool.

Adding a tool: write the function in the matching module (or a new one, imported in `ts_dsp/__init__.py` so its registrations run), decorate it with `@dsp_tool`, keep heavy imports inside the function (same lazy-import rule as templates), then regenerate + vendor the schemas (step 3 above) so the MCP `plot_file` docs advertise it.

## Generated schemas (`schema/`)

`schema/templates.json` and `schema/ts_tools.json` are generated snapshots of the template/tool contracts — the single source of truth is the code in `processor/templates/` and `processor/tools/`. Regenerate them with `make schemas` whenever a template's contract fields (`SUMMARY`, `ARGS_SPEC`, …) or a tool registry change.

pennsieve-mcp vendors copies at `internal/tools/schemas/` (embedded into the Go binary at compile time) to build the `plot_file` tool's template enum and description text. The copy step is manual by design — update both repos in the same PR pair so they can't drift. Tests in `processor/test_templates/test_generate_schemas.py` pin the JSON shape, check every declared arg is a real `render()` keyword, and assert the checked-in `schema/*.json` match a fresh regeneration — so a forgotten `make schemas` fails the test suite rather than shipping a stale schema.

## Workflow definitions

The workflow definitions consumed by `workflow-service` live in `workflows/`:

- **`plot-templates.json`** — the main workflow (v0.6+). Single stage; Lambda; canned-template-first with agent fallback; viewer-asset data-target.
- **`quick-plot.json`** — the legacy agent-only workflow (kept for backwards compat).
- **`populate-quick-plot-stack.json`** — one-time layer-population workflow. Uses `processor-build-python-layer` (a separate repo) + `persistent-layer` data-target to install the scientific Python stack into the `quick-plot-stack` EFS layer.

Workflow definition registration is API-only in workflow-service today — the JSONs are POSTed to `/definitions` at deploy time. See [`workflows/README.md`](workflows/README.md) for details and open questions.

## Local development

```sh
make build    # build the Docker image
make run      # run via docker-compose (ECS mode), reads dev.env
make schemas  # regenerate schema/*.json from the template/tool registries
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
- Electrophysiology (EDF/iEEG): `pyedflib`
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
