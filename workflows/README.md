# workflows/

Workflow definitions consumed by `workflow-service`.

Workflow definition registration is **API-only** in `workflow-service` today — there's no auto-registration on release. These JSON files get POSTed to `workflow-service`'s `/definitions` endpoint at deploy time (e.g. via a release-time script in CI, or manually by an admin).

## Files

| File | Purpose | Runtime | Trigger |
|---|---|---|---|
| `quick-plot.json` | The main v1 workflow. Prompt + file → figure as viewer-asset. | Lambda | Called by `plot_file` MCP tool when a user asks for a figure in chat |
| `populate-quick-plot-stack.json` | One-time layer-population workflow. Pip-installs the scientific Python stack into the `quick-plot-stack` EFS layer. | Fargate (Standard) | Admin-only, manually via `POST /runs` after the layer is provisioned |

## How they get registered

Until workflow-service supports auto-registration, the deploy story is:

```sh
# Pseudo-code; actual script TBD
curl -X POST "$WORKFLOW_SERVICE_URL/definitions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @workflows/quick-plot.json

curl -X POST "$WORKFLOW_SERVICE_URL/definitions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @workflows/populate-quick-plot-stack.json
```

## Open questions (verify before first deploy)

- **Exact JSON field names** — `defaultComputeType`, `dependsOn`, `targetType`, `defaultParams`, `params`, `sourceUrl` — these are derived from workflow-service docs/code. Confirm against the live `/definitions` schema.
- **No-data-source workflows** — does `populate-quick-plot-stack` validate without a `data-source` node? If not, add a placeholder source or use a different mechanism for layer-population.
- **`persistent-layer` data-target params** — `layer_name` is the assumed param key. Verify against the data-target registry (or `processor-post-viewer-asset` for the pattern).
- **`requirements` shape** — currently a multi-line string (matches what `processor-build-python-layer`'s parser expects). If workflow-service prefers structured arrays for params, update both ends.
- **Layer versioning / re-population semantics** — how does workflow-service handle running `populate-quick-plot-stack` a second time? Replace? New snapshot? Concurrent runs?
