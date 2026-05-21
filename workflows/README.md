# workflows/

Workflow definitions consumed by `workflow-service`.

Workflow definition registration is **API-only** in `workflow-service` today — there's no auto-registration on release. These JSON files get POSTed to `workflow-service`'s `/definitions` endpoint at deploy time (e.g. via a release-time script in CI, or manually by an admin).

## Files

| File | Purpose | Runtime | Trigger |
|---|---|---|---|
| `quick-plot.json` | The main v1 workflow. Prompt + file → PNG viewer-asset on the target package. | Lambda | Called by `plot_file` MCP tool when a user asks for a figure in chat |
| `populate-quick-plot-stack.json` | One-time layer-population workflow. Pip-installs the scientific Python stack into the `quick-plot-stack` EFS layer. | Fargate (standard) | Admin-only, manually via `POST /runs` after the layer is provisioned |

## Schema reference

Dev server: `https://api2.pennsieve.net/compute/workflows`
Prod server: `https://api2.pennsieve.io/compute/workflows`

Validated against the live `/definitions` endpoint. Key conventions confirmed:

- **No top-level `defaultComputeType`** on the workflow — set `computeType` per node. Enum: `standard | lambda | gpu`.
- **`tag`** on processor nodes pins the GitHub release version (e.g. `"tag": "v0.1.0"`).
- **`defaultParams`** works on both processor *and* data-target nodes. Values must be strings.
- **`dependsOn`** is omitted (or null) on data-source nodes.
- **`organizationId`** scopes the workflow to an org when set; user-scoped when omitted.
- **Multiple data-targets per processor** is supported — see the `Scout +RAG` workflow in dev for an example.

## Target types in the dev registry

(Fetched from `GET /target-types` on 2026-05-21)

| `targetType` | Purpose | Required params |
|---|---|---|
| `Pennsieve Package Asset` | Attach an asset to a Pennsieve package (used by `quick-plot` for the PNG viewer-asset) | `ASSET_TYPE` |
| `Pennsieve Dataset` | Upload results as a package to a dataset | `UPLOAD_BUCKET` |
| `persistent-layer` | Write output into a named EFS layer (used by `populate-*` workflows) | `layerName` |

## Deploying these workflows

```sh
# Set env
export TOKEN="<session-token>"
export BASE="https://api2.pennsieve.net/compute/workflows"  # dev

# POST each
curl -X POST "$BASE/definitions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @workflows/quick-plot.json

curl -X POST "$BASE/definitions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @workflows/populate-quick-plot-stack.json
```

The response includes the workflow `uuid`, which you'll need for `POST /runs`.

## Already deployed to dev

| Workflow | UUID (dev) | Created |
|---|---|---|
| `populate-quick-plot-stack` | `4c23010f-6548-49a8-a111-389fd4a8d98f` | 2026-05-21 |
| `quick-plot` | `505c3b41-717c-4e9a-aba0-0681083ffc88` | 2026-05-21 |

## Remaining open questions

- **Re-running a `populate-*` workflow** — does it replace the layer's contents or create a new snapshot? Confirm with workflow-service team before re-running.
- **`ASSET_TYPE: "PNG"`** — confirmed working for attaching a viewer-asset PNG, but verify that the figure surfaces correctly in the package viewer end-to-end.
