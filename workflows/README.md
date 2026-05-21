# workflows/

Workflow definitions consumed by `workflow-service`.

Workflow definition registration is **API-only** in `workflow-service` today — there's no auto-registration on release. These JSON files get POSTed to `workflow-service`'s `/definitions` endpoint at deploy time (e.g. via a release-time script in CI, or manually by an admin).

## Files

| File | Purpose | Runtime | Trigger |
|---|---|---|---|
| `quick-plot.json` | The main v1 workflow. Prompt + file → PNG viewer-asset on the target package. | Lambda | Called by `plot_file` MCP tool when a user asks for a figure in chat |
| `populate-quick-plot-stack.json` | One-time layer-population workflow. Pip-installs the scientific Python stack into the `quick-plot-stack` EFS layer. | Fargate (standard) | Admin-only, manually via `POST /runs` after the EFS layer is created |

## Schema reference

Dev server: `https://api2.pennsieve.net/compute/workflows`
Prod server: `https://api2.pennsieve.io/compute/workflows`

Validated against the live `/definitions` endpoint AND a successful end-to-end run. Key conventions confirmed:

- **No top-level `defaultComputeType`** on the workflow — set `computeType` per node. Enum: `standard | lambda | gpu`.
- **`tag`** on processor nodes pins the GitHub release version (e.g. `"tag": "v0.1.0"`). app-deploy-service auto-builds the container from the GitHub release and tags the ECR image with the same version.
- **`defaultParams`** works on both processor *and* data-target nodes. Values must be strings.
- **`dependsOn`** is omitted (or null) on data-source nodes.
- **`organizationId`** scopes the workflow to an org when set; user-scoped when omitted.
- **Multiple data-targets per processor** is supported.
- **Workflows can have NO data-source node** if they don't consume packages (e.g. `populate-quick-plot-stack`). All other workflows need at least one data-source with at least one packageId at run time.

## Target types in the dev registry

(Fetched from `GET /target-types`)

| `targetType` | Purpose | Required params |
|---|---|---|
| `Pennsieve Package Asset` | Attach an asset to a Pennsieve package (used by `quick-plot` for the PNG viewer-asset) | `ASSET_TYPE` |
| `Pennsieve Dataset` | Upload results as a package to a dataset | `UPLOAD_BUCKET` |
| `persistent-layer` | Write output into a named EFS layer (used by `populate-*` workflows). The layer must exist (status EMPTY) before the workflow runs. | `layerName` |

## Deploying these workflows

```sh
export TOKEN="<session-token>"
export BASE="https://api2.pennsieve.net/compute/workflows"  # dev

# Create the EFS layer first (status EMPTY)
curl -X POST "$BASE/compute-nodes/<COMPUTE_NODE_ID>/layers" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"layerName": "quick-plot-stack", "description": "Scientific Python stack for quick-plot"}'

# Register the workflow definitions
curl -X POST "$BASE/definitions" -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d @workflows/populate-quick-plot-stack.json

curl -X POST "$BASE/definitions" -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d @workflows/quick-plot.json
```

## Triggering `populate-quick-plot-stack` (one-time)

```sh
curl -X POST "$BASE/runs" -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "workflowInstanceConfiguration": {
      "workflowId": "<populate-workflow-uuid>",
      "computeNodeId": "<compute-node-uuid>",
      "processorConfigs": [
        {"nodeId": "python-layer-builder", "version": "v0.1.0", "executionTarget": "standard"}
      ]
    },
    "datasetId": "<any-dataset-in-the-org>"
  }'
```

**Important: `datasetId` is required** even though the workflow doesn't read packages. The authorizer's callback-validation flow looks up a dataset claim for the run; without a real `datasetId`, the provisioner's callback to fetch the run gets 403 ("unable to get dataset claim"). Pick any dataset in the org as a placeholder — the layer-builder doesn't read it.

## Verified-working examples in dev

| Workflow | Definition UUID | Notes |
|---|---|---|
| `populate-quick-plot-stack` (no data-source) | `6f7345f8-9f45-4b9c-9362-f43c857ce3f8` | Successfully populated `quick-plot-stack` layer (~14,869 files, 791 MB, 6 min) |
| `quick-plot` | `505c3b41-717c-4e9a-aba0-0681083ffc88` | Definition registered; not yet run |

(There's also an older `populate-quick-plot-stack` at `4c23010f-...` with a `trigger` data-source — superseded by the no-source version above, since data-source nodes require non-empty packageIds at run time which makes no sense for a layer-population workflow.)

## Open questions still

- **Re-running a `populate-*` workflow** — does it replace the layer's contents or create a new snapshot? Confirm with workflow-service team before re-running.
- **`ASSET_TYPE: "PNG"`** — confirmed working for attaching a viewer-asset PNG (workflow definition accepted), but end-to-end render in package viewer not yet verified.
- **Authorizer dataset-claim requirement for non-dataset workflows** — having to pass a placeholder `datasetId` is a workaround. Long-term fix is in the authorizer's callback flow (treat missing datasetId gracefully).
