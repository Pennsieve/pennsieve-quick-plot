# workflows/

Workflow definitions consumed by `workflow-service`.

Workflow definition registration is **API-only** in `workflow-service` today — there's no auto-registration on release. These JSON files get POSTed to `workflow-service`'s `/definitions` endpoint at deploy time (e.g. via a release-time script in CI, or manually by an admin).

## Files

| File | Purpose | Runtime | Trigger |
|---|---|---|---|
| `quick-plot.json` | The main v1 workflow. Prompt + file → PNG viewer-asset on the target package. | Lambda | Called by `plot_file` MCP tool when a user asks for a figure in chat |

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
- **Workflows can have NO data-source node** if they don't consume packages. All other workflows need at least one data-source with at least one packageId at run time.

## Target types in the dev registry

(Fetched from `GET /target-types`)

| `targetType` | Purpose | Required params |
|---|---|---|
| `Pennsieve Package Asset` | Attach an asset to a Pennsieve package (used by `quick-plot` for the PNG viewer-asset) | `ASSET_TYPE` |
| `Pennsieve Dataset` | Upload results as a package to a dataset | `UPLOAD_BUCKET` |
| `persistent-layer` | Write output into a named EFS layer (being retired; quick-plot no longer uses one). | `layerName` |

## Deploying these workflows

```sh
export TOKEN="<session-token>"
export BASE="https://api2.pennsieve.net/compute/workflows"  # dev

# Register the workflow definitions
curl -X POST "$BASE/definitions" -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d @workflows/populate-quick-plot-stack.json

curl -X POST "$BASE/definitions" -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d @workflows/quick-plot.json
```


## Verified-working examples in dev

| Workflow | Definition UUID | Notes |
|---|---|---|
| `quick-plot` (v0.2.1 + stub_mode) | `505c3b41-717c-4e9a-aba0-0681083ffc88` | End-to-end smoke test succeeded in 31s — run UUID `1de1fee4-e865-44f8-9785-d096583cb5f5`. Viewer-asset `a32c18ec-225f-428f-b868-22c2fda87b41` attached to package `971f22e1-0b79-41f0-9b9c-4fb9e5ae4fa1` in dataset `74aa3052-fab2-467b-a71f-fa19e493b1c0`. |

### Sample run payload for `quick-plot` (stub-mode smoke test)

```json
{
  "workflowInstanceConfiguration": {
    "workflowId": "505c3b41-717c-4e9a-aba0-0681083ffc88",
    "computeNodeId": "<compute-node-uuid>",
    "processorConfigs": [
      {"nodeId": "quick-plot-processor", "version": "v0.2.1", "executionTarget": "lambda"}
    ]
  },
  "datasetId": "<dataset where user has write permission>",
  "dataSources": {
    "target-file": { "packageIds": ["<csv-or-other-readable-package>"] }
  },
  "dataTargets": {
    "figure-asset": { "params": { "ASSET_TYPE": "PNG", "ASSET_NAME": "quick-plot-smoke" } }
  },
  "processorParams": {
    "quick-plot-processor": { "stub_mode": "1" }
  }
}
```

**Required permission**: the user (whose token POSTs the run) must have **write permission (editor/manager/owner) on the target dataset**. The data-target Lambda creates a viewer-asset via `POST /packages/assets` which requires write — a viewer-only role returns 401 from packages-service's business logic (not from the authorizer). Verified by reading the run's claims: `DatasetClaim.Role: 2` failed; the same user with manager/owner role on a different dataset succeeded.


## The scientific stack

Since v0.8.0 the image carries the scientific stack (`/opt/plot-stack`, `processor/stack-requirements.txt`). Runs no longer name a layer, and the `populate-quick-plot-stack` workflow, which built the former `quick-plot-stack` EFS layer, is retired.

## Open questions still

- **`ASSET_TYPE: "PNG"`** — confirmed working for attaching a viewer-asset PNG (workflow definition accepted), but end-to-end render in package viewer not yet verified.
- **Authorizer dataset-claim requirement for non-dataset workflows** — having to pass a placeholder `datasetId` is a workaround. Long-term fix is in the authorizer's callback flow (treat missing datasetId gracefully).
- **Data-target `defaultParams` ignored at run-creation**: the data-target Lambda received an empty param. Real workflow-service bug or undocumented gap; for now we work around it by passing `dataTargets.<nodeId>.params` at run time.
