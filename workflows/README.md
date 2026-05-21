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
    "datasetId": "<any-dataset-in-the-org>",
    "dataTargets": {
      "quick-plot-stack-layer": {
        "params": { "layerName": "quick-plot-stack" }
      }
    }
  }'
```

**Two non-obvious requirements at run time:**

1. **`datasetId` is required** even though the workflow doesn't read packages. The authorizer's callback-validation flow looks up a dataset claim for the run; without a real `datasetId`, the provisioner's callback to fetch the run gets 403 ("unable to get dataset claim"). Pick any dataset in the org as a placeholder — the layer-builder doesn't read it.

2. **Data-target `defaultParams` are silently ignored at run-creation time.** The workflow definition has `defaultParams.layerName = "quick-plot-stack"` on the `quick-plot-stack-layer` node, but the param-merge logic in workflow-service only merges `defaultParams` for **processor** nodes, not data-targets. Result: the CommitLayer Lambda receives `layerName=""` and fails with `"layerName is required"`. Workaround: pass `dataTargets.<nodeId>.params` at run time as shown above. Permanent fix would be in `workflow-service`'s `compute_trigger` / param-merge logic.

## Verified-working examples in dev

| Workflow | Definition UUID | Notes |
|---|---|---|
| `populate-quick-plot-stack` (no data-source) | `6f7345f8-9f45-4b9c-9362-f43c857ce3f8` | Successfully populated `quick-plot-stack` layer (14,869 files, 792 MB) — run UUID `fe0574e7-449a-481d-9f2f-cabe209be11d` |
| `quick-plot` | `505c3b41-717c-4e9a-aba0-0681083ffc88` | Definition registered; not yet end-to-end run |

(There's also an older `populate-quick-plot-stack` at `4c23010f-...` with a `trigger` data-source — superseded by the no-source version above, since data-source nodes require non-empty packageIds at run time which makes no sense for a layer-population workflow.)

## Performance characteristics — observed

A successful `populate-quick-plot-stack` run takes **~16 minutes total**, broken down:

| Step | Duration | Notes |
|---|---|---|
| `python-layer-builder` (Fargate processor) | ~6 min | `pip install` of the full scientific stack (matplotlib + scanpy + flowkit etc.) into `OUTPUT_DIR/lib/python3.12/site-packages/` |
| `CommitLayer` (Lambda) | **~7-8 min** | Copy from EFS workdir → EFS layer path. **Bottleneck: file count, not bytes.** |
| `CleanupEFS` + `FinalizeWorkflow` | ~2 min | Workdir cleanup, metrics submission, log archival |

**Why CommitLayer is slow** (~33 files/sec, 1.8 MB/s):

The scientific Python stack produces ~15k small files (mostly `.py`, `.pyc`, `.so`, package metadata). EFS metadata operations are limited (~3000 IOPS bursting in elastic mode) and `cp -r`-style serial copy doesn't parallelize them. So you pay ~3 metadata round-trips per file × 15k files, serially.

This is a **one-time cost** — subsequent `quick-plot` workflow runs mount the populated layer read-only in sub-second time. Acceptable for v1 unless we anticipate frequent layer re-populates.

Future optimizations (out of v1 scope):
- Parallel copy in CommitLayer (`xargs -P` or similar)
- Tar-stream between EFS paths (turns N metadata ops into 1)
- Cache-aware re-population (only copy changed files)
- Build outside, ship as a single archive

## Open questions still

- **Re-running a `populate-*` workflow** — does it replace the layer's contents or create a new snapshot? Empirically the existing layer was overwritten in-place (file count grew from 0 → 14,869). Confirm with workflow-service team before re-running on a layer that's currently in use.
- **`ASSET_TYPE: "PNG"`** — confirmed working for attaching a viewer-asset PNG (workflow definition accepted), but end-to-end render in package viewer not yet verified.
- **Authorizer dataset-claim requirement for non-dataset workflows** — having to pass a placeholder `datasetId` is a workaround. Long-term fix is in the authorizer's callback flow (treat missing datasetId gracefully).
- **Data-target `defaultParams` ignored at run-creation** — see "Two non-obvious requirements at run time" above. Real workflow-service bug or undocumented gap; for now we work around it by passing `dataTargets.<nodeId>.params` at run time.
- **CommitLayer serial copy** — the 7-8 min for 15k small files is metadata-bound. Acceptable for a one-time setup but would hurt UX if we re-populate often. Worth tracking as a workflow-service / provisioner perf issue.
