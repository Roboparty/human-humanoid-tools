# HHTools Agent contract map

Use the live MCP tool input/output schema as the runtime authority. These repository snapshots
explain the stable Agent v1 documents and are useful when a field, state, or resource is
unfamiliar. Load only the contracts needed for the current step.

Installation and supported boundaries are documented in the public
[Agent guide](../../../../docs/agent.md).

## Tool and schema routing

| MCP operation | Request contract | Success contract |
|---|---|---|
| `get_capabilities` | No request document | [capabilities](../../../../docs/schemas/agent/v1/capabilities.schema.json) |
| `list_robots` | No request document | [robot list](../../../../docs/schemas/agent/v1/robot-list-response.schema.json) |
| `register_asset_bundle` | [asset registration request](../../../../docs/schemas/agent/v1/asset-registration-request.schema.json) | [asset bundle](../../../../docs/schemas/agent/v1/asset-bundle.schema.json) |
| `search_assets` | Bounded scalar filters from the live tool schema | [asset search response](../../../../docs/schemas/agent/v1/asset-search-response.schema.json) |
| `list_available_assets` | [available catalog request](../../../../docs/schemas/agent/v1/available-asset-catalog-request.schema.json) | [available catalog response](../../../../docs/schemas/agent/v1/available-asset-catalog-response.schema.json) |
| `inspect_asset_bundle` | `asset_id`, `verify_hashes`, and `parse_content` from the live tool schema | [asset inspection](../../../../docs/schemas/agent/v1/asset-inspection.schema.json) |
| `preflight_retarget` | [retarget preflight request](../../../../docs/schemas/agent/v1/retarget-preflight-request.schema.json) | [preflight response](../../../../docs/schemas/agent/v1/preflight-response.schema.json) |
| `preflight_r2r` | [R2R preflight request](../../../../docs/schemas/agent/v1/r2r-preflight-request.schema.json) | [R2R preflight response](../../../../docs/schemas/agent/v1/r2r-preflight-response.schema.json) |
| `preflight_batch` | [batch preflight request](../../../../docs/schemas/agent/v1/batch-preflight-request.schema.json) | [batch preflight response](../../../../docs/schemas/agent/v1/batch-preflight-response.schema.json) |
| `get_calibration_status` | [calibration status request](../../../../docs/schemas/agent/v1/calibration-status-request.schema.json) | [calibration status response](../../../../docs/schemas/agent/v1/calibration-status-response.schema.json) |
| `propose_calibration` | [calibration proposal request](../../../../docs/schemas/agent/v1/calibration-proposal-request.schema.json) | [calibration proposal response](../../../../docs/schemas/agent/v1/calibration-proposal-response.schema.json) |
| `validate_calibration` | [calibration validation request](../../../../docs/schemas/agent/v1/calibration-validation-request.schema.json) | [calibration validation report](../../../../docs/schemas/agent/v1/calibration-validation-report.schema.json) |
| `preview_calibration` | [calibration preview request](../../../../docs/schemas/agent/v1/calibration-preview-request.schema.json) | [calibration preview metadata](../../../../docs/schemas/agent/v1/calibration-preview.schema.json) plus one MCP image block |
| `save_calibration` | [calibration save request](../../../../docs/schemas/agent/v1/calibration-save-request.schema.json) | [calibration save receipt](../../../../docs/schemas/agent/v1/calibration-save-receipt.schema.json) |
| `get_r2r_calibration_status` | [R2R calibration status request](../../../../docs/schemas/agent/v1/r2r-calibration-status-request.schema.json) | [R2R calibration status response](../../../../docs/schemas/agent/v1/r2r-calibration-status-response.schema.json) |
| `propose_r2r_calibration` | [R2R calibration proposal request](../../../../docs/schemas/agent/v1/r2r-calibration-proposal-request.schema.json) | [R2R calibration proposal response](../../../../docs/schemas/agent/v1/r2r-calibration-proposal-response.schema.json) |
| `validate_r2r_calibration` | [R2R calibration validation request](../../../../docs/schemas/agent/v1/r2r-calibration-validation-request.schema.json) | [calibration validation report](../../../../docs/schemas/agent/v1/calibration-validation-report.schema.json) |
| `preview_r2r_calibration` | [R2R calibration preview request](../../../../docs/schemas/agent/v1/r2r-calibration-preview-request.schema.json) | [R2R calibration preview metadata](../../../../docs/schemas/agent/v1/r2r-calibration-preview.schema.json) plus one MCP image block |
| `save_r2r_calibration` | [R2R calibration save request](../../../../docs/schemas/agent/v1/r2r-calibration-save-request.schema.json) | [R2R calibration save receipt](../../../../docs/schemas/agent/v1/r2r-calibration-save-receipt.schema.json) |
| `start_job` / `start_retarget` | [job start request](../../../../docs/schemas/agent/v1/job-start-request.schema.json) | [agent job view](../../../../docs/schemas/agent/v1/agent-job-view.schema.json) |
| `lookup_job` | [job lookup request](../../../../docs/schemas/agent/v1/job-lookup-request.schema.json) | [agent job view](../../../../docs/schemas/agent/v1/agent-job-view.schema.json) |
| `get_job` / `wait_job` / `cancel_job` | Scalar job identity and live tool fields | [agent job view](../../../../docs/schemas/agent/v1/agent-job-view.schema.json) |
| `retry_job` | [job retry request](../../../../docs/schemas/agent/v1/job-retry-request.schema.json) | [agent job view](../../../../docs/schemas/agent/v1/agent-job-view.schema.json) |
| `list_job_artifacts` | `job_id`, `limit`, and `offset` | [artifact list response](../../../../docs/schemas/agent/v1/artifact-list-response.schema.json) |
| `export_artifact` | Scalar `job_id` and `artifact_id` | [artifact export receipt](../../../../docs/schemas/agent/v1/artifact-export-receipt.schema.json) |

Expected tool failures use the [API error](../../../../docs/schemas/agent/v1/api-error.schema.json)
contract rather than a prose-only exception. Inspect `code`, `retryable`, `stage`, `details`, and
`next_action`; do not recover from the human-readable message alone.

## Executable next-action mapping

`NextAction.action` is executable only when it has an exact mapping in this table:

| `actor` | `action` | Tool | Parameter contract |
|---|---|---|---|
| `agent` | `register_asset_bundle` | `register_asset_bundle` | `parameters` is the complete tool argument object: `{"request": <AssetRegistrationRequest>}`. Pass it unchanged. |
| `agent` | `get_calibration_status` | `get_calibration_status` | `parameters` is the complete tool argument object: `{"request": <CalibrationStatusRequest>}`. Pass it unchanged, then follow candidate validation before saving. |
| `agent` | `get_r2r_calibration_status` | `get_r2r_calibration_status` | `parameters` is the complete tool argument object: `{"request": <R2RCalibrationStatusRequest>}`. Pass it unchanged, preserve both robot identities, then follow R2R candidate validation before saving. |

The returned request contains only a capability-advertised `root_id` and normalized
`relative_path`; it never contains the installed preset's host path. After registration, inspect
the returned bundle and rerun preflight with its `asset_id`. An unknown action or malformed
parameter object is a stop condition, not permission to infer another tool or browse a root.

## Read-only resources

Use these exact URI shapes:

```text
hhtools://capabilities
hhtools://schemas/agent/v1/{schema_name}
hhtools://robots/{robot_id}
hhtools://assets/{asset_id}/manifest
hhtools://plans/{plan_id}
hhtools://jobs/{job_id}/status
hhtools://jobs/{job_id}/manifest
hhtools://jobs/{job_id}/evaluation
hhtools://jobs/{job_id}/batch
hhtools://jobs/{job_id}/failures
hhtools://jobs/{job_id}/artifacts/{artifact_id}
```

For the schema resource, `{schema_name}` is the exact registry slug with no filename suffix—for
example, `capabilities` or `job-spec-v2`, never `capabilities.schema.json`.

The report resources validate managed bytes before returning the versioned
[evaluation report](../../../../docs/schemas/agent/v1/evaluation-report.schema.json),
[failure report](../../../../docs/schemas/agent/v1/failure-report.schema.json), or
[job manifest](../../../../docs/schemas/agent/v1/job-manifest.schema.json). The job-scoped
artifact resource returns only a verified
[artifact descriptor](../../../../docs/schemas/agent/v1/artifact.schema.json). It does not stream
binary content.

## Field placement and identity rules

- Operational Agent v1 request and response envelopes use `schema_version: "1.0"` and reject
  unknown fields. The audit-only JobSpec v2 embedded in a manifest is the explicit exception and
  uses integer `schema_version: 2`.
- `register_asset_bundle` identifies a deployment-owned source with `root_id + relative_path`.
  Backslashes, absolute paths, drive paths, `.` segments, and `..` traversal are not portable
  registration inputs.
- `list_available_assets` is the only root browsing boundary. It returns a bounded page of
  registerable metadata, never host paths or file bytes. Use its returned fields unchanged when
  constructing an `AssetRegistrationRequest`; do not infer another path from display text.
- `asset_id`, `plan_id`, `job_id`, and `artifact_id` are distinct identities. Never derive one
  from a display name or host path.
- A [calibration candidate](../../../../docs/schemas/agent/v1/calibration-candidate.schema.json)
  is an immutable content-addressed pose bound to one robot bundle and reference. Revision creates
  a new candidate with `parent_candidate_id`; it never mutates the original.
- An [R2R calibration candidate](../../../../docs/schemas/agent/v1/r2r-calibration-candidate.schema.json)
  is bound to both source and target robot bundle digests. Its joint vector belongs to the target;
  the source zero-configuration FK supplies the reference. It cannot cross either pair identity.
- H2R returns its motion/robot plan through `preflight_retarget`. Scene-free R2R returns an
  immutable [R2R plan](../../../../docs/schemas/agent/v1/r2r-plan.schema.json) binding the
  trajectory, source robot, target robot, and pair calibration. Do not exchange either robot after
  preflight.
- Batch takes only ordered ready child plan IDs and returns an immutable
  [batch plan](../../../../docs/schemas/agent/v1/batch-plan.schema.json). Its
  [batch report](../../../../docs/schemas/agent/v1/batch-report.schema.json) contains the complete
  per-item results; routine job status exposes only compact completed/total counts. A report too
  large for inline MCP context remains available as a verified exportable artifact.
- `run_mode` belongs to the workflow preflight request's `parameters`. It is frozen in the returned
  plan. [Job start](../../../../docs/schemas/agent/v1/job-start-request.schema.json) has no override.
- Use `output_policy: create_new`. The current PreflightService rejects `overwrite` and
  `fail_if_exists` as unsupported rather than treating them as user-selectable alternatives.
- An idempotency key binds one logical start request. Reuse it only with the exact same plan
  when delivery of the response is uncertain. Persist that pair before calling `start_job`;
  `lookup_job` accepts only the exact pair and recovers one submission without listing other jobs.
- `AgentJobView.artifacts` is compact and may contain only the first page. Use
  `artifact_count` and `list_job_artifacts` for canonical pagination.
- Every artifact lookup requires the owning `job_id` and `artifact_id`; verify one descriptor by
  reading its exact job-scoped resource URI. Binary data is never embedded as Base64.
- `export_artifact` is the MCP file-delivery boundary. It verifies canonical managed bytes, writes
  them only below the service-configured `agent-exports` root, and returns a portable
  `root_id + relative_path` receipt with size and SHA-256. It accepts no caller-selected host path
  and exposes neither the private content-addressed store nor file bytes.

## Audit-only public schemas

The immutable [JobSpec v2](../../../../docs/schemas/agent/v1/job-spec-v2.schema.json), with integer
`schema_version: 2`, appears in the terminal manifest but is not a replacement for preflight. The
public REST/CLI legacy upgrade
contracts—[request](../../../../docs/schemas/agent/v1/legacy-job-upgrade-request.schema.json),
[response](../../../../docs/schemas/agent/v1/legacy-job-upgrade-response.schema.json), and
[receipt](../../../../docs/schemas/agent/v1/legacy-migration-receipt.schema.json)—remain useful for
audit interpretation, but the initial MCP surface has no legacy-upgrade tool. Do not fall back to
the CLI or manufacture a v2 document inside this skill.

## Current boundary

The MCP stdio process assembles the same transport-neutral application services directly; it is
not a REST client and does not need `hhtools web` running. H2R and R2R pair-calibration status,
proposals, validation, visual preview, and validated silent save run inside this same owner. The
model hint and visual-review declaration are audit metadata, not proof of client identity. A given `save_dir`
still has exactly one local runtime owner; never request a WebUI session token or run MCP and Web
concurrently against the same directory. There is no authenticated remote MCP
transport, multi-user authorization, cross-process native-worker resume, or guaranteed actual-GPU
provenance in this phase. `lookup_job` can recover the persisted identity and truthful status of a
known submission; it cannot resume interrupted native execution.
