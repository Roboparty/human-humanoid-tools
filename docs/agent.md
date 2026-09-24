# Agent interfaces

HHTools exposes the same versioned H2R, scene-free R2R, and scalable Batch Agent contracts
through two local adapters. Use the JSON CLI from scripts and use MCP when a compatible coding
agent should discover and call tools directly.

| Adapter | Runtime | Entry point |
|---|---|---|
| JSON CLI | Client of a running WebUI Agent API | `uv run hhtools agent ...` |
| MCP | Own local stdio process; no WebUI required | `uv run --extra mcp hhtools-mcp` |

The current Agent surface supports human-to-robot (H2R) retargeting—plain motion through Newton
and safely inspectable object-interaction or terrain-scene bundles through Interaction-Mesh—and
scene-free robot-to-robot (R2R) trajectories through Newton, plus ordered H2R/R2R batches. It
includes content-bound H2R and R2R pair-calibration status, constrained proposals, deterministic
validation, front/side visual previews, validated silent save, capability and robot
discovery, allowlisted asset registration and inspection, workflow-specific preflight, jobs,
revision-aware waiting, verified artifacts, and export. Scene-bearing R2R, Video2Motion, Analysis,
remote service access, and robot deployment are not part of this interface. Code-capable
source formats still require safe content inspection; the Agent never bypasses an
isolated-validation requirement.

## Install

Use any compatible Python 3.12 or newer and install the adapter you need:

```bash
# JSON CLI plus the resident WebUI service
uv sync --locked --extra web --extra retarget

# Self-contained local MCP H2R/R2R server
uv sync --locked --extra mcp
```

After installation, `uv run hhtools doctor --require mcp` provides a
side-effect-free readiness check. Add `--json` for a single machine-readable
document; optional body-model and GVHMR checks do not fail the default command.

The JSON CLI always emits one strict JSON document. Start the WebUI service,
then query it from another terminal:

```bash
uv run hhtools web
uv run hhtools agent capabilities
uv run hhtools agent --help
uv run hhtools agent preflight batch --request batch-request.json
uv run hhtools agent calibration status --request calibration-status.json
uv run hhtools agent calibration propose --request calibration-proposal.json
uv run hhtools agent calibration validate --request calibration-validation.json
uv run hhtools agent calibration save --request calibration-save.json
uv run hhtools agent calibration r2r status --request r2r-calibration-status.json
uv run hhtools agent calibration r2r propose --request r2r-calibration-proposal.json
uv run hhtools agent calibration r2r validate --request r2r-calibration-validation.json
uv run hhtools agent calibration r2r save --request r2r-calibration-save.json
uv run hhtools agent job wait JOB_ID --after-revision REVISION --wait-timeout 20
```

Use `hhtools agent asset catalog` to discover registerable Motion Library and
Robot Library entries before the registry contains any assets. MCP clients use
the equivalent read-only `list_available_assets` tool. Both return only an
allowlisted `root_id` and portable `relative_path`, never a host path.

`hhtools-mcp` is a stdio server, so start it through an MCP client rather than
an interactive terminal. Its available options can be inspected with
`uv run --extra mcp hhtools-mcp --help`. Batch caps default to `0` (unlimited); server
administrators may opt into positive `--max-batch-items` and
`--max-batch-total-frames` values. Web, desktop-sidecar, and MCP startup all load the same
persisted setting fields unless an explicit CLI or environment value overrides them.

## Safe H2R, R2R, and Batch workflows

For a new run, discover capabilities and inspect every input. H2R binds one motion and one robot;
R2R binds one scene-free robot trajectory, its declared source robot, the target robot, and their
pair calibration. Call the matching preflight with `run_mode: smoke`, then submit only an immutable
plan returned with `status: ready` through `start_job`; retain its `plan_id` and caller-owned key.
Wait by revision and review the evaluation and manifest before considering a
full run. Validated H2R and R2R pair calibrations may be saved automatically; final motion quality
and full-run approval remain human decisions.

H2R calibration assistance is available before preflight or after a
`CALIBRATION_REQUIRED` response. Bind every request to the exact registered robot bundle and
reference, then call `get_calibration_status`, `propose_calibration`, and
`validate_calibration`. Candidates are immutable and content-addressed; revisions name a parent
candidate plus bounded joint overrides instead of editing stored JSON. `preview_calibration`
returns a deterministic front/side PNG as MCP image content so a vision-capable GPT client can
inspect limb direction, symmetry, trunk attitude, feet, and semantic-target mistakes. Current
[OpenAI models](https://developers.openai.com/api/docs/models) support image input and vision,
but the HHTools server cannot authenticate a client model from a self-reported name;
`model_hint` is audit metadata only.

When deterministic validation is still current, `save_calibration` may write a user-overlay
calibration with `validated_silent`. A GPT client that actually inspected the preview uses
`gpt_vision_silent` and supplies a passing visual-review record. An automatic-calibration request
authorizes this validated save without another prompt. Existing bytes are archived before a
replacement, exact retries are idempotent, and a changed baseline rejects the candidate. Saving
does not authorize a full retarget or physical robot deployment. After saving, always run status
and preflight again so the new
calibration digest is bound into a fresh plan.

H2R preflight requires a current geometric validation record for a manual
calibration. Status checks and successful saves persist that record, bound to
the robot asset, reference family, calibration digest, and validator version
(also the exact motion asset for GLB). Preflight reads the record without
constructing a solver. Missing or failed evidence returns an executable
`get_calibration_status` action and no ready plan. A changed calibration, robot,
reference, clip, or validator version requires a new assessment. Existing
calibrations can be assessed through status without being rewritten.

For robots installed in the user library, automatic saves use
`<user robot root>/.calibration-overlays/<robot id>/` when the historical override
directory overlaps the registered bundle. This keeps the robot asset ID unchanged:
reuse the original IDs for status, exact save retries (including after restart),
and the next preflight. Existing sibling calibrations remain readable fallbacks,
and subsequent Web/CLI saves follow an adopted overlay. The selected calibration
file and its digest are still bound into each execution plan; editing the overlay
invalidates a plan that used its previous contents.

R2R calibration uses the parallel `get_r2r_calibration_status`, `propose_r2r_calibration`,
`validate_r2r_calibration`, `preview_r2r_calibration`, and `save_r2r_calibration` tools. Every
request binds both registered robot bundles. The source robot's zero-configuration FK becomes the
semantic reference skeleton, while the candidate contains only the target robot's joint pose.
Candidates cannot be reused with another pair. The same deterministic checks, bounded GPT visual
review, stale-baseline rejection, previous-version archive, idempotent replay, user-overlay write,
audit-note persistence, and fresh-preflight requirement apply. R2R preflight returns an exact
`get_r2r_calibration_status` Agent action when the pair calibration is missing.

Batch composition happens only after every single item already has a ready H2R or R2R plan.
Call `preflight_batch` with one workflow and the ordered child `plan_id` list. The service rejects
mixed run modes, mixed robot identities, and duplicate inputs. Batch item/frame settings use
`0` for unlimited and default to unlimited; positive administrator-selected values become
content-bound plan limits. Execution is serial so cancellation can stop the current child and
guarantee later children never start. Job status carries only `completed_items / total_items`;
the complete per-item report and portable ZIP are stored as `batch_report` and `batch_archive`
artifacts. The MCP report resource is intended for model-sized reports; if it exceeds the inline
context budget, export the verified artifact instead. A retry always creates a new whole-batch
attempt rather than silently selecting failed items.

Only one local runtime may own a `save-dir`. MCP normally owns its directory
for its entire stdio connection. H2R and R2R automatic calibration run inside that owner and
therefore do not require a WebUI handoff. If a proposal cannot pass after bounded revisions, or a workflow
still requires human-only calibration, disconnect MCP before opening the WebUI against the same
directory; close WebUI before reconnecting. Never run both concurrently.

## Codex

This repository includes a project-scoped [Codex configuration](../.codex/config.toml).
It starts the `hhtools` stdio server with `uv` and keeps its save/cache state in
the ignored `.hhtools/agent/` directory, separate from the default WebUI. Its
job-admission settings live in that directory too. Open a new Codex session from
a trusted checkout after installing the `mcp` extra, then use `/mcp` or
`codex mcp list` to confirm that `hhtools` connected.

The checked-in launcher uses `uv run --frozen --no-sync`, so opening Codex never
changes the repository environment. Run the explicit `uv sync --extra mcp`
installation command again after dependency updates.

Codex should normally use the in-process H2R or R2R calibration tools. For a human-only fallback,
first disconnect the MCP server and run:

```bash
HHTOOLS_WEB_SETTINGS_PATH=.hhtools/agent/job-settings.json \
uv run --frozen --no-sync hhtools web \
  --save-dir .hhtools/agent/save \
  --cache .hhtools/agent/cache \
  --port 8010
```

Close that WebUI before reconnecting MCP. Project-level MCP configuration and
the supported stdio fields are documented in the
[official Codex MCP guide](https://developers.openai.com/codex/mcp).
