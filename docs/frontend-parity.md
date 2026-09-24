# Frontend Parity

This checklist compares the React renderer with the legacy frontend preserved in Git history.
The local `hhtools/web/frontend-old/` archive is ignored and no longer tracked.
The old frontend defines behavior and Stage semantics; the current visual design
may improve spacing, rounded corners, accessibility, and narrow-screen layout.

| Area | Current state | Required parity work |
| --- | --- | --- |
| Stage / Three.js | Complete | Rendering, playback, HUD state, calibration projection, and interaction match or improve on the old renderer |
| Motion | Complete | Loaded assets include a compact validation summary |
| Robot | Complete | Loaded models include mapping and renderability validation |
| Video to Motion | Runtime-gated | Official GVHMR flow is connected; local execution only lacks the licensed SMPL-X neutral model |
| Human to Robot | Complete | Calibration, comparison, diagnostics, and export are connected |
| Robot to Robot | Complete | Source loading, calibration, comparison, diagnostics, and export are connected |
| Batch | Functional | Run full solver smoke when the licensed/runtime dependencies are available |
| Analysis | Complete | Scatter preview, stable filtering, histogram brushing, and removable upload basket are connected |
| Application menu | Complete | Typed navigation, imports, current export, settings, help, theme, and desktop-only exit are connected |

## Order

- [x] Match the Stage visibility HUD state and asset-family colors.
- [x] Share one complete React calibration editor between H2R and R2R.
- [x] Restore Canvas joint selection, constrained dragging, and synchronized calibration HUD controls.
- [x] Synchronize Inspector/Canvas joint selection and restore robot-link hover/selected feedback.
- [x] Keep projected calibration labels inside the Stage without collisions.
- [x] Apply light/dark semantic tokens to Stage overlays and workflow status surfaces.
- [x] Fit tall robot models against the actual Stage field of view and aspect ratio.
- [x] Implement the three Batch pipelines against the existing FastAPI routes.
  - [x] Add the shared typed import, scan, job, progress, and download boundary.
  - [x] Refresh calibration on re-entry and feed completed V2M clips into the H2R draft.
- [x] Restore H2R scaled preview and omitted result/export controls.
  - [x] Load the calibrated scaled skeleton/scene before Retarget without forcing it visible.
  - [x] Keep the optional pre-Retarget scaled preview frozen on its reference frame.
  - [x] Show shared diagnostics and complete FPS/time/header export controls for H2R/R2R.
  - [x] Restore persisted Source, Target, Result, and Overlay Stage presets.
- [x] Restore remaining Motion, Robot, V2M, Analysis, and application-menu actions.
  - [x] Share one persistent H2R Batch draft with Motion.
  - [x] Preserve the complete registered V2M Motion payload for Stage and H2R.
  - [x] Preserve the last successful Motion while a replacement video is pending or invalid.
  - [x] Keep V2M and Batch mounted while their jobs or drafts are inactive.
  - [x] Preview selected videos and import existing GVHMR `.pt` results.
  - [x] Restore Analysis scatter selection/preview, fixed coordinates, histogram brushing, and upload-basket removal.
  - [x] Connect the desktop Exit command through the trusted Electron IPC boundary.
  - [x] Connect all five application menus without global DOM commands.
  - [x] Keep workflow navigation explicit and remove application-level shortcuts.
  - [x] Restore the nine-step React tutorial; show it automatically only on first entry and keep Help available for replay.
  - [x] Keep File > Exit in the desktop GUI and omit it entirely from the WebUI.
  - [x] Restore language, panel layout, Motion Library, GVHMR, and job settings.
  - [x] Localize every Inspector workflow and shared workflow control in English and Chinese.
  - [x] Keep Batch ownership inside Batch; Motion no longer exposes an add-to-Batch action.
  - [x] Persist Force re-analysis in Workspace Settings; Analysis only consumes the policy.
  - [x] Expose cached analysis as the read-only Load existing result action.
  - [x] Restore About metadata, source, license, and contact information.
  - [x] Use shared refresh icons and semantic destructive-action colors.
  - [x] Keep feature refresh icons borderless in Motion, Robot, V2M, and Batch.
  - [x] Keep hover menus open across the visual trigger-to-popup gap.
  - [x] Keep Robot mesh-folder selection available before import completion.
  - [x] Show Robot Library row status only while a model is loading.
  - [x] Give all six Stage layer types distinct semantic colors.
  - [x] Color V2M, H2R, and R2R step status by neutral, active, success, warning, and failure semantics.
  - [x] Keep the Stage visibility menu legible in dark and narrow layouts.
  - [x] Refresh R2R robot and trajectory catalogs whenever the workspace is entered.
  - [x] Restore compact Motion, Robot, and calibration validation summaries.
  - [x] Link H2R/R2R asset selectors back to the owning import workspaces.
  - [x] Keep Task result export on V2M, H2R, R2R, and Batch rows; retain new artifacts across restarts.
- [x] Simplify Motion Library to search, type filtering, loading, and a Settings link.
- [x] Run route-contract tests and desktop/narrow-screen browser smoke for every pipeline.

## Verification

- Frontend: 105 tests passed; production build passed.
- Web backend: 277 tests passed; 2 dependency-gated tests skipped.
- Task artifacts: 33 focused history, lifecycle, and export-path tests passed.
- Desktop: 38 tests and the Electron window E2E passed.
- Browser: desktop and 390 x 844 layouts have no overflow or panel overlap;
  Motion, all six built-in robots, H2R/R2R calibration, Analysis, and menus were
  exercised without running a long solver.

## External Gate

GVHMR is installed with its public checkpoints and CUDA runtime. A real V2M run
still requires the licensed file at
`/home/nora/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz`.
Workspace Settings links to the official SMPL-X download and model-license pages;
licensed model files remain outside this repository.
True server-side cancellation remains a future job-protocol enhancement; the UI
does not present a client-only abort as cancellation.

## Rules

- FastAPI remains the source of truth for assets, calibration, jobs, and exports.
- Feature views own their transport and workflow state.
- Shared components own reusable presentation and pure interaction rules only.
- Stage consumes typed data; it never starts jobs or calls feature endpoints.
- No compatibility runtime, global DOM lookup, hidden fallback, or duplicate state model.
