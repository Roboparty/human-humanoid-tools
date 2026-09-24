import { useEffect, useMemo, useRef, useState } from "react";

import { Field, fieldClass } from "@/components/Field";
import { CalibrationEditor } from "@/components/CalibrationEditor";
import { AssetImportButton } from "@/components/AssetImportButton";
import {
  normalizeCalibrationValues,
  setCalibrationJointValue,
  type CalibrationAngleUnit,
} from "@/components/calibrationEditorState";
import { InspectorPage } from "@/components/Inspector";
import { Button } from "@/components/ui/button";
import {
  WorkflowPipeline,
  WorkflowStep,
  type WorkflowStatusTone,
} from "@/components/WorkflowSteps";
import { useLocaleText } from "@/LocaleProvider";
import { displayFileName } from "@/lib/api";
import { ResultDiagnostics } from "@/features/result/ResultDiagnostics";
import { ResultExportControls } from "@/features/result/ResultExportControls";
import type { ComparisonPreset } from "@/features/result/comparison";
import {
  DEFAULT_CALIBRATION_DISPLAY,
  type CalibrationDisplayOptions,
} from "@/stage/calibrationDisplay";
import type { CalibrationInteractionModel } from "@/stage/calibrationInteraction";

import {
  getR2rCalibrationSession,
  getR2rCalibrationStatus,
  getR2rLibrary,
  getRobotLibrary,
  loadR2rLibraryEntry,
  loadRobot,
  previewR2rCalibrationPose,
  r2rEntriesForSourceRobot,
  r2rExportUrl,
  runR2rRetarget,
  saveR2rCalibration,
  uploadR2rTrajectory,
  type MotionLibraryEntry,
  type RobotPayload,
  type RobotSummary,
  type R2rBackend,
  type R2rCalibrationReference,
  type R2rCalibrationSession,
  type R2rCalibrationPose,
  type R2rRetargetResult,
  type R2rSourceResult,
} from "./api";

type BusyAction =
  | "source-robot"
  | "source-trajectory"
  | "target-robot"
  | "calibration-open"
  | "calibration-save"
  | "retarget";

interface StepStatus {
  readonly label: string;
  readonly tone: WorkflowStatusTone;
}

export interface RobotToRobotViewProps {
  active: boolean;
  currentSourceRobot?: RobotPayload | null;
  currentTargetRobot?: RobotPayload | null;
  currentSourceResult?: R2rSourceResult | null;
  currentResult?: R2rRetargetResult | null;
  onSourceRobotLoaded?: (robot: RobotPayload | null) => void;
  onTargetRobotLoaded?: (robot: RobotPayload | null) => void;
  onSourceLoaded?: (result: R2rSourceResult | null) => void;
  onResultLoaded?: (result: R2rRetargetResult | null) => void;
  onCalibrationReference?: (reference: R2rCalibrationReference | null) => void;
  onTargetPose?: (pose: R2rCalibrationPose | null) => void;
  calibrationDisplay?: CalibrationDisplayOptions;
  onCalibrationDisplayChange?: (value: CalibrationDisplayOptions) => void;
  onCalibrationInteraction?: (
    interaction: CalibrationInteractionModel | null,
  ) => void;
  comparisonPreset?: ComparisonPreset;
  onComparisonPresetChange?: (preset: ComparisonPreset) => void;
  onOpenRobotLibrary: () => void;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function positiveNumber(value: string): number | undefined {
  const parsed = Number(value);
  return value.trim() && Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

function entryLabel(entry: MotionLibraryEntry, fallback = "Trajectory"): string {
  return (
    entry.stem ||
    entry.sequence_id ||
    entry.label ||
    displayFileName(entry.source_path, fallback)
  );
}

function suggestedBackend(result?: R2rSourceResult | null): R2rBackend {
  return result?.suggested_backend === "interaction_mesh"
    ? "interaction_mesh"
    : "newton";
}

function RobotSelect({
  label,
  robots,
  value,
  loaded,
  disabled,
  onChange,
  onLoad,
  onImport,
}: {
  label: string;
  robots: readonly RobotSummary[];
  value: string;
  loaded: RobotPayload | null;
  disabled: boolean;
  onChange: (value: string) => void;
  onLoad: () => void;
  onImport: () => void;
}) {
  const text = useLocaleText();
  return (
    <div className="grid gap-2.5">
      <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-2">
        <select
          className={fieldClass}
          aria-label={label}
          value={value}
          disabled={disabled || robots.length === 0}
          onChange={(event) => onChange(event.target.value)}
        >
          {!robots.length && (
            <option value="">{text("No robots available", "没有可用机器人")}</option>
          )}
          {robots.map((robot) => (
            <option
              key={robot.name}
              value={robot.name}
              disabled={!robot.has_urdf}
            >
              {robot.display_name} ({robot.num_dof} {text("DoF", "自由度")})
            </option>
          ))}
        </select>
        <AssetImportButton kind="robot" onClick={onImport} />
      </div>
      <Button
        size="sm"
        variant="primary"
        disabled={disabled || !value}
        onClick={onLoad}
      >
        {text("Load", "加载")}
      </Button>
      <p className="text-xs text-muted-foreground">
        {loaded
          ? text(`${loaded.display_name} loaded`, `${loaded.display_name} 已加载`)
          : text("Not loaded", "未加载")}
      </p>
    </div>
  );
}

export function RobotToRobotView({
  active,
  currentSourceRobot,
  currentTargetRobot,
  currentSourceResult,
  currentResult,
  onSourceRobotLoaded,
  onTargetRobotLoaded,
  onSourceLoaded,
  onResultLoaded,
  onCalibrationReference,
  onTargetPose,
  calibrationDisplay: controlledCalibrationDisplay,
  onCalibrationDisplayChange,
  onCalibrationInteraction,
  comparisonPreset,
  onComparisonPresetChange,
  onOpenRobotLibrary,
}: RobotToRobotViewProps) {
  const text = useLocaleText();
  const pipeline = [
    text("Source Robot", "源机器人"),
    text("Source Trajectory", "源轨迹"),
    text("Target Robot", "目标机器人"),
    text("Calibration", "标定"),
    text("Result", "结果"),
  ];
  const [robots, setRobots] = useState<readonly RobotSummary[]>([]);
  const [entries, setEntries] = useState<readonly MotionLibraryEntry[]>([]);
  const [sourceChoice, setSourceChoice] = useState(currentSourceRobot?.name ?? "");
  const [targetChoice, setTargetChoice] = useState(currentTargetRobot?.name ?? "");
  const [trajectoryChoice, setTrajectoryChoice] = useState("");
  const [sourceRobot, setSourceRobot] = useState<RobotPayload | null>(
    currentSourceRobot ?? null,
  );
  const [targetRobot, setTargetRobot] = useState<RobotPayload | null>(
    currentTargetRobot ?? null,
  );
  const [sourceResult, setSourceResult] = useState<R2rSourceResult | null>(
    currentSourceResult ?? null,
  );
  const [retargetResult, setRetargetResult] =
    useState<R2rRetargetResult | null>(currentResult ?? null);
  const [sourceFps, setSourceFps] = useState("");
  const [retargetFps, setRetargetFps] = useState("");
  const [backend, setBackend] = useState<R2rBackend>(
    suggestedBackend(currentSourceResult),
  );
  const [calibrated, setCalibrated] = useState(false);
  const [checkingCalibration, setCheckingCalibration] = useState(false);
  const [calibration, setCalibration] = useState<R2rCalibrationSession | null>(null);
  const [jointQ, setJointQ] = useState<Record<string, number>>({});
  const [jointGeometry, setJointGeometry] = useState<{
    readonly jointWorld: R2rCalibrationSession["joint_world"];
    readonly groundOffsetZ: number;
  } | null>(null);
  const [angleUnit, setAngleUnit] = useState<CalibrationAngleUnit>("rad");
  const [selectedCalibrationJoint, setSelectedCalibrationJoint] =
    useState<string | null>(null);
  const [calibrationBaseline, setCalibrationBaseline] = useState<
    Record<string, number>
  >({});
  const [localCalibrationDisplay, setLocalCalibrationDisplay] = useState(
    DEFAULT_CALIBRATION_DISPLAY,
  );
  const calibrationDisplay =
    controlledCalibrationDisplay ?? localCalibrationDisplay;
  const publishCalibrationDisplay =
    onCalibrationDisplayChange ?? setLocalCalibrationDisplay;
  const [calibrationSaved, setCalibrationSaved] = useState(false);
  const [busy, setBusy] = useState<BusyAction | null>(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [errorOwner, setErrorOwner] = useState<BusyAction | null>(null);
  const fileInput = useRef<HTMLInputElement | null>(null);
  const folderInput = useRef<HTMLInputElement | null>(null);
  const actionRequest = useRef<AbortController | null>(null);
  const calibrationStatusRequest = useRef<AbortController | null>(null);
  const catalogPrefetched = useRef(false);
  const poseCallback = useRef(onTargetPose);
  const referenceCallback = useRef(onCalibrationReference);
  const interactionCallback = useRef(onCalibrationInteraction);
  const poseWasActive = useRef(false);
  poseCallback.current = onTargetPose;
  referenceCallback.current = onCalibrationReference;
  interactionCallback.current = onCalibrationInteraction;

  useEffect(
    () => () => {
      actionRequest.current?.abort();
      calibrationStatusRequest.current?.abort();
    },
    [],
  );

  // Preload once with the rest of the workspace, then refresh again on entry.
  // Leaving the panel keeps its draft mounted without issuing another request.
  useEffect(() => {
    if (!active && catalogPrefetched.current) return;
    const request = new AbortController();
    let pendingCatalogs = 2;
    const settleCatalog = () => {
      pendingCatalogs -= 1;
      if (pendingCatalogs === 0 && !request.signal.aborted) {
        catalogPrefetched.current = true;
      }
    };
    void getRobotLibrary({ signal: request.signal })
      .then((catalog) => {
        if (request.signal.aborted) return;
        setRobots(catalog.robots);
        const available = catalog.robots.filter((robot) => robot.has_urdf);
        const first = available[0]?.name ?? "";
        setSourceChoice((current) => current || first);
        setTargetChoice((current) => current || available[1]?.name || first);
      })
      .catch((reason: unknown) => {
        if (!request.signal.aborted) {
          setError(errorMessage(reason));
          setErrorOwner(null);
        }
      })
      .finally(settleCatalog);
    void getR2rLibrary({ signal: request.signal })
      .then((libraryEntries) => {
        if (request.signal.aborted) return;
        setEntries(libraryEntries);
        setTrajectoryChoice((current) => current || libraryEntries[0]?.source_path || "");
      })
      .catch((reason: unknown) => {
        if (!request.signal.aborted) {
          setError(errorMessage(reason));
          setErrorOwner(null);
        }
      })
      .finally(settleCatalog);
    return () => request.abort();
  }, [active]);

  useEffect(() => {
    calibrationStatusRequest.current?.abort();
    if (currentSourceRobot === undefined) return;
    setError(null);
    setErrorOwner(null);
    setSourceRobot(currentSourceRobot);
    setSourceChoice(currentSourceRobot?.name ?? "");
  }, [currentSourceRobot]);

  useEffect(() => {
    if (currentTargetRobot === undefined) return;
    setError(null);
    setErrorOwner(null);
    setTargetRobot(currentTargetRobot);
    setTargetChoice(currentTargetRobot?.name ?? "");
  }, [currentTargetRobot]);

  useEffect(() => {
    if (currentSourceResult === undefined) return;
    setError(null);
    setErrorOwner(null);
    setSourceResult(currentSourceResult);
    setBackend(suggestedBackend(currentSourceResult));
  }, [currentSourceResult]);

  useEffect(() => {
    if (currentResult === undefined) return;
    setError(null);
    setErrorOwner(null);
    setRetargetResult(currentResult);
  }, [currentResult]);

  // Calibration is stored for one exact source/target pair. Recheck it whenever
  // either loaded robot changes instead of trusting stale UI state.
  useEffect(() => {
    setCalibration(null);
    referenceCallback.current?.(null);
    setJointQ({});
    setJointGeometry(null);
    setSelectedCalibrationJoint(null);
    setCalibrationBaseline({});
    setCalibrationSaved(false);
    setCalibrated(false);
    setCheckingCalibration(false);
    if (!sourceRobot || !targetRobot) return;
    const request = new AbortController();
    calibrationStatusRequest.current = request;
    setCheckingCalibration(true);
    void getR2rCalibrationStatus(targetRobot.name, sourceRobot.name, {
      signal: request.signal,
    })
      .then((response) => {
        if (!request.signal.aborted) setCalibrated(Boolean(response.calibrated));
      })
      .catch((reason: unknown) => {
        if (!request.signal.aborted) {
          setError(errorMessage(reason));
          setErrorOwner("calibration-open");
        }
      })
      .finally(() => {
        if (!request.signal.aborted) setCheckingCalibration(false);
      });
    return () => request.abort();
  }, [sourceRobot, targetRobot]);

  useEffect(() => {
    if (!calibration || !targetRobot) {
      if (poseWasActive.current) {
        poseWasActive.current = false;
        poseCallback.current?.(null);
      }
      return;
    }
    poseWasActive.current = true;
    const request = new AbortController();
    const timer = window.setTimeout(() => {
      void previewR2rCalibrationPose(targetRobot.name, jointQ, {
        signal: request.signal,
      })
        .then((pose) => {
          if (!request.signal.aborted) {
            setJointGeometry({
              jointWorld: pose.joint_world,
              groundOffsetZ: pose.ground_offset_z,
            });
            poseCallback.current?.(pose);
          }
        })
        .catch((reason: unknown) => {
          if (!request.signal.aborted) {
            setError(errorMessage(reason));
            setErrorOwner("calibration-open");
          }
        });
    }, 120);
    return () => {
      window.clearTimeout(timer);
      request.abort();
    };
  }, [calibration, jointQ, targetRobot]);

  useEffect(
    () => () => {
      if (poseWasActive.current) poseCallback.current?.(null);
      referenceCallback.current?.(null);
      interactionCallback.current?.(null);
    },
    [],
  );

  useEffect(() => {
    if (!calibration || !jointGeometry) {
      interactionCallback.current?.(null);
      return;
    }
    interactionCallback.current?.({
      jointQ,
      jointLimits: calibration.joint_limits,
      jointWorld: jointGeometry.jointWorld,
      groundOffsetZ: jointGeometry.groundOffsetZ,
      angleUnit,
      selectedJoint: selectedCalibrationJoint,
      disabled: busy !== null,
      onJointChange: (name, value) => {
        setJointQ((current) =>
          setCalibrationJointValue(
            calibration.joint_limits,
            current,
            name,
            value,
          ),
        );
      },
      onSelectedJointChange: setSelectedCalibrationJoint,
      onAngleUnitChange: setAngleUnit,
    });
  }, [
    angleUnit,
    busy,
    calibration,
    jointGeometry,
    jointQ,
    selectedCalibrationJoint,
  ]);

  const compatibleEntries = useMemo(
    () => r2rEntriesForSourceRobot(entries, sourceRobot?.name),
    [entries, sourceRobot?.name],
  );
  const selectedEntry = useMemo(
    () =>
      compatibleEntries.find(
        (entry) => entry.source_path === trajectoryChoice,
      ) ?? null,
    [compatibleEntries, trajectoryChoice],
  );

  useEffect(() => {
    setTrajectoryChoice((current) =>
      compatibleEntries.some((entry) => entry.source_path === current)
        ? current
        : compatibleEntries[0]?.source_path || "",
    );
  }, [compatibleEntries]);
  const activeStep = calibration
    ? 3
    : retargetResult || calibrated
      ? 4
      : targetRobot
        ? 3
        : sourceResult
          ? 2
          : sourceRobot
            ? 1
            : 0;
  const blockedReason = (() => {
    if (!sourceRobot) return text("Load the source robot first.", "请先加载源机器人。");
    if (!sourceResult) return text("Load a source trajectory first.", "请先加载源轨迹。");
    if (!targetRobot) return text("Load the target robot first.", "请先加载目标机器人。");
    if (calibration) {
      return text(
        "Save the open calibration before retargeting.",
        "开始重定向前，请保存当前标定。",
      );
    }
    if (checkingCalibration) return text("Checking calibration…", "正在检查标定…");
    if (!calibrated) {
      return text(
        "Save calibration for this robot pair first.",
        "请先保存这组机器人的标定。",
      );
    }
    return null;
  })();

  function beginAction(action: BusyAction): AbortController {
    actionRequest.current?.abort();
    const request = new AbortController();
    actionRequest.current = request;
    setBusy(action);
    setProgress(0);
    setError(null);
    setErrorOwner(null);
    return request;
  }

  function finishAction(request: AbortController): void {
    if (actionRequest.current !== request) return;
    actionRequest.current = null;
    setBusy(null);
  }

  function clearRetargetResult(): void {
    setRetargetResult(null);
    onResultLoaded?.(null);
    if (errorOwner === "retarget") {
      setError(null);
      setErrorOwner(null);
    }
  }

  async function loadSourceRobot(): Promise<void> {
    if (!sourceChoice || calibration) return;
    const request = beginAction("source-robot");
    setStatus(
      text(
        `Loading source robot ${sourceChoice}…`,
        `正在加载源机器人 ${sourceChoice}…`,
      ),
    );
    try {
      const payload = await loadRobot(sourceChoice, { signal: request.signal });
      if (request.signal.aborted) return;
      setSourceRobot(payload);
      setSourceResult(null);
      onSourceRobotLoaded?.(payload);
      onSourceLoaded?.(null);
      clearRetargetResult();
      setStatus(
        text(
          `Source robot loaded: ${payload.display_name}`,
          `源机器人已加载：${payload.display_name}`,
        ),
      );
    } catch (reason) {
      if (!request.signal.aborted) {
        setError(errorMessage(reason));
        setErrorOwner("source-robot");
      }
    } finally {
      finishAction(request);
    }
  }

  async function loadTargetRobot(): Promise<void> {
    if (!targetChoice || calibration) return;
    const request = beginAction("target-robot");
    setStatus(
      text(
        `Loading target robot ${targetChoice}…`,
        `正在加载目标机器人 ${targetChoice}…`,
      ),
    );
    try {
      const payload = await loadRobot(targetChoice, { signal: request.signal });
      if (request.signal.aborted) return;
      setTargetRobot(payload);
      onTargetRobotLoaded?.(payload);
      clearRetargetResult();
      setStatus(
        text(
          `Target robot loaded: ${payload.display_name}`,
          `目标机器人已加载：${payload.display_name}`,
        ),
      );
    } catch (reason) {
      if (!request.signal.aborted) {
        setError(errorMessage(reason));
        setErrorOwner("target-robot");
      }
    } finally {
      finishAction(request);
    }
  }

  async function receiveSourceResult(
    load: (request: AbortController) => Promise<R2rSourceResult>,
  ): Promise<void> {
    if (!sourceRobot || calibration) return;
    const request = beginAction("source-trajectory");
    setStatus(text("Loading source trajectory…", "正在加载源轨迹…"));
    try {
      const result = await load(request);
      if (request.signal.aborted) return;
      clearRetargetResult();
      setSourceResult(result);
      onSourceLoaded?.(result);
      if (result.suggested_backend === "interaction_mesh") {
        setBackend("interaction_mesh");
      } else if (result.suggested_backend === "newton") {
        setBackend("newton");
      }
      setProgress(1);
      setStatus(
        text(
          `Trajectory loaded: ${result.num_frames} frames @ ${result.framerate.toFixed(1)} fps`,
          `轨迹已加载：${result.num_frames} 帧 @ ${result.framerate.toFixed(1)} fps`,
        ),
      );
    } catch (reason) {
      if (!request.signal.aborted) {
        setError(errorMessage(reason));
        setErrorOwner("source-trajectory");
      }
    } finally {
      finishAction(request);
    }
  }

  function loadLibraryTrajectory(): void {
    if (!sourceRobot || !selectedEntry) return;
    void receiveSourceResult((request) =>
      loadR2rLibraryEntry(
        selectedEntry,
        sourceRobot.name,
        positiveNumber(sourceFps),
        {
          signal: request.signal,
          onUpdate: (job) => {
            if (!request.signal.aborted) {
              setProgress(job.progress ?? 0);
              setStatus(
                job.message || text("Loading source trajectory…", "正在加载源轨迹…"),
              );
            }
          },
        },
      ),
    );
  }

  function uploadTrajectory(files: FileList | null): void {
    const selectedFiles = files ? Array.from(files) : [];
    if (!sourceRobot || !selectedFiles.length) return;
    void receiveSourceResult((request) =>
      uploadR2rTrajectory(
        selectedFiles,
        sourceRobot.name,
        positiveNumber(sourceFps),
        {
          signal: request.signal,
          onUpdate: (job) => {
            if (!request.signal.aborted) {
              setProgress(job.progress ?? 0);
              setStatus(
                job.message || text("Processing source trajectory…", "正在处理源轨迹…"),
              );
            }
          },
        },
      ),
    );
  }

  async function openCalibration(): Promise<void> {
    if (!sourceRobot || !targetRobot || calibration) return;
    const request = beginAction("calibration-open");
    setStatus(text("Loading calibration…", "正在加载标定…"));
    try {
      const session = await getR2rCalibrationSession(
        targetRobot.name,
        sourceRobot.name,
        { signal: request.signal },
      );
      if (request.signal.aborted) return;
      const initial = normalizeCalibrationValues(
        session.joint_limits,
        session.joint_q,
      );
      setCalibration(session);
      setJointQ(initial);
      setJointGeometry({
        jointWorld: session.joint_world,
        groundOffsetZ: session.ground_offset_z,
      });
      setCalibrationBaseline(initial);
      clearRetargetResult();
      referenceCallback.current?.(session.reference);
      setStatus(
        text(
          `Calibration ready: ${session.reference_name}`,
          `标定已就绪：${session.reference_name}`,
        ),
      );
    } catch (reason) {
      if (!request.signal.aborted) {
        setError(errorMessage(reason));
        setErrorOwner("calibration-open");
      }
    } finally {
      finishAction(request);
    }
  }

  async function saveCalibration(): Promise<void> {
    if (!sourceRobot || !targetRobot || !calibration) return;
    calibrationStatusRequest.current?.abort();
    setCheckingCalibration(false);
    const request = beginAction("calibration-save");
    setStatus(text("Saving calibration…", "正在保存标定…"));
    try {
      const safeJointQ = normalizeCalibrationValues(
        calibration.joint_limits,
        jointQ,
      );
      await saveR2rCalibration(
        targetRobot.name,
        sourceRobot.name,
        safeJointQ,
        { signal: request.signal },
      );
      if (request.signal.aborted) return;
      setCalibrationSaved(true);
      setCalibrated(true);
      closeCalibration();
      clearRetargetResult();
      setStatus(text("Calibration saved.", "标定已保存。"));
    } catch (reason) {
      if (!request.signal.aborted) {
        setError(errorMessage(reason));
        setErrorOwner("calibration-save");
      }
    } finally {
      finishAction(request);
    }
  }

  function closeCalibration(cancelled = false): void {
    setCalibration(null);
    setJointQ({});
    setJointGeometry(null);
    setSelectedCalibrationJoint(null);
    setCalibrationBaseline({});
    referenceCallback.current?.(null);
    poseCallback.current?.(null);
    poseWasActive.current = false;
    if (
      errorOwner === "calibration-open" ||
      errorOwner === "calibration-save"
    ) {
      setError(null);
      setErrorOwner(null);
    }
    if (cancelled) setStatus(text("Calibration cancelled.", "标定已取消。"));
  }

  async function retarget(): Promise<void> {
    if (!sourceRobot || !targetRobot || !sourceResult || !calibrated || calibration) return;
    const request = beginAction("retarget");
    clearRetargetResult();
    setStatus(
      text(
        "Retargeting… The first run for a robot can take longer.",
        "正在重定向… 首次处理某个机器人可能需要更长时间。",
      ),
    );
    try {
      const result = await runR2rRetarget(
        {
          source: sourceRobot.name,
          target: targetRobot.name,
          sourceToken: sourceResult.token,
          backend,
          retargetFps: positiveNumber(retargetFps),
        },
        {
          signal: request.signal,
          onUpdate: (job) => {
            if (!request.signal.aborted) {
              setProgress(job.progress ?? 0);
              setStatus(job.message || text("Retargeting…", "正在重定向…"));
            }
          },
        },
      );
      if (request.signal.aborted) return;
      setRetargetResult(result);
      onResultLoaded?.(result);
      setProgress(1);
      setStatus(
        text(
          `Completed: ${result.num_frames} frames @ ${result.source_fps.toFixed(1)} fps`,
          `已完成：${result.num_frames} 帧 @ ${result.source_fps.toFixed(1)} fps`,
        ),
      );
    } catch (reason) {
      if (!request.signal.aborted) {
        setError(errorMessage(reason));
        setErrorOwner("retarget");
      }
    } finally {
      finishAction(request);
    }
  }

  const sourceRobotStep: StepStatus = busy === "source-robot"
    ? { label: text("Loading…", "加载中…"), tone: "info" }
    : errorOwner === "source-robot"
      ? { label: text("Load failed", "加载失败"), tone: "danger" }
      : sourceRobot
        ? { label: sourceRobot.display_name, tone: "success" }
        : { label: text("Not loaded", "未加载"), tone: "neutral" };
  const sourceTrajectoryStep: StepStatus = busy === "source-trajectory"
    ? { label: text("Loading…", "加载中…"), tone: "info" }
    : errorOwner === "source-trajectory"
      ? { label: text("Load failed", "加载失败"), tone: "danger" }
      : sourceResult
        ? {
            label: sourceResult.name || text("Loaded", "已加载"),
            tone: "success",
          }
        : { label: text("Not loaded", "未加载"), tone: "neutral" };
  const targetRobotStep: StepStatus = busy === "target-robot"
    ? { label: text("Loading…", "加载中…"), tone: "info" }
    : errorOwner === "target-robot"
      ? { label: text("Load failed", "加载失败"), tone: "danger" }
      : targetRobot
        ? { label: targetRobot.display_name, tone: "success" }
        : { label: text("Not loaded", "未加载"), tone: "neutral" };
  const calibrationStep: StepStatus = busy === "calibration-open"
    ? { label: text("Opening…", "打开中…"), tone: "info" }
    : busy === "calibration-save"
      ? { label: text("Saving…", "保存中…"), tone: "info" }
      : errorOwner === "calibration-open" ||
          errorOwner === "calibration-save"
        ? { label: text("Calibration failed", "标定失败"), tone: "danger" }
        : calibration
          ? { label: text("Editing…", "编辑中…"), tone: "info" }
          : checkingCalibration
            ? { label: text("Checking…", "检查中…"), tone: "info" }
            : calibrated
              ? { label: text("Ready", "已就绪"), tone: "success" }
              : {
                  label: text("Required", "需要标定"),
                  tone: sourceRobot && targetRobot ? "warning" : "neutral",
                };
  const resultStep: StepStatus = busy === "retarget"
    ? { label: text("Running", "运行中"), tone: "info" }
    : errorOwner === "retarget"
      ? { label: text("Retarget failed", "重定向失败"), tone: "danger" }
      : retargetResult
        ? { label: text("Completed", "已完成"), tone: "success" }
        : { label: text("Not ready", "未就绪"), tone: "neutral" };

  return (
    <InspectorPage title={text("Robot → Robot", "机器人 → 机器人")}>
      <WorkflowPipeline
        label={text("Robot to Robot pipeline", "机器人到机器人流程")}
        steps={pipeline}
        activeIndex={activeStep}
        completedIndex={retargetResult ? 4 : activeStep - 1}
      />
      <div className="flex shrink-0 flex-col">
        <WorkflowStep
          title={text("1. Source robot", "1. 源机器人")}
          status={sourceRobotStep.label}
          statusTone={sourceRobotStep.tone}
          defaultOpen
        >
          <RobotSelect
            label={text("Select source robot", "选择源机器人")}
            robots={robots}
            value={sourceChoice}
            loaded={sourceRobot}
            disabled={busy !== null || calibration !== null}
            onChange={setSourceChoice}
            onLoad={() => void loadSourceRobot()}
            onImport={onOpenRobotLibrary}
          />
        </WorkflowStep>

        <WorkflowStep
          title={text("2. Source trajectory", "2. 源轨迹")}
          status={sourceTrajectoryStep.label}
          statusTone={sourceTrajectoryStep.tone}
        >
          <div className="grid gap-2.5">
            <Field label={text("Robot trajectory library", "机器人轨迹资源库")}>
              <select
                className={fieldClass}
                value={trajectoryChoice}
                disabled={
                  !sourceRobot ||
                  busy !== null ||
                  calibration !== null ||
                  compatibleEntries.length === 0
                }
                onChange={(event) => setTrajectoryChoice(event.target.value)}
              >
                {!compatibleEntries.length && (
                  <option value="">
                    {text(
                      entries.length
                        ? "No trajectories match this source robot"
                        : "No robot trajectories available",
                      entries.length
                        ? "没有与此源机器人匹配的轨迹"
                        : "没有可用的机器人轨迹",
                    )}
                  </option>
                )}
                {compatibleEntries.map((entry) => (
                  <option key={entry.source_path} value={entry.source_path}>
                    {entryLabel(entry, text("Trajectory", "轨迹"))}
                  </option>
                ))}
              </select>
            </Field>
            <div className="grid grid-cols-3 gap-2 max-[420px]:grid-cols-2">
              <Button
                size="sm"
                variant="primary"
                disabled={!sourceRobot || !selectedEntry || busy !== null || calibration !== null}
                onClick={loadLibraryTrajectory}
              >
                {text("Load from library", "从资源库加载")}
              </Button>
              <Button
                size="sm"
                variant="primaryOutline"
                disabled={!sourceRobot || busy !== null || calibration !== null}
                onClick={() => fileInput.current?.click()}
              >
                {text("Upload files", "上传文件")}
              </Button>
              <Button
                size="sm"
                variant="primaryOutline"
                disabled={!sourceRobot || busy !== null || calibration !== null}
                onClick={() => folderInput.current?.click()}
              >
                {text("Upload folder", "上传文件夹")}
              </Button>
              <input
                ref={fileInput}
                className="hidden"
                type="file"
                multiple
                accept=".csv,.pkl,.npz"
                onChange={(event) => {
                  uploadTrajectory(event.currentTarget.files);
                  event.currentTarget.value = "";
                }}
              />
              <input
                ref={folderInput}
                className="hidden"
                type="file"
                multiple
                {...({ webkitdirectory: "" } as React.InputHTMLAttributes<HTMLInputElement>)}
                onChange={(event) => {
                  uploadTrajectory(event.currentTarget.files);
                  event.currentTarget.value = "";
                }}
              />
            </div>
            <Field label={text("Source trajectory FPS", "源轨迹 FPS")}>
              <input
                className={fieldClass}
                inputMode="decimal"
                placeholder={text("Use trajectory FPS", "使用轨迹 FPS")}
                value={sourceFps}
                disabled={!sourceRobot || busy !== null || calibration !== null}
                onChange={(event) => setSourceFps(event.target.value)}
              />
            </Field>
          </div>
        </WorkflowStep>

        <WorkflowStep
          title={text("3. Target robot", "3. 目标机器人")}
          status={targetRobotStep.label}
          statusTone={targetRobotStep.tone}
        >
          <RobotSelect
            label={text("Select target robot", "选择目标机器人")}
            robots={robots}
            value={targetChoice}
            loaded={targetRobot}
            disabled={busy !== null || calibration !== null}
            onChange={setTargetChoice}
            onLoad={() => void loadTargetRobot()}
            onImport={onOpenRobotLibrary}
          />
        </WorkflowStep>

        <WorkflowStep
          title={text("4. Calibration", "4. 标定")}
          status={calibrationStep.label}
          statusTone={calibrationStep.tone}
        >
          <div className="grid gap-2.5">
            <div className="flex items-center justify-between gap-3">
              <span className="min-w-0 truncate text-xs text-muted-foreground">
                {sourceRobot && targetRobot
                  ? `${sourceRobot.display_name} → ${targetRobot.display_name}`
                  : text("Load both robots first", "请先加载两个机器人")}
              </span>
              <Button
                size="sm"
                variant="primary"
                disabled={
                  !sourceRobot ||
                  !targetRobot ||
                  checkingCalibration ||
                  busy !== null ||
                  calibration !== null
                }
                onClick={() => void openCalibration()}
              >
                {calibration
                  ? text("Editing…", "编辑中…")
                  : calibrated
                    ? text("Edit", "编辑")
                    : text("Calibrate", "标定")}
              </Button>
            </div>
            {calibration && (
              <CalibrationEditor
                limits={calibration.joint_limits}
                value={jointQ}
                baseline={calibrationBaseline}
                hasSavedBaseline={Boolean(calibration.has_saved_calibration)}
                reference={calibration.reference}
                robot={targetRobot!}
                display={calibrationDisplay}
                angleUnit={angleUnit}
                selectedJoint={selectedCalibrationJoint}
                disabled={busy !== null}
                saving={busy === "calibration-save"}
                onChange={setJointQ}
                onDisplayChange={publishCalibrationDisplay}
                onAngleUnitChange={setAngleUnit}
                onJointSelected={setSelectedCalibrationJoint}
                onCancel={() => closeCalibration(true)}
                onSave={() => void saveCalibration()}
              />
            )}
            {calibrationSaved && (
              <p className="text-[11px] text-success">
                {text("Calibration saved.", "标定已保存。")}
              </p>
            )}
          </div>
        </WorkflowStep>

        <WorkflowStep
          title={text("5. Result", "5. 结果")}
          status={resultStep.label}
          statusTone={resultStep.tone}
        >
          <div className="grid gap-2.5">
            <div className="grid grid-cols-2 gap-2">
              <Field label={text("Solver", "求解器")}>
                <select
                  className={fieldClass}
                  value={backend}
                  disabled={busy !== null || calibration !== null}
                  onChange={(event) => {
                    const value = event.target.value as R2rBackend;
                    setBackend(value);
                    clearRetargetResult();
                  }}
                >
                  <option value="newton">Newton IK</option>
                  <option value="interaction_mesh">Interaction Mesh</option>
                </select>
              </Field>
              <Field label={text("Retarget FPS", "重定向 FPS")}>
                <input
                  className={fieldClass}
                  inputMode="decimal"
                  placeholder={text("Trajectory FPS", "轨迹 FPS")}
                  value={retargetFps}
                  disabled={busy !== null || calibration !== null}
                  onChange={(event) => {
                    setRetargetFps(event.target.value);
                    clearRetargetResult();
                  }}
                />
              </Field>
            </div>
            <Button variant="primary" size="sm" disabled={Boolean(blockedReason) || busy !== null} onClick={() => void retarget()}>
              {busy === "retarget"
                ? text("Retargeting…", "重定向中…")
                : text("Start Retarget", "开始重定向")}
            </Button>
            {blockedReason && <p className="text-xs leading-[1.4] text-muted-foreground">{blockedReason}</p>}
            {busy && progress > 0 && (
              <div
                className="h-1.5 overflow-hidden rounded-full bg-border-subtle"
                role="progressbar"
                aria-valuenow={Math.round(progress * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div className="h-full bg-primary transition-[width]" style={{ width: `${Math.max(2, progress * 100)}%` }} />
              </div>
            )}
            <p className="min-h-4 text-xs text-muted-foreground" aria-live="polite">{status}</p>
            {retargetResult && (
              <>
                <ResultDiagnostics
                  diagnostics={retargetResult.diagnostics}
                  preset={comparisonPreset}
                  onPresetChange={onComparisonPresetChange}
                />
                <ResultExportControls
                  key={retargetResult.export_token}
                  token={retargetResult.export_token}
                  resultFps={retargetResult.source_fps}
                  hasScene={retargetResult.has_scene}
                  buildUrl={r2rExportUrl}
                />
              </>
            )}
          </div>
        </WorkflowStep>
      </div>
      {error && (
        <p
          className="rounded-md border border-danger-border bg-danger-muted px-2.5 py-2 text-[11px] leading-relaxed break-words text-danger"
          role="alert"
        >
          {error}
        </p>
      )}
    </InspectorPage>
  );
}
