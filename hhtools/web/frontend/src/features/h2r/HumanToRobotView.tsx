import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { Field, fieldClass } from "@/components/Field";
import { CalibrationEditor } from "@/components/CalibrationEditor";
import {
  AssetImportButton,
  type ImportAssetKind,
} from "@/components/AssetImportButton";
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
import {
  getHumanMotionLibrary,
  loadMotionLibraryEntry,
  type MotionLibraryEntry,
  type MotionPayload,
} from "@/features/motion/api";
import {
  getRobotLibrary,
  loadRobot,
  type RobotPayload,
  type RobotSummary,
} from "@/features/robot/api";
import { ResultDiagnostics } from "@/features/result/ResultDiagnostics";
import { ResultExportControls } from "@/features/result/ResultExportControls";
import type { ComparisonPreset } from "@/features/result/comparison";
import type { StageMotionPayload } from "@/stage/types";
import {
  DEFAULT_CALIBRATION_DISPLAY,
  type CalibrationDisplayOptions,
} from "@/stage/calibrationDisplay";
import type { CalibrationInteractionModel } from "@/stage/calibrationInteraction";

import {
  getCalibrationReferences,
  getCalibrationStatus,
  loadScaledPreview,
  previewCalibrationPose,
  proposeCalibration,
  retarget,
  retargetExportUrl,
  saveCalibration,
  startCalibrationSession,
  type CalibrationSession,
  type CalibrationPose,
  type CalibrationProposal,
  type CalibrationStatus,
  type RetargetResult,
  type ScaledPreviewResult,
} from "./api";

type Action =
  | "motion"
  | "robot"
  | "calibration"
  | "proposal"
  | "save"
  | "retarget";
type Backend = "newton" | "interaction_mesh";

interface StepStatus {
  readonly label: string;
  readonly tone: WorkflowStatusTone;
}

export interface HumanToRobotViewProps {
  readonly currentMotion?: StageMotionPayload | null;
  readonly currentRobot?: RobotPayload | null;
  readonly currentResult?: RetargetResult | null;
  readonly onMotionLoaded?: (motion: MotionPayload) => void;
  readonly onRobotLoaded?: (robot: RobotPayload) => void;
  readonly onRetargetResult?: (result: RetargetResult | null) => void;
  readonly onCalibrationReference?: (reference: StageMotionPayload | null) => void;
  readonly onRobotPose?: (pose: CalibrationPose | null) => void;
  readonly onScaledPreview?: (preview: ScaledPreviewResult | null) => void;
  readonly calibrationDisplay?: CalibrationDisplayOptions;
  readonly onCalibrationDisplayChange?: (value: CalibrationDisplayOptions) => void;
  readonly onCalibrationInteraction?: (
    interaction: CalibrationInteractionModel | null,
  ) => void;
  readonly comparisonPreset?: ComparisonPreset;
  readonly onComparisonPresetChange?: (preset: ComparisonPreset) => void;
  readonly forceCalibrationOpen?: boolean;
  readonly forceResultOpen?: boolean;
  readonly onOpenMotionLibrary: () => void;
  readonly onOpenRobotLibrary: () => void;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function motionLabel(entry: MotionLibraryEntry, fallback = "Motion"): string {
  const name =
    entry.stem ||
    entry.sequence_id ||
    entry.label ||
    displayFileName(entry.source_path, fallback);
  return entry.folder_label ? `${entry.folder_label} / ${name}` : name;
}

function positiveNumber(value: string): number | undefined {
  const parsed = Number(value);
  return value.trim() && Number.isFinite(parsed) && parsed > 0
    ? parsed
    : undefined;
}

function Picker({
  label,
  value,
  disabled,
  buttonLabel,
  importKind,
  onImport,
  onChange,
  onLoad,
  children,
}: {
  label: string;
  value: string;
  disabled: boolean;
  buttonLabel: string;
  importKind: ImportAssetKind;
  onImport(): void;
  onChange(value: string): void;
  onLoad(): void;
  children: ReactNode;
}) {
  return (
    <div className="grid gap-2">
      <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-2">
        <select
          className={fieldClass}
          aria-label={label}
          value={value}
          disabled={disabled}
          onChange={(event) => onChange(event.currentTarget.value)}
        >
          {children}
        </select>
        <AssetImportButton kind={importKind} onClick={onImport} />
      </div>
      <Button
        size="sm"
        variant="primary"
        disabled={disabled || !value}
        onClick={onLoad}
      >
        {buttonLabel}
      </Button>
    </div>
  );
}

/**
 * React owns this four-step transaction; FastAPI owns heavy motion/robot data.
 * The optional props make the same view work with App-owned or local inputs.
 */
export function HumanToRobotView({
  currentMotion,
  currentRobot,
  currentResult,
  onMotionLoaded,
  onRobotLoaded,
  onRetargetResult,
  onCalibrationReference,
  onRobotPose,
  onScaledPreview,
  calibrationDisplay: controlledCalibrationDisplay,
  onCalibrationDisplayChange,
  onCalibrationInteraction,
  comparisonPreset,
  onComparisonPresetChange,
  forceCalibrationOpen = false,
  forceResultOpen = false,
  onOpenMotionLibrary,
  onOpenRobotLibrary,
}: HumanToRobotViewProps) {
  const text = useLocaleText();
  const pipeline = [
    text("Motion", "动作"),
    text("Robot", "机器人"),
    text("Calibration", "标定"),
    text("Result", "结果"),
  ];
  const [motionEntries, setMotionEntries] = useState<
    readonly MotionLibraryEntry[]
  >([]);
  const [robotEntries, setRobotEntries] = useState<readonly RobotSummary[]>([]);
  const [references, setReferences] = useState<readonly string[]>([]);
  const [localMotion, setLocalMotion] = useState<StageMotionPayload | null>(
    null,
  );
  const [localRobot, setLocalRobot] = useState<RobotPayload | null>(null);
  const motion = currentMotion === undefined ? localMotion : currentMotion;
  const robot = currentRobot === undefined ? localRobot : currentRobot;

  const [motionPath, setMotionPath] = useState("");
  const [robotName, setRobotName] = useState(currentRobot?.name ?? "");
  const [reference, setReference] = useState(
    currentMotion?.suggested_reference ?? "",
  );
  const [calibration, setCalibration] = useState<CalibrationStatus | null>(
    null,
  );
  const [session, setSession] = useState<CalibrationSession | null>(null);
  const [jointQ, setJointQ] = useState<Record<string, number>>({});
  const [jointGeometry, setJointGeometry] = useState<{
    readonly jointWorld: CalibrationSession["joint_world"];
    readonly groundOffsetZ: number;
  } | null>(null);
  const [angleUnit, setAngleUnit] = useState<CalibrationAngleUnit>("rad");
  const [selectedCalibrationJoint, setSelectedCalibrationJoint] =
    useState<string | null>(null);
  const [calibrationBaseline, setCalibrationBaseline] = useState<
    Record<string, number>
  >({});
  const [proposalValidation, setProposalValidation] = useState<
    CalibrationProposal["validation"] | null
  >(null);
  const [localCalibrationDisplay, setLocalCalibrationDisplay] = useState(
    DEFAULT_CALIBRATION_DISPLAY,
  );
  const calibrationDisplay =
    controlledCalibrationDisplay ?? localCalibrationDisplay;
  const publishCalibrationDisplay =
    onCalibrationDisplayChange ?? setLocalCalibrationDisplay;
  const [checking, setChecking] = useState(false);
  const [busy, setBusy] = useState<Action | null>(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [errorOwner, setErrorOwner] = useState<Action | null>(null);
  const [result, setResult] = useState<RetargetResult | null>(
    currentResult ?? null,
  );
  const [retargetFps, setRetargetFps] = useState("");
  const [backend, setBackend] = useState<Backend>("newton");
  const actionRequest = useRef<AbortController | null>(null);
  const calibrationStatusRequest = useRef<AbortController | null>(null);
  const resultCallback = useRef(onRetargetResult);
  resultCallback.current = onRetargetResult;
  const referenceCallback = useRef(onCalibrationReference);
  referenceCallback.current = onCalibrationReference;
  const poseCallback = useRef(onRobotPose);
  poseCallback.current = onRobotPose;
  const scaledPreviewCallback = useRef(onScaledPreview);
  const interactionCallback = useRef(onCalibrationInteraction);
  scaledPreviewCallback.current = onScaledPreview;
  interactionCallback.current = onCalibrationInteraction;
  const inputKey = `${motion?.token ?? ""}|${robot?.name ?? ""}|${reference}`;
  const previousInputKey = useRef(inputKey);

  useEffect(() => {
    const request = new AbortController();
    void Promise.all([
      getHumanMotionLibrary({ signal: request.signal }),
      getRobotLibrary({ signal: request.signal }),
      getCalibrationReferences({ signal: request.signal }),
    ])
      .then(([motionLibrary, robotLibrary, referenceNames]) => {
        if (request.signal.aborted) return;
        setMotionEntries(motionLibrary.entries);
        setRobotEntries(robotLibrary.robots);
        setReferences(referenceNames);
      })
      .catch((reason: unknown) => {
        if (!request.signal.aborted) {
          setError(errorMessage(reason));
          setErrorOwner(null);
        }
      });
    return () => {
      request.abort();
      actionRequest.current?.abort();
      calibrationStatusRequest.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (currentResult === undefined) return;
    setError(null);
    setErrorOwner(null);
    setResult(currentResult);
  }, [currentResult]);

  useEffect(() => {
    setRobotName(robot?.name ?? "");
  }, [robot?.name]);

  useEffect(() => {
    const suggested = motion?.suggested_reference;
    const next = suggested ||
      (motion
        ? references.includes("smpl")
          ? "smpl"
          : references[0] || ""
        : "");
    setReference(next);
  }, [motion?.token, references]);

  useEffect(() => {
    if (
      motion?.suggested_backend === "newton" ||
      motion?.suggested_backend === "interaction_mesh"
    ) {
      setBackend(motion.suggested_backend);
    }
  }, [motion?.token]);

  // A result belongs to one exact motion/robot/reference tuple. The key starts
  // with the mounted tuple so returning to this view keeps an App-owned result.
  useEffect(() => {
    if (previousInputKey.current === inputKey) return;
    previousInputKey.current = inputKey;
    actionRequest.current?.abort();
    setBusy(null);
    setSession(null);
    setJointQ({});
    setJointGeometry(null);
    setSelectedCalibrationJoint(null);
    setCalibrationBaseline({});
    setProposalValidation(null);
    setResult(null);
    setProgress(0);
    setError(null);
    setErrorOwner(null);
    referenceCallback.current?.(null);
    poseCallback.current?.(null);
    resultCallback.current?.(null);
  }, [inputKey]);

  // For this Web workflow calibration/status is the lightweight preflight.
  useEffect(() => {
    if (!robot || !reference) {
      calibrationStatusRequest.current?.abort();
      setChecking(false);
      return;
    }
    const request = new AbortController();
    calibrationStatusRequest.current?.abort();
    calibrationStatusRequest.current = request;
    setCalibration(null);
    setChecking(true);
    void getCalibrationStatus(robot.name, reference, { signal: request.signal })
      .then((value) => {
        if (!request.signal.aborted) setCalibration(value);
      })
      .catch((reason: unknown) => {
        if (!request.signal.aborted) {
          setError(errorMessage(reason));
          setErrorOwner("calibration");
        }
      })
      .finally(() => {
        if (!request.signal.aborted) setChecking(false);
      });
    return () => request.abort();
  }, [robot, reference]);

  // A scaled preview belongs to one exact calibrated input tuple. Keep the
  // optional visualization out of the retarget transaction and discard stale
  // responses when any owner changes or calibration editing starts.
  useEffect(() => {
    scaledPreviewCallback.current?.(null);
    if (
      !motion?.token ||
      !robot ||
      !reference ||
      checking ||
      !calibration?.calibrated ||
      session
    ) {
      return;
    }

    const request = new AbortController();
    void loadScaledPreview(
      {
        robot: robot.name,
        motion_token: motion.token,
        reference,
      },
      { signal: request.signal },
    )
      .then((preview) => {
        if (!request.signal.aborted) {
          scaledPreviewCallback.current?.(preview);
        }
      })
      .catch((reason: unknown) => {
        if (!request.signal.aborted) {
          scaledPreviewCallback.current?.(null);
          console.warn("scaled preview", errorMessage(reason));
        }
      });

    return () => {
      request.abort();
      scaledPreviewCallback.current?.(null);
    };
  }, [
    calibration?.calibrated,
    checking,
    motion?.token,
    reference,
    robot?.name,
    session,
  ]);

  useEffect(
    () => () => {
      referenceCallback.current?.(null);
      poseCallback.current?.(null);
      scaledPreviewCallback.current?.(null);
      interactionCallback.current?.(null);
    },
    [],
  );

  useEffect(() => {
    if (!session || !jointGeometry) {
      interactionCallback.current?.(null);
      return;
    }
    interactionCallback.current?.({
      jointQ,
      jointLimits: session.joint_limits,
      jointWorld: jointGeometry.jointWorld,
      groundOffsetZ: jointGeometry.groundOffsetZ,
      angleUnit,
      selectedJoint: selectedCalibrationJoint,
      disabled: Boolean(busy),
      onJointChange: (name, value) => {
        setProposalValidation(null);
        setJointQ((current) =>
          setCalibrationJointValue(session.joint_limits, current, name, value),
        );
      },
      onSelectedJointChange: setSelectedCalibrationJoint,
      onAngleUnitChange: setAngleUnit,
    });
  }, [
    angleUnit,
    busy,
    jointGeometry,
    jointQ,
    selectedCalibrationJoint,
    session,
  ]);

  useEffect(() => {
    if (!session || !robot) {
      poseCallback.current?.(null);
      return;
    }
    const request = new AbortController();
    const timer = window.setTimeout(() => {
      void previewCalibrationPose(robot.name, jointQ, {
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
            setErrorOwner("calibration");
          }
        });
    }, 120);
    return () => {
      window.clearTimeout(timer);
      request.abort();
    };
  }, [jointQ, robot, session]);

  async function runAction(
    action: Action,
    work: (signal: AbortSignal) => Promise<void>,
  ) {
    actionRequest.current?.abort();
    const request = new AbortController();
    actionRequest.current = request;
    setBusy(action);
    setError(null);
    setErrorOwner(null);
    try {
      await work(request.signal);
    } catch (reason) {
      if (!request.signal.aborted) {
        setError(errorMessage(reason));
        setErrorOwner(action);
      }
    } finally {
      if (actionRequest.current === request) setBusy(null);
    }
  }

  function clearResult() {
    setResult(null);
    setStatus("");
    setError(null);
    setErrorOwner(null);
    resultCallback.current?.(null);
  }

  function selectMotion() {
    const entry = motionEntries.find((item) => item.source_path === motionPath);
    if (!entry || busy || session) return;
    void runAction("motion", async (signal) => {
      const label = motionLabel(entry, text("Motion", "动作"));
      setStatus(text(`Loading ${label}…`, `正在加载 ${label}…`));
      const payload = await loadMotionLibraryEntry(entry, {
        signal,
        usage: "human_to_robot",
        onUpdate: (job) => {
          setProgress(job.progress ?? 0);
          setStatus(job.message || text("Loading motion…", "正在加载动作…"));
        },
      });
      if (signal.aborted) return;
      setLocalMotion(payload);
      setStatus(text(`Loaded ${payload.name}`, `已加载 ${payload.name}`));
      onMotionLoaded?.(payload);
    });
  }

  function selectRobot() {
    if (!robotName || busy || session) return;
    void runAction("robot", async (signal) => {
      setStatus(text("Loading robot…", "正在加载机器人…"));
      const payload = await loadRobot(robotName, { signal });
      if (signal.aborted) return;
      setLocalRobot(payload);
      setStatus(text(`Loaded ${payload.display_name}`, `已加载 ${payload.display_name}`));
      onRobotLoaded?.(payload);
    });
  }

  function editCalibration() {
    if (!robot || !reference || busy || session) return;
    void runAction("calibration", async (signal) => {
      setStatus(text("Opening calibration…", "正在打开标定…"));
      const value = await startCalibrationSession(
        {
          robot: robot.name,
          reference,
          ...(motion?.token ? { motion_token: motion.token } : {}),
        },
        { signal },
      );
      if (signal.aborted) return;
      clearResult();
      const initial = normalizeCalibrationValues(
        value.joint_limits,
        value.joint_q,
      );
      setSession(value);
      setJointQ(initial);
      setJointGeometry({
        jointWorld: value.joint_world,
        groundOffsetZ: value.ground_offset_z,
      });
      setCalibrationBaseline(initial);
      setProposalValidation(null);
      referenceCallback.current?.(value.reference);
      setStatus(
        text(
          "Edit joint values, then save calibration.",
          "编辑关节值，然后保存标定。",
        ),
      );
    });
  }

  function closeCalibration(cancelled = false) {
    setSession(null);
    setJointQ({});
    setJointGeometry(null);
    setSelectedCalibrationJoint(null);
    setCalibrationBaseline({});
    setProposalValidation(null);
    referenceCallback.current?.(null);
    poseCallback.current?.(null);
    if (cancelled) setStatus(text("Calibration cancelled.", "标定已取消。"));
  }

  function suggestCalibration() {
    if (!robot || !reference || !session || busy) return;
    void runAction("proposal", async (signal) => {
      setStatus(text("Generating calibration proposal…", "正在生成标定建议…"));
      const proposal = await proposeCalibration(
        {
          robot: robot.name,
          reference,
          joint_q: jointQ,
          ...(motion?.token ? { motion_token: motion.token } : {}),
        },
        { signal },
      );
      if (signal.aborted) return;
      setJointQ(
        normalizeCalibrationValues(session.joint_limits, proposal.joint_q),
      );
      setProposalValidation(proposal.validation);
      setStatus(
        proposal.validation.valid
          ? text(
              `Calibration proposal ready · score ${proposal.validation.score.toFixed(2)}. Review and save it.`,
              `标定建议已生成 · 评分 ${proposal.validation.score.toFixed(2)}。请检查后保存。`,
            )
          : text(
              "The proposal still needs adjustment; review highlighted joints.",
              "建议姿态仍需调整，请检查高亮关节。",
            ),
      );
    });
  }

  function persistCalibration() {
    if (!robot || !reference || !session || busy) return;
    calibrationStatusRequest.current?.abort();
    setChecking(false);
    void runAction("save", async (signal) => {
      setStatus(text("Saving calibration…", "正在保存标定…"));
      const safeJointQ = normalizeCalibrationValues(session.joint_limits, jointQ);
      const saved = await saveCalibration(
        {
          robot: robot.name,
          reference,
          joint_q: safeJointQ,
          ...(motion?.token ? { motion_token: motion.token } : {}),
        },
        { signal },
      );
      if (signal.aborted) return;
      setCalibration({ calibrated: true, path: saved.path ?? null });
      closeCalibration();
      setStatus(text("Calibration saved.", "标定已保存。"));
    });
  }

  const blockedReason = useMemo(() => {
    if (!motion?.token) return text("Select a human motion first.", "请先选择人体动作。");
    if (!robot) return text("Select a target robot first.", "请先选择目标机器人。");
    if (!reference) return text("Select a reference pose.", "请选择参考姿势。");
    if (session) {
      return text(
        "Save or cancel the open calibration before retargeting.",
        "开始重定向前，请保存或取消当前标定。",
      );
    }
    if (checking) return text("Checking calibration…", "正在检查标定…");
    if (!calibration?.calibrated) {
      return text("Save calibration before retargeting.", "重定向前请先保存标定。");
    }
    return null;
  }, [calibration?.calibrated, checking, motion?.token, reference, robot, session, text]);

  function startRetarget() {
    if (!motion?.token || !robot || !reference || blockedReason || busy) return;
    void runAction("retarget", async (signal) => {
      clearResult();
      setProgress(0);
      setStatus(
        backend === "newton"
          ? text("Starting Newton IK…", "正在启动 Newton IK…")
          : text("Starting Interaction-Mesh…", "正在启动 Interaction-Mesh…"),
      );
      const value = await retarget(
        {
          robot: robot.name,
          motion_token: motion.token!,
          reference,
          backend,
          retarget_fps: positiveNumber(retargetFps),
        },
        {
          signal,
          onUpdate: (job) => {
            setProgress(job.progress ?? 0);
            setStatus(job.message || text("Retargeting…", "正在重定向…"));
          },
        },
      );
      if (signal.aborted) return;
      setResult(value);
      setProgress(1);
      setStatus(
        text(
          `Completed ${value.num_frames} frames.`,
          `已完成 ${value.num_frames} 帧。`,
        ),
      );
      resultCallback.current?.(value);
    });
  }

  const motionStep: StepStatus = busy === "motion"
    ? { label: text("Loading…", "加载中…"), tone: "info" }
    : errorOwner === "motion"
      ? { label: text("Load failed", "加载失败"), tone: "danger" }
      : motion
        ? { label: motion.name || text("Loaded", "已加载"), tone: "success" }
        : { label: text("Not loaded", "未加载"), tone: "neutral" };
  const robotStep: StepStatus = busy === "robot"
    ? { label: text("Loading…", "加载中…"), tone: "info" }
    : errorOwner === "robot"
      ? { label: text("Load failed", "加载失败"), tone: "danger" }
      : robot
        ? { label: robot.display_name, tone: "success" }
        : { label: text("Not loaded", "未加载"), tone: "neutral" };
  const calibrationStep: StepStatus = busy === "calibration"
    ? { label: text("Opening…", "打开中…"), tone: "info" }
    : busy === "proposal"
      ? { label: text("Proposing…", "生成建议中…"), tone: "info" }
      : busy === "save"
        ? { label: text("Saving…", "保存中…"), tone: "info" }
        : errorOwner === "calibration" ||
            errorOwner === "proposal" ||
            errorOwner === "save"
          ? { label: text("Calibration failed", "标定失败"), tone: "danger" }
          : session
            ? { label: text("Editing…", "编辑中…"), tone: "info" }
            : checking
              ? { label: text("Checking…", "检查中…"), tone: "info" }
              : calibration?.calibrated
                ? {
                    label: calibration.bundled && !calibration.path
                      ? text("Built-in", "内置")
                      : text("Calibrated", "已标定"),
                    tone: "success",
                  }
                : {
                    label: text("Not calibrated", "未标定"),
                    tone: motion && robot && reference ? "warning" : "neutral",
                  };
  const resultStep: StepStatus = busy === "retarget"
    ? { label: text("Retargeting…", "重定向中…"), tone: "info" }
    : errorOwner === "retarget"
      ? { label: text("Retarget failed", "重定向失败"), tone: "danger" }
      : result
        ? { label: text("Ready", "已就绪"), tone: "success" }
        : { label: text("Not ready", "未就绪"), tone: "neutral" };
  const activeIndex = session
    ? 2
    : !motion
      ? 0
      : !robot
        ? 1
        : !calibration?.calibrated
          ? 2
          : 3;
  return (
    <InspectorPage title={text("Human → Robot", "人体 → 机器人")}>
      <WorkflowPipeline
        label={text("Human to Robot pipeline", "人体到机器人流程")}
        steps={pipeline}
        activeIndex={activeIndex}
        completedIndex={result ? 3 : activeIndex - 1}
      />
      <div className="flex shrink-0 flex-col">
        <WorkflowStep
          title={text("1. Motion", "1. 动作")}
          status={motionStep.label}
          statusTone={motionStep.tone}
          defaultOpen
        >
          <Picker
            label={text("Select human motion", "选择人体动作")}
            value={motionPath}
            disabled={Boolean(busy || session)}
            buttonLabel={text("Load motion", "加载动作")}
            importKind="motion"
            onImport={onOpenMotionLibrary}
            onChange={setMotionPath}
            onLoad={selectMotion}
          >
            <option value="">
              {text("Select from Motion Library…", "从动作资源库选择…")}
            </option>
            {motionEntries.map((entry) => (
              <option key={entry.source_path} value={entry.source_path}>
                {motionLabel(entry, text("Motion", "动作"))}
              </option>
            ))}
          </Picker>
        </WorkflowStep>

        <WorkflowStep
          title={text("2. Target robot", "2. 目标机器人")}
          status={robotStep.label}
          statusTone={robotStep.tone}
        >
          <Picker
            label={text("Select target robot", "选择目标机器人")}
            value={robotName}
            disabled={Boolean(busy || session)}
            buttonLabel={text("Load robot", "加载机器人")}
            importKind="robot"
            onImport={onOpenRobotLibrary}
            onChange={setRobotName}
            onLoad={selectRobot}
          >
            <option value="">{text("Select a robot…", "选择机器人…")}</option>
            {robotEntries.map((entry) => (
              <option
                key={entry.name}
                value={entry.name}
                disabled={!entry.has_urdf}
              >
                {entry.display_name} ({entry.num_dof} {text("DoF", "自由度")})
              </option>
            ))}
          </Picker>
        </WorkflowStep>

        <WorkflowStep
          title={text("3. Calibration", "3. 标定")}
          status={calibrationStep.label}
          statusTone={calibrationStep.tone}
          forceOpen={forceCalibrationOpen}
          tutorialAnchor="h2r-calibration"
        >
          <div className="grid gap-2.5">
            <Field label={text("Reference pose", "参考姿势")}>
              <select
                className={fieldClass}
                value={reference}
                disabled={!motion || Boolean(busy || session)}
                onChange={(event) => {
                  const value = event.currentTarget.value;
                  setReference(value);
                }}
              >
                <option value="">—</option>
                {references.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </Field>
            <Button
              size="sm"
              variant="primary"
              disabled={!robot || !reference || checking || Boolean(busy || session)}
              onClick={editCalibration}
            >
              {session
                ? text("Editing…", "编辑中…")
                : calibration?.calibrated
                  ? text("Edit calibration", "编辑标定")
                  : text("Calibrate", "标定")}
            </Button>
            {session && (
              <CalibrationEditor
                limits={session.joint_limits}
                value={jointQ}
                baseline={calibrationBaseline}
                hasSavedBaseline={session.has_saved_calibration}
                reference={session.reference}
                robot={robot!}
                display={calibrationDisplay}
                angleUnit={angleUnit}
                selectedJoint={selectedCalibrationJoint}
                disabled={Boolean(busy)}
                saving={busy === "save"}
                suggesting={busy === "proposal"}
                assistantValidation={proposalValidation}
                onChange={(value) => {
                  setProposalValidation(null);
                  setJointQ(value);
                }}
                onDisplayChange={publishCalibrationDisplay}
                onAngleUnitChange={setAngleUnit}
                onJointSelected={setSelectedCalibrationJoint}
                onCancel={() => closeCalibration(true)}
                onSuggest={suggestCalibration}
                onSave={persistCalibration}
              />
            )}
          </div>
        </WorkflowStep>

        <WorkflowStep
          title={text("4. Result", "4. 结果")}
          status={resultStep.label}
          statusTone={resultStep.tone}
          forceOpen={forceResultOpen}
          tutorialAnchor="h2r-result"
        >
          <div className="grid gap-2.5">
            <div className="grid grid-cols-2 gap-2">
              <Field label={text("Solver", "求解器")}>
                <select
                  className={fieldClass}
                  value={backend}
                  disabled={Boolean(busy || session)}
                  onChange={(event) => {
                    const value = event.currentTarget.value as Backend;
                    setBackend(value);
                    clearResult();
                  }}
                >
                  <option value="newton">Newton IK</option>
                  <option value="interaction_mesh">Interaction-Mesh</option>
                </select>
              </Field>
              <Field label={text("Retarget FPS", "重定向 FPS")}>
                <input
                  className={fieldClass}
                  type="number"
                  min="1"
                  step="1"
                  placeholder={text("Original FPS", "原始 FPS")}
                  value={retargetFps}
                  disabled={Boolean(busy || session)}
                  onChange={(event) => {
                    setRetargetFps(event.currentTarget.value);
                    clearResult();
                  }}
                />
              </Field>
            </div>
            <Button
              variant="primary"
              size="sm"
              disabled={Boolean(blockedReason) || Boolean(busy)}
              onClick={startRetarget}
            >
              {busy === "retarget"
                ? text("Retargeting…", "重定向中…")
                : text("Start Retarget", "开始重定向")}
            </Button>
            {blockedReason && (
              <p className="text-xs text-muted-foreground">{blockedReason}</p>
            )}
            {busy === "retarget" && (
              <div
                className="h-1.5 overflow-hidden rounded-full bg-border-subtle"
                role="progressbar"
                aria-valuenow={Math.round(progress * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div
                  className="h-full bg-primary transition-[width]"
                  style={{ width: `${Math.max(2, progress * 100)}%` }}
                />
              </div>
            )}
            {result && (
              <>
                <ResultDiagnostics
                  diagnostics={result.diagnostics}
                  preset={comparisonPreset}
                  onPresetChange={onComparisonPresetChange}
                />
                <ResultExportControls
                  key={result.export_token}
                  token={result.export_token}
                  resultFps={result.retarget_fps ?? result.source_fps}
                  hasScene={result.has_scene}
                  buildUrl={retargetExportUrl}
                />
              </>
            )}
          </div>
        </WorkflowStep>
      </div>
      {status && (
        <p className="text-xs text-muted-foreground" aria-live="polite">
          {status}
        </p>
      )}
      {error && (
        <p
          className="rounded-md border border-danger-border bg-danger-muted px-2.5 py-2 text-[11px] text-danger break-words"
          role="alert"
        >
          {error}
        </p>
      )}
    </InspectorPage>
  );
}
