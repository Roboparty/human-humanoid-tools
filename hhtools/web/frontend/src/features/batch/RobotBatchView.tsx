import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { WorkflowStep } from "@/components/WorkflowSteps";
import { getR2rCalibrationStatus } from "@/features/r2r/api";
import type { MotionLibraryEntry } from "@/features/motion/api";
import { loadRobot, type RobotPayload, type RobotSummary } from "@/features/robot/api";
import type { JobSnapshot, UploadFile } from "@/lib/api";
import { useLocaleText } from "@/LocaleProvider";

import {
  runR2rBatch,
  uploadR2rBatchInputs,
  type BatchResult,
} from "./api";
import {
  BatchProgress,
  BatchResultPanel,
  CommonBatchSettings,
  EntryList,
  FileImport,
  RobotSelect,
  StatusMessage,
  type CommonBatchSettingsValue,
} from "./BatchParts";
import {
  appendUniqueEntries,
  entryKey,
  optionalNonNegativeNumber,
  optionalPositiveNumber,
  suggestedBackend,
  timeRangeError,
} from "./model";

type CalibrationPhase = "idle" | "checking" | "ready" | "missing" | "error";

const initialSettings: CommonBatchSettingsValue = {
  backend: "newton",
  format: "pkl",
  csvHeader: true,
  retargetFps: "",
  exportFps: "",
  start: "0",
  end: "",
  output: "r2r_batch_export",
};

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function invalidPositive(
  label: string,
  chineseLabel: string,
  value: string,
  text: (english: string, chinese: string) => string,
): string | null {
  return value.trim() && optionalPositiveNumber(value) === undefined
    ? text(`${label} must be a positive number.`, `${chineseLabel}必须是正数。`)
    : null;
}

export function RobotBatchView({
  active,
  robots,
  catalogError,
}: {
  active: boolean;
  robots: readonly RobotSummary[];
  catalogError?: string | null;
}) {
  const text = useLocaleText();
  const [entries, setEntries] = useState<readonly MotionLibraryEntry[]>([]);
  const [sourceChoice, setSourceChoice] = useState("");
  const [targetChoice, setTargetChoice] = useState("");
  const [sourceRobot, setSourceRobot] = useState<RobotPayload | null>(null);
  const [targetRobot, setTargetRobot] = useState<RobotPayload | null>(null);
  const [calibration, setCalibration] = useState<CalibrationPhase>("idle");
  const [settings, setSettings] = useState(initialSettings);
  const [sourceFps, setSourceFps] = useState("50");
  const [action, setAction] = useState<"import" | "source" | "target" | "run" | null>(null);
  const [job, setJob] = useState<JobSnapshot<BatchResult> | null>(null);
  const [completed, setCompleted] = useState<{ jobId: string; result: BatchResult } | null>(null);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState<string | null>(null);
  const operation = useRef<AbortController | null>(null);
  const busy = action !== null;
  const loadedPair = Boolean(
    sourceRobot &&
      targetRobot &&
      sourceChoice === sourceRobot.name &&
      targetChoice === targetRobot.name,
  );

  useEffect(() => () => operation.current?.abort(), []);

  useEffect(() => {
    if (!active) return;
    if (!sourceRobot || !targetRobot || !loadedPair) {
      setCalibration("idle");
      return;
    }
    const request = new AbortController();
    setCalibration("checking");
    void getR2rCalibrationStatus(targetRobot.name, sourceRobot.name, {
      signal: request.signal,
    })
      .then((status) => {
        if (!request.signal.aborted) setCalibration(status.calibrated ? "ready" : "missing");
      })
      .catch(() => {
        if (!request.signal.aborted) setCalibration("error");
      });
    return () => request.abort();
  }, [active, loadedPair, sourceRobot, targetRobot]);

  function resetRunResult(): void {
    setJob(null);
    setCompleted(null);
  }

  function addEntries(incoming: readonly MotionLibraryEntry[]): void {
    const next = appendUniqueEntries(entries, incoming);
    const added = next.length - entries.length;
    setEntries(next);
    setNotice(
      text(
        `${added} added · ${incoming.length - added} duplicates skipped`,
        `已添加 ${added} 项 · 跳过 ${incoming.length - added} 个重复项`,
      ),
    );
    const backend = suggestedBackend(incoming);
    if (backend) setSettings((current) => ({ ...current, backend }));
    setError(null);
    resetRunResult();
  }

  async function importFiles(files: readonly UploadFile[]): Promise<void> {
    if (!files.length || busy) return;
    operation.current?.abort();
    const request = new AbortController();
    operation.current = request;
    setAction("import");
    setError(null);
    setNotice(
      text(
        `Reading ${files.length} uploaded file${files.length === 1 ? "" : "s"}…`,
        `正在读取 ${files.length} 个上传文件…`,
      ),
    );
    try {
      const result = await uploadR2rBatchInputs(files, "auto", {
        signal: request.signal,
        onUpdate: (snapshot) =>
          setNotice(snapshot.message || text("Recognizing trajectories…", "正在识别轨迹…")),
      });
      if (!request.signal.aborted) addEntries(result.entries);
    } catch (reason) {
      if (!request.signal.aborted) setError(errorMessage(reason));
    } finally {
      if (operation.current === request) {
        operation.current = null;
        setAction(null);
      }
    }
  }

  async function loadSelectedRobot(kind: "source" | "target"): Promise<void> {
    const choice = kind === "source" ? sourceChoice : targetChoice;
    if (!choice || busy) return;
    operation.current?.abort();
    const request = new AbortController();
    operation.current = request;
    setAction(kind);
    setError(null);
    setCalibration("idle");
    try {
      const loaded = await loadRobot(choice, { signal: request.signal });
      if (request.signal.aborted) return;
      if (kind === "source") setSourceRobot(loaded);
      else setTargetRobot(loaded);
      setNotice(text(`Loaded ${loaded.display_name}`, `已加载 ${loaded.display_name}`));
      resetRunResult();
    } catch (reason) {
      if (!request.signal.aborted) setError(errorMessage(reason));
    } finally {
      if (operation.current === request) {
        operation.current = null;
        setAction(null);
      }
    }
  }

  const rawTimeRangeError = timeRangeError(settings.start, settings.end);
  const localizedTimeRangeError = rawTimeRangeError
    ? rawTimeRangeError === "Start time cannot be later than end time."
      ? text(rawTimeRangeError, "开始时间不能晚于结束时间。")
      : text(rawTimeRangeError, "请输入有效的非负时间范围。")
    : null;
  const settingsError =
    localizedTimeRangeError ||
    invalidPositive("Source FPS", "源帧率", sourceFps, text) ||
    invalidPositive("Retarget FPS", "重定向帧率", settings.retargetFps, text) ||
    invalidPositive("Export FPS", "导出帧率", settings.exportFps, text);
  const disabledReason = busy
    ? action === "run"
      ? text("An R2R batch task is running.", "R2R 批处理任务正在运行。")
      : text("Finish the current Batch operation first.", "请先完成当前批处理操作。")
    : !entries.length
      ? text("Add at least one source trajectory.", "请至少添加一条源轨迹。")
      : !sourceRobot
        ? text("Load the source robot.", "请加载源机器人。")
        : sourceChoice !== sourceRobot.name
          ? text("Load the selected source robot.", "请加载所选源机器人。")
          : !targetRobot
            ? text("Load the target robot.", "请加载目标机器人。")
            : targetChoice !== targetRobot.name
              ? text("Load the selected target robot.", "请加载所选目标机器人。")
              : calibration === "checking"
                ? text("Checking robot-pair calibration…", "正在检查机器人配对校准…")
                : calibration !== "ready"
                  ? text(
                      "Calibrate this robot pair in Robot → Robot first.",
                      "请先在机器人 → 机器人中校准这一机器人组合。",
                    )
                  : settingsError;

  async function run(): Promise<void> {
    if (disabledReason || !sourceRobot || !targetRobot) return;
    operation.current?.abort();
    const request = new AbortController();
    operation.current = request;
    setAction("run");
    setError(null);
    setCompleted(null);
    setJob({ id: "starting", kind: "r2r_batch", status: "running", progress: 0, clip_progress: 0, message: text("Starting R2R batch…", "正在启动 R2R 批处理…") });
    try {
      const done = await runR2rBatch(
        {
          source: sourceRobot.name,
          target: targetRobot.name,
          entries,
          backend: settings.backend,
          out_dir: settings.output.trim().replace(/\.zip$/i, "") || "r2r_batch_export",
          format: settings.format,
          csv_header: settings.csvHeader,
          source_fps: optionalPositiveNumber(sourceFps),
          retarget_fps: optionalPositiveNumber(settings.retargetFps),
          export_fps: optionalPositiveNumber(settings.exportFps),
          t_start: optionalNonNegativeNumber(settings.start),
          t_end: optionalNonNegativeNumber(settings.end),
        },
        { signal: request.signal, onUpdate: setJob },
      );
      if (!request.signal.aborted) {
        setJob((current) => current ? { ...current, status: "done", progress: 1, clip_progress: 1 } : current);
        setCompleted(done);
        setNotice(
          text(
            `${done.result.written.length} trajectories completed.`,
            `${done.result.written.length} 条轨迹已完成。`,
          ),
        );
      }
    } catch (reason) {
      if (!request.signal.aborted) setError(errorMessage(reason));
    } finally {
      if (operation.current === request) {
        operation.current = null;
        setAction(null);
      }
    }
  }

  const calibrationLabel = !loadedPair
    ? text("Load both robots", "加载两个机器人")
    : calibration === "checking"
      ? text("Checking…", "检查中…")
      : calibration === "ready"
        ? text("Pair ready", "组合已就绪")
        : calibration === "missing"
          ? text("Calibration needed", "需要校准")
          : calibration === "error"
            ? text("Check failed", "检查失败")
            : text("Not checked", "未检查");

  return (
    <div className="flex flex-col">
      <WorkflowStep
        title={text("1. Source trajectories", "1. 源轨迹")}
        status={text(`${entries.length} trajectories`, `${entries.length} 条轨迹`)}
        defaultOpen
      >
        <div className="grid gap-2.5">
          <FileImport
            title={text("Drop trajectory files or a folder", "拖放轨迹文件或文件夹")}
            hint={text("CSV, PKL, NPZ and dataset folders", "CSV、PKL、NPZ 和数据集文件夹")}
            icon="/icons/sidebar/r2r.svg"
            accept=".csv,.pkl,.npz"
            busy={busy}
            onFiles={importFiles}
          />
          <EntryList
            entries={entries}
            kind="robot"
            busy={busy}
            onRemove={(key) => {
              setEntries((current) => current.filter((entry) => entryKey(entry) !== key));
              resetRunResult();
            }}
            onClear={() => {
              setEntries([]);
              resetRunResult();
            }}
          />
        </div>
      </WorkflowStep>

      <WorkflowStep
        title={text("2. Source robot", "2. 源机器人")}
        status={sourceRobot?.display_name ?? text("Not loaded", "未加载")}
        defaultOpen
      >
        <RobotSelect
          label={text("Source robot", "源机器人")}
          robots={robots}
          value={sourceChoice}
          loadedName={sourceRobot?.name}
          busy={busy}
          onChange={setSourceChoice}
          onLoad={() => void loadSelectedRobot("source")}
        />
      </WorkflowStep>

      <WorkflowStep
        title={text("3. Target robot", "3. 目标机器人")}
        status={targetRobot?.display_name ?? text("Not loaded", "未加载")}
        defaultOpen
      >
        <div className="grid gap-2.5">
          <RobotSelect
            label={text("Target robot", "目标机器人")}
            robots={robots}
            value={targetChoice}
            loadedName={targetRobot?.name}
            busy={busy}
            onChange={setTargetChoice}
            onLoad={() => void loadSelectedRobot("target")}
          />
          <p
            className={`text-[11px] ${calibration === "ready" ? "text-success" : calibration === "missing" ? "text-warning" : calibration === "error" ? "text-danger" : "text-muted-foreground"}`}
          >
            {calibrationLabel}
          </p>
        </div>
      </WorkflowStep>

      <WorkflowStep title={text("4. Run settings", "4. 运行设置")} defaultOpen>
        <CommonBatchSettings
          value={settings}
          sourceFps={sourceFps}
          disabled={busy}
          onChange={(patch) => setSettings((current) => ({ ...current, ...patch }))}
          onSourceFpsChange={setSourceFps}
        />
      </WorkflowStep>

      <section className="grid gap-2.5 pt-4">
        <p className="text-xs text-muted-foreground">
          {entries.length
            ? `${entries.length} ${text("trajectories", "条轨迹")} · ${sourceRobot?.display_name ?? text("no source", "无源机器人")} → ${targetRobot?.display_name ?? text("no target", "无目标机器人")}`
            : text("No source trajectories selected.", "尚未选择源轨迹。")}
        </p>
        <Button variant="primary" size="sm" disabled={Boolean(disabledReason)} onClick={() => void run()}>
          {action === "run"
            ? text("Running R2R batch…", "R2R 批处理中…")
            : text("Start R2R batch", "启动 R2R 批处理")}
        </Button>
        {disabledReason && <p className="text-[11px] text-muted-foreground">{disabledReason}</p>}
        <BatchProgress job={job} />
        <StatusMessage error={Boolean(error || catalogError)}>{error || catalogError}</StatusMessage>
        {!error && <StatusMessage>{notice}</StatusMessage>}
        {completed && <BatchResultPanel jobId={completed.jobId} result={completed.result} />}
      </section>
    </div>
  );
}
