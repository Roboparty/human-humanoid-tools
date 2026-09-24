import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { Button } from "@/components/ui/button";
import { WorkflowStep } from "@/components/WorkflowSteps";
import { getCalibrationStatus } from "@/features/h2r/api";
import type { MotionLibraryEntry } from "@/features/motion/api";
import { loadRobot, type RobotPayload, type RobotSummary } from "@/features/robot/api";
import type { JobSnapshot, UploadFile } from "@/lib/api";
import { useLocaleText } from "@/LocaleProvider";

import {
  runHumanBatch,
  uploadHumanBatchInputs,
  type BatchResult,
} from "./api";
import {
  BatchProgress,
  BatchResultPanel,
  CommonBatchSettings,
  EntryList,
  FileImport,
  LibraryPicker,
  RobotSelect,
  StatusMessage,
  type CommonBatchSettingsValue,
} from "./BatchParts";
import {
  appendUniqueEntries,
  entryKey,
  entryReference,
  optionalNonNegativeNumber,
  optionalPositiveNumber,
  suggestedBackend,
  timeRangeError,
} from "./model";

type Compatibility = "checking" | "ready" | "missing" | "error";

interface ReferenceCheck {
  readonly reference: string;
  readonly count: number;
  readonly status: Compatibility;
}

const initialSettings: CommonBatchSettingsValue = {
  backend: "newton",
  format: "pkl",
  csvHeader: true,
  retargetFps: "",
  exportFps: "",
  start: "0",
  end: "",
  output: "batch_export",
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

export function HumanBatchView({
  active,
  library,
  robots,
  entries,
  onEntriesChange,
  catalogError,
}: {
  active: boolean;
  library: readonly MotionLibraryEntry[];
  robots: readonly RobotSummary[];
  entries: readonly MotionLibraryEntry[];
  onEntriesChange(entries: readonly MotionLibraryEntry[]): void;
  catalogError?: string | null;
}) {
  const text = useLocaleText();
  const humanLibrary = useMemo(
    () => library.filter((entry) => entry.asset_kind !== "robot_trajectory"),
    [library],
  );
  const [librarySelection, setLibrarySelection] = useState<ReadonlySet<string>>(new Set());
  const [libraryQuery, setLibraryQuery] = useState("");
  const [robotChoice, setRobotChoice] = useState("");
  const [robot, setRobot] = useState<RobotPayload | null>(null);
  const [checks, setChecks] = useState<readonly ReferenceCheck[]>([]);
  const [settings, setSettings] = useState(initialSettings);
  const [batchSize, setBatchSize] = useState("");
  const [action, setAction] = useState<"import" | "robot" | "run" | null>(null);
  const [job, setJob] = useState<JobSnapshot<BatchResult> | null>(null);
  const [completed, setCompleted] = useState<{ jobId: string; result: BatchResult } | null>(null);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState<string | null>(null);
  const operation = useRef<AbortController | null>(null);
  const entriesRef = useRef(entries);
  entriesRef.current = entries;
  const busy = action !== null;

  const referenceGroups = useMemo(() => {
    const groups = new Map<string, number>();
    for (const entry of entries) {
      const reference = entryReference(entry);
      groups.set(reference, (groups.get(reference) ?? 0) + 1);
    }
    return [...groups].sort(([left], [right]) => left.localeCompare(right));
  }, [entries]);
  const referenceKey = referenceGroups.map(([reference, count]) => `${reference}:${count}`).join("|");

  useEffect(() => () => operation.current?.abort(), []);

  useEffect(() => {
    if (!active) return;
    if (!robot || !referenceGroups.length) {
      setChecks([]);
      return;
    }
    const request = new AbortController();
    setChecks(referenceGroups.map(([reference, count]) => ({ reference, count, status: "checking" })));
    void Promise.all(
      referenceGroups.map(async ([reference, count]): Promise<ReferenceCheck> => {
        try {
          const result = await getCalibrationStatus(robot.name, reference, {
            signal: request.signal,
          });
          return {
            reference,
            count,
            status: result.calibrated ? "ready" : "missing",
          };
        } catch {
          return { reference, count, status: "error" };
        }
      }),
    ).then((next) => {
      if (!request.signal.aborted) setChecks(next);
    });
    return () => request.abort();
    // `referenceKey` is the compact identity of all unique calibration scopes.
  }, [active, robot, referenceKey]);

  function resetRunResult(): void {
    setJob(null);
    setCompleted(null);
  }

  function addEntries(incoming: readonly MotionLibraryEntry[]): void {
    const current = entriesRef.current;
    const next = appendUniqueEntries(current, incoming);
    const added = next.length - current.length;
    onEntriesChange(next);
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

  function addSelectedLibraryEntries(): void {
    addEntries(humanLibrary.filter((entry) => librarySelection.has(entryKey(entry))));
    setLibrarySelection(new Set());
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
      const result = await uploadHumanBatchInputs(files, "auto", {
        signal: request.signal,
        onUpdate: (snapshot) =>
          setNotice(snapshot.message || text("Recognizing clips…", "正在识别动作…")),
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

  async function loadSelectedRobot(): Promise<void> {
    if (!robotChoice || busy) return;
    operation.current?.abort();
    const request = new AbortController();
    operation.current = request;
    setAction("robot");
    setError(null);
    try {
      const loaded = await loadRobot(robotChoice, { signal: request.signal });
      if (!request.signal.aborted) {
        setRobot(loaded);
        setNotice(text(`Loaded ${loaded.display_name}`, `已加载 ${loaded.display_name}`));
        resetRunResult();
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

  const rawTimeRangeError = timeRangeError(settings.start, settings.end);
  const localizedTimeRangeError = rawTimeRangeError
    ? rawTimeRangeError === "Start time cannot be later than end time."
      ? text(rawTimeRangeError, "开始时间不能晚于结束时间。")
      : text(rawTimeRangeError, "请输入有效的非负时间范围。")
    : null;
  const settingsError =
    localizedTimeRangeError ||
    invalidPositive("Retarget FPS", "重定向帧率", settings.retargetFps, text) ||
    invalidPositive("Export FPS", "导出帧率", settings.exportFps, text) ||
    (batchSize.trim() &&
    (!Number.isSafeInteger(Number(batchSize)) || Number(batchSize) < 1 || Number(batchSize) > 256)
      ? text(
          "GPU batch size must be an integer from 1 to 256.",
          "GPU 批大小必须是 1 到 256 之间的整数。",
        )
      : null);
  const checkedReferenceKey = checks
    .map((check) => `${check.reference}:${check.count}`)
    .sort()
    .join("|");
  const checksAreCurrent = checkedReferenceKey === referenceKey;
  const missingReferences = checks.filter((check) => check.status !== "ready");
  const disabledReason = busy
    ? action === "run"
      ? text("A batch task is running.", "批处理任务正在运行。")
      : text("Finish the current Batch operation first.", "请先完成当前批处理操作。")
    : !entries.length
      ? text("Add at least one motion.", "请至少添加一个动作。")
      : !robot
        ? text("Select and load a target robot.", "请选择并加载目标机器人。")
        : robotChoice !== robot.name
          ? text("Load the selected target robot.", "请加载所选目标机器人。")
          : !checksAreCurrent || checks.some((check) => check.status === "checking")
            ? text("Checking calibration compatibility…", "正在检查校准兼容性…")
            : missingReferences.length
              ? text(
                  `Complete calibration for ${missingReferences.map((check) => check.reference.toUpperCase()).join(", ")}.`,
                  `请完成 ${missingReferences.map((check) => check.reference.toUpperCase()).join("、")} 的校准。`,
                )
              : settingsError;

  async function run(): Promise<void> {
    if (disabledReason || !robot) return;
    operation.current?.abort();
    const request = new AbortController();
    operation.current = request;
    setAction("run");
    setError(null);
    setCompleted(null);
    setJob({ id: "starting", kind: "batch", status: "running", progress: 0, clip_progress: 0, message: text("Starting batch task…", "正在启动批处理任务…") });
    try {
      const done = await runHumanBatch(
        {
          robot: robot.name,
          entries,
          reference: "smpl",
          backend: settings.backend,
          out_dir: settings.output.trim().replace(/\.zip$/i, "") || "batch_export",
          format: settings.format,
          csv_header: settings.csvHeader,
          foot_clamp_anti_penetration: false,
          batch_size: batchSize.trim() ? Number(batchSize) : undefined,
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
            `${done.result.written.length} clips completed.`,
            `${done.result.written.length} 个动作已完成。`,
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

  return (
    <div className="flex flex-col">
      <WorkflowStep
        title={text("1. Inputs", "1. 输入")}
        status={text(`${entries.length} clips`, `${entries.length} 个动作`)}
        defaultOpen
      >
        <div className="grid gap-2.5">
          <LibraryPicker
            entries={humanLibrary}
            selection={librarySelection}
            query={libraryQuery}
            disabled={busy}
            onQueryChange={setLibraryQuery}
            onSelectionChange={setLibrarySelection}
            onAdd={addSelectedLibraryEntries}
          />
          <FileImport
            title={text("Drop motion files or a folder", "拖放动作文件或文件夹")}
            hint={text("Files are recognized and cached by FastAPI", "文件将由 FastAPI 识别并缓存")}
            icon="/icons/sidebar/motion.svg"
            busy={busy}
            onFiles={importFiles}
          />
          <EntryList
            entries={entries}
            kind="human"
            busy={busy}
            onRemove={(key) => {
              onEntriesChange(
                entriesRef.current.filter((entry) => entryKey(entry) !== key),
              );
              resetRunResult();
            }}
            onClear={() => {
              onEntriesChange([]);
              resetRunResult();
            }}
          />
        </div>
      </WorkflowStep>

      <WorkflowStep
        title={text("2. Target robot & calibration", "2. 目标机器人与校准")}
        status={robot?.display_name ?? text("Not loaded", "未加载")}
        defaultOpen
      >
        <div className="grid gap-2.5">
          <RobotSelect
            label={text("Target robot", "目标机器人")}
            robots={robots}
            value={robotChoice}
            loadedName={robot?.name}
            busy={busy}
            onChange={setRobotChoice}
            onLoad={() => void loadSelectedRobot()}
          />
          {referenceGroups.length > 0 && (
            <div className="rounded-md border border-border-subtle bg-background">
              {referenceGroups.map(([reference, count]) => {
                const status = checks.find((check) => check.reference === reference)?.status;
                return (
                  <div key={reference} className="flex min-h-9 items-center justify-between gap-3 border-b border-border-subtle px-2.5 py-1.5 text-[11px] last:border-b-0">
                    <span>
                      <strong className="text-foreground">{reference.toUpperCase()}</strong> · {count} {text("clips", "个动作")}
                    </span>
                    <span className={status === "ready" ? "text-success" : status === "missing" ? "text-warning" : status === "error" ? "text-danger" : "text-muted-foreground"}>
                      {!robot
                        ? text("Load robot", "加载机器人")
                        : status === "checking"
                          ? text("Checking…", "检查中…")
                          : status === "ready"
                            ? text("Ready", "就绪")
                            : status === "error"
                              ? text("Check failed", "检查失败")
                              : text("Calibration needed", "需要校准")}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </WorkflowStep>

      <WorkflowStep title={text("3. Run settings", "3. 运行设置")} defaultOpen>
        <CommonBatchSettings
          value={settings}
          batchSize={batchSize}
          disabled={busy}
          onChange={(patch) => setSettings((current) => ({ ...current, ...patch }))}
          onBatchSizeChange={setBatchSize}
        />
        <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">
          {settings.backend === "newton"
            ? text(
                "Newton uses GPU chunks; leave batch size empty for automatic tuning.",
                "Newton 使用 GPU 分块；批大小留空时将自动调整。",
              )
            : text(
                "Interaction-Mesh processes clips sequentially.",
                "Interaction-Mesh 将依次处理各个动作。",
              )}
        </p>
      </WorkflowStep>

      <section className="grid gap-2.5 pt-4">
        <p className="text-xs text-muted-foreground">
          {entries.length
            ? `${entries.length} ${text("clips", "个动作")} → ${robot?.display_name ?? text("no target", "无目标机器人")} → ${(settings.output || "batch_export").replace(/\.zip$/i, "")}.zip`
            : text("No inputs selected.", "尚未选择输入。")}
        </p>
        <Button variant="primary" size="sm" disabled={Boolean(disabledReason)} onClick={() => void run()}>
          {action === "run"
            ? text("Running batch…", "批处理中…")
            : text("Start H2R batch", "启动 H2R 批处理")}
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
